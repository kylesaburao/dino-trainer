#!/usr/bin/env node
'use strict';
// Dino trainer CLI (plain Node, no npm dependencies).
//
//   node trainer/cli.js train  [--target 10000] [--resume [--force]] [--workers N]
//                              [--time-budget MIN] [--generations N] [--dir runs/main] ...
//   node trainer/cli.js eval   [--dir runs/main] [--which best|current] [--model model.js] [--target N]
//   node trainer/cli.js bench  [--seconds 3] [--workers N]
//   node trainer/cli.js export [--dir runs/main] [--which best|current] [--out model.js]
//
// Training is OpenAI-ES (antithetic sampling, centred ranks, Adam) over the
// shared policy MLP, with candidates evaluated in parallel on worker_threads.
const fs = require('fs');
const path = require('path');
const { CFG, createRng } = require('../core.js');
const { FORMAT, OBS_DIM, numParams, initParams } = require('../policy.js');
const { runEpisode, median, trainSeed, VALIDATION_SEEDS, EVAL_SEEDS } = require('./env.js');
const { gaussian, esGradient, adamStep } = require('./es.js');
const { createPool, defaultWorkers } = require('./pool.js');
const { ROOT, coreHash, writeAtomic, writeJsonAtomic, readJson, acquireLock } = require('./store.js');

const CHECKPOINT_FORMAT = 'dino-es-checkpoint-v1';

// Defaults for a fresh run. Everything here is saved in the checkpoint's config.
const DEFAULTS = {
  target: 10000,          // score cap per episode and the held-out stop target
  hidden: [16],           // MLP hidden layer sizes (tanh)
  actionRepeat: 1,        // frames each decision is held
  popSize: 128,           // candidates per generation (even: antithetic pairs)
  episodes: 4,            // training episodes per candidate (shared seeds per generation)
  sigma: 0.1,             // perturbation std
  lr: 0.03,               // Adam step size
  weightDecay: 0.005,
  checkpointEvery: 5,     // generations between checkpoints
  evalEvery: 5,           // generations between validation evals
  generations: Infinity,  // stop once this many generations are done in total
  seed: 1,                // trainer RNG seed (init, noise seeds, training seeds)
};
// Settings that change the parameter layout or the policy's behaviour cannot change on resume.
const STRUCTURAL = ['hidden', 'actionRepeat'];

const FLAG_TO_KEY = {
  '--target': ['target', Number], '--hidden': ['hidden', (v) => v.split(',').map(Number)],
  '--action-repeat': ['actionRepeat', Number], '--pop': ['popSize', Number], '--episodes': ['episodes', Number],
  '--sigma': ['sigma', Number], '--lr': ['lr', Number], '--weight-decay': ['weightDecay', Number],
  '--checkpoint-every': ['checkpointEvery', Number], '--eval-every': ['evalEvery', Number],
  '--generations': ['generations', Number], '--seed': ['seed', Number],
};

function parseArgs(argv) {
  const out = { cmd: argv[0], config: {}, opts: {} };
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    const takesValue = () => {
      const v = argv[++i];
      if (v === undefined || v.startsWith('--')) throw new Error(`${a} needs a value`);
      return v;
    };
    if (FLAG_TO_KEY[a]) {
      const [key, parse] = FLAG_TO_KEY[a];
      out.config[key] = parse(takesValue());
    } else if (['--resume', '--force', '--overwrite'].includes(a)) {
      out.opts[a.slice(2)] = true;
    } else if (['--workers', '--time-budget', '--dir', '--which', '--model', '--out', '--seconds'].includes(a)) {
      out.opts[a.slice(2).replace(/-(\w)/g, (m, c) => c.toUpperCase())] = takesValue();
    } else {
      throw new Error(`unknown argument ${a}`);
    }
  }
  return out;
}

const layersOf = (config) => [OBS_DIM, ...config.hidden, 3];
const ts = () => new Date().toISOString();
const log = (...a) => console.log(`[${ts()}]`, ...a);
const fmt = (x) => (Number.isFinite(x) ? Math.round(x * 10) / 10 : x);

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------
// Score fixed seeds with one model (params), in parallel when a pool is given.
async function scoreSeeds(pool, model, seeds, target) {
  if (!pool) return seeds.map(s => runEpisode(model, s, target).score);
  const setup = { theta: model.params, layers: model.layers, actionRepeat: model.actionRepeat, sigma: 0, target };
  const res = await pool.run(setup, seeds.map(s => ({ noiseSeed: 0, sign: 0, seeds: [s] })));
  return res.map(r => r.scores[0]);
}

const summary = (scores) => ({ median: median(scores), min: Math.min(...scores), max: Math.max(...scores) });

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------
async function train({ config: cli, opts }) {
  const dir = path.resolve(opts.dir || path.join(ROOT, 'runs', 'main'));
  fs.mkdirSync(dir, { recursive: true });
  const release = acquireLock(dir);
  process.on('exit', release);

  const ckptFile = path.join(dir, 'checkpoint.json');
  const bestFile = path.join(dir, 'best.json');
  const hash = coreHash();
  let config, st;

  if (opts.resume) {
    if (!fs.existsSync(ckptFile)) throw new Error(`--resume: no checkpoint at ${ckptFile}`);
    const ck = readJson(ckptFile);
    if (ck.format !== CHECKPOINT_FORMAT) throw new Error(`unsupported checkpoint format ${ck.format}`);
    if (ck.coreHash !== hash) {
      if (!opts.force) {
        throw new Error(`core hash mismatch: checkpoint was trained on ${ck.coreHash.slice(0, 12)}, current core/policy is ${hash.slice(0, 12)}. ` +
          'The simulation or observations changed; pass --force to resume anyway.');
      }
      log(`WARNING: core hash mismatch (${ck.coreHash.slice(0, 12)} -> ${hash.slice(0, 12)}), continuing because of --force`);
    }
    for (const k of STRUCTURAL) {
      if (k in cli && JSON.stringify(cli[k]) !== JSON.stringify(ck.config[k])) throw new Error(`--${k} cannot change on resume`);
    }
    config = { ...ck.config, ...cli };
    if (config.generations === null) config.generations = Infinity;
    st = {
      generation: ck.generation, theta: Float64Array.from(ck.theta),
      adam: { t: ck.adam.t, m: Float64Array.from(ck.adam.m), v: Float64Array.from(ck.adam.v) },
      rngState: ck.rngState, best: ck.best, cumulativeMs: ck.cumulativeMs, lastValidated: ck.lastValidated ?? -1,
      heldOut: ck.heldOut || null,
    };
    log(`resuming ${ckptFile} at generation ${st.generation}`);
  } else {
    if (fs.existsSync(ckptFile) && !opts.overwrite) {
      throw new Error(`${ckptFile} exists; use --resume to continue it or --overwrite to start over`);
    }
    config = { ...DEFAULTS, ...cli };
    if (config.popSize % 2) throw new Error('--pop must be even (antithetic pairs)');
    const rng = createRng(config.seed);
    const layers = layersOf(config);
    const theta = initParams(layers, gaussian(rng));
    st = {
      generation: 0, theta,
      adam: { t: 0, m: new Float64Array(theta.length), v: new Float64Array(theta.length) },
      rngState: rng.getState(), best: null, cumulativeMs: 0, lastValidated: -1, heldOut: null,
    };
    for (const f of ['metrics.jsonl', 'metrics.csv']) fs.rmSync(path.join(dir, f), { force: true });
  }

  const layers = layersOf(config);
  const workers = Number(opts.workers) || defaultWorkers();
  const budgetMs = opts.timeBudget !== undefined ? Number(opts.timeBudget) * 60000 : Infinity;
  const startedAt = Date.now();
  let lastCommitAt = startedAt;
  const committedMs = () => st.cumulativeMs + (Date.now() - lastCommitAt);

  log(`train: dir=${dir} workers=${workers} params=${numParams(layers)} layers=${layers.join('-')} core=${hash.slice(0, 12)}`);
  log(`watch it live: node trainer/dashboard.js  ->  http://127.0.0.1:8080  (run from ${ROOT})`);
  log(`config ${JSON.stringify(config)}${budgetMs < Infinity ? ` time-budget=${opts.timeBudget}min` : ''}`);

  // Only state committed at the end of a generation is ever saved, so a signal
  // mid-generation saves the last completed generation.
  function saveCheckpoint(reason) {
    writeJsonAtomic(ckptFile, {
      format: CHECKPOINT_FORMAT, coreHash: hash, savedAt: ts(), reason,
      generation: st.generation, config: { ...config, generations: Number.isFinite(config.generations) ? config.generations : null },
      layers, theta: Array.from(st.theta),
      adam: { t: st.adam.t, m: Array.from(st.adam.m), v: Array.from(st.adam.v) },
      rngState: st.rngState, best: st.best, lastValidated: st.lastValidated, heldOut: st.heldOut,
      cumulativeMs: committedMs(),
    });
    log(`checkpoint saved: generation ${st.generation} (${reason}) -> ${ckptFile}`);
  }

  let stopping = false;
  function stopNow(reason, code) {
    if (stopping) return;
    stopping = true;
    log(`stopping: ${reason}`);
    try { saveCheckpoint(reason); } finally { release(); process.exit(code); }
  }
  process.on('SIGINT', () => stopNow('SIGINT', 130));
  process.on('SIGTERM', () => stopNow('SIGTERM', 143));
  if (budgetMs < Infinity) setTimeout(() => stopNow('time budget reached (mid-generation)', 0), budgetMs).unref();

  const pool = createPool(workers);
  const metricsJsonl = path.join(dir, 'metrics.jsonl');
  const metricsCsv = path.join(dir, 'metrics.csv');
  const CSV_COLS = ['generation', 'time', 'genMs', 'cumulativeMs', 'fitMean', 'fitMedian', 'fitMax', 'fitMin',
    'valMedian', 'valMin', 'valMax', 'bestValMedian', 'frames', 'fps', 'sigma', 'lr', 'thetaNorm'];
  if (!fs.existsSync(metricsCsv)) fs.writeFileSync(metricsCsv, CSV_COLS.join(',') + '\n');

  async function validate() {
    const model = { layers, params: st.theta, actionRepeat: config.actionRepeat };
    const scores = await scoreSeeds(pool, model, VALIDATION_SEEDS, config.target);
    const s = summary(scores);
    st.lastValidated = st.generation;
    if (!st.best || s.median > st.best.valMedian) {
      st.best = { generation: st.generation, valMedian: s.median, valMin: s.min, valMax: s.max, params: Array.from(st.theta) };
      writeJsonAtomic(bestFile, {
        format: FORMAT, layers, actionRepeat: config.actionRepeat, params: st.best.params,
        generation: st.generation, validation: s, coreHash: hash, config,
      });
    }
    return s;
  }

  let stopReason = null;
  while (!stopReason) {
    if (st.generation >= config.generations) { stopReason = `reached ${config.generations} generations`; break; }
    if (Date.now() - startedAt >= budgetMs) { stopReason = 'time budget reached'; break; }

    const t0 = Date.now();
    const rng = createRng(st.rngState);
    const half = config.popSize / 2;
    const noiseSeeds = Array.from({ length: half }, () => Math.floor(rng() * 4294967296));
    const episodeSeeds = Array.from({ length: config.episodes }, () => trainSeed(rng));
    const jobs = [];
    for (const s of noiseSeeds) jobs.push({ noiseSeed: s, sign: 1, seeds: episodeSeeds }, { noiseSeed: s, sign: -1, seeds: episodeSeeds });
    const setup = { theta: st.theta, layers, actionRepeat: config.actionRepeat, sigma: config.sigma, target: config.target };
    const results = await pool.run(setup, jobs);

    // Fitness: mean capped score, using fractional distance for a finer signal.
    const fitness = results.map(r => r.distances.reduce((a, d) => a + Math.min(d * CFG.scoreCoef, config.target), 0) / r.distances.length);
    const frames = results.reduce((a, r) => a + r.frames, 0);
    const grad = esGradient(noiseSeeds, fitness, st.theta.length, config.sigma);
    const theta = Float64Array.from(st.theta);
    const adam = { t: st.adam.t, m: Float64Array.from(st.adam.m), v: Float64Array.from(st.adam.v) };
    adamStep(theta, grad, adam, config.lr, config.weightDecay);

    // Commit (synchronous: a signal handler cannot observe a half-updated state).
    const genMs = Date.now() - t0;
    st.theta = theta;
    st.adam = adam;
    st.rngState = rng.getState();
    st.generation++;
    st.cumulativeMs += Date.now() - lastCommitAt;
    lastCommitAt = Date.now();

    let val = null;
    if (st.generation % config.evalEvery === 0) val = await validate();
    if (val && val.median >= config.target) {
      const heldOut = summary(await scoreSeeds(pool, { layers, params: st.best.params, actionRepeat: config.actionRepeat }, EVAL_SEEDS, config.target));
      st.heldOut = { generation: st.best.generation, target: config.target, ...heldOut };
      log(`held-out eval of best (gen ${st.best.generation}): median ${heldOut.median} min ${heldOut.min} max ${heldOut.max}`);
      if (heldOut.median >= config.target) stopReason = `held-out median ${heldOut.median} >= target ${config.target}`;
    }

    const sorted = [...fitness].sort((a, b) => a - b);
    const row = {
      generation: st.generation, time: ts(), genMs, cumulativeMs: st.cumulativeMs,
      fitMean: fitness.reduce((a, b) => a + b, 0) / fitness.length, fitMedian: median(sorted),
      fitMax: sorted[sorted.length - 1], fitMin: sorted[0],
      valMedian: val ? val.median : null, valMin: val ? val.min : null, valMax: val ? val.max : null,
      bestValMedian: st.best ? st.best.valMedian : null,
      frames, fps: Math.round(frames / (genMs / 1000)), sigma: config.sigma, lr: config.lr,
      thetaNorm: Math.sqrt(st.theta.reduce((a, x) => a + x * x, 0)),
    };
    fs.appendFileSync(metricsJsonl, JSON.stringify(row) + '\n');
    fs.appendFileSync(metricsCsv, CSV_COLS.map(k => row[k] ?? '').join(',') + '\n');
    log(`gen ${row.generation}: fit mean ${fmt(row.fitMean)} max ${fmt(row.fitMax)}` +
      (val ? ` | val median ${val.median} (best ${st.best.valMedian} @ gen ${st.best.generation})` : '') +
      ` | ${frames} frames in ${(genMs / 1000).toFixed(1)}s (${row.fps} fps)`);

    if (st.generation % config.checkpointEvery === 0) saveCheckpoint('periodic');
  }

  if (st.lastValidated !== st.generation) {
    const val = await validate();
    log(`final validation: median ${val.median} (best ${st.best.valMedian} @ gen ${st.best.generation})`);
  }
  saveCheckpoint(stopReason);
  await pool.terminate();
  release();
  log(`done: ${stopReason}; generation ${st.generation}, cumulative wall ${(st.cumulativeMs / 60000).toFixed(1)} min`);
}

// ---------------------------------------------------------------------------
// Model loading for eval / export
// ---------------------------------------------------------------------------
function loadModel(opts) {
  if (opts.model) {
    const file = path.resolve(opts.model);
    delete require.cache[file];
    const m = require(file);
    return { model: m, source: file, target: undefined };
  }
  const dir = path.resolve(opts.dir || path.join(ROOT, 'runs', 'main'));
  const which = opts.which || 'best';
  if (which === 'best') {
    const file = path.join(dir, 'best.json');
    if (!fs.existsSync(file)) throw new Error(`no ${file} yet (run train until a validation eval)`);
    const b = readJson(file);
    return { model: b, source: file, target: b.config.target, meta: { generation: b.generation, validation: b.validation, coreHash: b.coreHash } };
  }
  if (which === 'current') {
    const file = path.join(dir, 'checkpoint.json');
    const ck = readJson(file);
    return {
      model: { format: FORMAT, layers: ck.layers, actionRepeat: ck.config.actionRepeat, params: ck.theta },
      source: file, target: ck.config.target, meta: { generation: ck.generation, coreHash: ck.coreHash },
    };
  }
  throw new Error(`--which must be best or current`);
}

async function evalCmd({ config: cli, opts }) {
  const { model, source, target: savedTarget } = loadModel(opts);
  const target = cli.target || savedTarget || DEFAULTS.target;
  const workers = Number(opts.workers) || defaultWorkers();
  const pool = workers > 1 ? createPool(Math.min(workers, EVAL_SEEDS.length)) : null;
  const t0 = Date.now();
  const scores = await scoreSeeds(pool, model, EVAL_SEEDS, target);
  if (pool) await pool.terminate();
  const s = summary(scores);
  const record = { time: ts(), source: path.relative(ROOT, source), generation: model.generation ?? null, target, seeds: EVAL_SEEDS.length, ...s, scores };
  console.log(`held-out eval (${EVAL_SEEDS.length} seeds, cap ${target}) of ${record.source}:`);
  console.log(`  median ${s.median}  min ${s.min}  max ${s.max}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  console.log(`  scores ${scores.join(' ')}`);
  if (!opts.model) fs.appendFileSync(path.join(path.dirname(source), 'evals.jsonl'), JSON.stringify(record) + '\n');
}

function exportCmd({ opts }) {
  const { model, source, meta } = loadModel(opts);
  const out = path.resolve(opts.out || path.join(ROOT, 'model.js'));
  const data = {
    format: FORMAT, layers: model.layers, actionRepeat: model.actionRepeat || 1,
    generation: meta && meta.generation, validation: meta && meta.validation, coreHash: meta && meta.coreHash,
    exportedAt: ts(), source: path.relative(ROOT, source), params: Array.from(model.params),
  };
  if (data.coreHash && data.coreHash !== coreHash()) console.warn('WARNING: exporting a model trained against a different core/policy hash');
  const text = `// AI model for the dino game. Generated by \`node trainer/cli.js export\`; do not edit.
// Loads as a classic <script> (sets window.DINO_MODEL) and as a Node require().
(function (root) {
var DINO_MODEL = ${JSON.stringify(data)};
root.DINO_MODEL = DINO_MODEL;
if (typeof module !== 'undefined' && module.exports) module.exports = DINO_MODEL;
})(typeof globalThis !== 'undefined' ? globalThis : this);
`;
  writeAtomic(out, text);
  console.log(`exported ${data.source} (generation ${data.generation}, ${data.params.length} params) -> ${path.relative(process.cwd(), out) || out}`);
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------
async function benchCmd({ opts }) {
  const { benchSingle } = require('./bench.js');
  const seconds = Number(opts.seconds) || 3;
  const workers = Number(opts.workers) || defaultWorkers();
  console.log(`bench: ${seconds}s per measurement; workers detected: ${defaultWorkers()} (using ${workers})`);
  const sim = benchSingle('sim', seconds * 1000);
  console.log(`  1 core, simulation only:     ${Math.round(sim.fps).toLocaleString()} frames/s`);
  const pol = benchSingle('policy', seconds * 1000);
  console.log(`  1 core, simulation + policy: ${Math.round(pol.fps).toLocaleString()} frames/s`);
  const { Worker } = require('worker_threads');
  const per = await Promise.all(Array.from({ length: workers }, () => new Promise((resolve, reject) => {
    const w = new Worker(path.join(__dirname, 'bench.js'), { workerData: { mode: 'policy', ms: seconds * 1000 } });
    w.once('message', resolve);
    w.once('error', reject);
  })));
  const total = per.reduce((a, r) => a + r.fps, 0);
  console.log(`  ${workers} workers, simulation + policy: ${Math.round(total).toLocaleString()} frames/s total, ` +
    `${Math.round(total / workers).toLocaleString()} per worker`);
  console.log(JSON.stringify({ seconds, workersDetected: defaultWorkers(), workers, simFps: Math.round(sim.fps), policyFps: Math.round(pol.fps), parallelFps: Math.round(total) }));
}

// ---------------------------------------------------------------------------
const COMMANDS = { train, eval: evalCmd, bench: benchCmd, export: exportCmd };

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(e.message);
    process.exit(2);
  }
  const cmd = COMMANDS[args.cmd];
  if (!cmd) {
    console.error('usage: node trainer/cli.js <train|eval|bench|export> [options]  (see the header of trainer/cli.js)');
    process.exit(2);
  }
  await cmd(args);
}

main().catch((e) => {
  console.error(`error: ${e.message}`);
  process.exit(1);
});
