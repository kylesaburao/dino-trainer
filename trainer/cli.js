#!/usr/bin/env node
'use strict';
// Dino trainer CLI (plain Node, no npm dependencies).
//
//   node trainer/cli.js train  [--target 10000] [--resume [--force]] [--workers N]
//                              [--time-budget MIN] [--generations N] [--dir runs/main] ...
//   node trainer/cli.js eval   [--dir runs/main] [--which best|current] [--model model.js] [--target N]
//                              [--seeds 25] [--seed-offset 0] [--wasm]
//   node trainer/cli.js bench  [--seconds 3] [--workers N] [--hidden 16] [--wasm]
//   node trainer/cli.js export [--dir runs/main] [--which best|current] [--out model.js]
//                              (also writes model.css next to it: the same weights for index-css.html)
//   node trainer/cli.js export-css [--model model.js]   regenerate model.css from an existing model.js
//   node trainer/cli.js report [--dir runs/main]        regenerate PROGRESS.md's status block
//   node trainer/cli.js finish [--dir runs/main] [--no-commit] [--skip-browser]
//   node trainer/cli.js session [train options]          train, then finish (one detached command)
//
// Training-only curriculum (off by default; validation, held-out eval, the stop
// check, eval and export always play full games from 0 to --target):
//   --window POINTS       training episodes cover a window of this many points;
//                         fitness = mean over episodes of (score - start) / span, capped at 1,
//                         where span = the episode's cap - start (see --start-zero-cap)
//   --start-zero-frac F   share of training episodes that start at 0 (default 0.5)
//   --start-min POINTS    lowest late start level (default 2000)
//   --start-dist log|points  late starts log-uniform in [start-min, target - window]
//                         (default), or uniform in points on [start-min, target) with
//                         the window clipped at target (span = target - start)
//   --start-zero-cap P    start-0 episodes end at P points (default: --window as first set)
//   --window-adapt        every --window-adapt-every generations (default 5), on the pass
//                         rates pooled since the last decision: window /1.5 while <30% of
//                         start-0 episodes passed; once >60% did (or there are none),
//                         window x1.5 if >60% of late windows were passed, /1.5 if <30%;
//                         clamped to [5000, target] (a shrink never grows the window).
//                         Needs --start-dist points. The window and counts are checkpointed.
//   --fitness-agg mean|geomean  per-candidate fitness over its episodes: mean (default) or
//                         geometric mean of max(progress, 1/span), so a failing stage
//                         (e.g. the opening) dominates. geomean needs --window.
// These can be changed (or the curriculum turned on/off) on --resume.
//
// Selection and stopping:
//   --select median|count   best = highest validation median (default), or highest
//                         (seeds reaching target, mean ln(1+score)); at full count a tie
//                         also replaces best (prefer the newest all-pass θ)
//   --stop-rule median|allpass[:N]  held-out check when the validation median reaches
//                         target, stop on held-out median >= target (default); or when
//                         every validation seed passes, stop when all N (default 25)
//                         held-out seeds pass, played in rounds of --workers seeds and
//                         abandoned at the first round with a failure.
//                         Either rule never re-tests a best already tested at this target.
//   --val-seeds N         validation seeds (default 25; the first 25 are always the same)
//   --eval-when-fit-max   also validate in any generation whose best candidate reached
//                         target (with the curriculum: passed a window ending at target)
//   --fold-validation     validate every generation inside the generation's worker batch;
//                         this scores the pre-update θ, so it is labelled generation g-1
// ES:
//   --rank-ties index|mid   centred-rank tie breaking (default index; mid gives tied
//                         antithetic pairs zero weight)
//   --decay coupled|decoupled  coupled (default): --weight-decay is L2 inside Adam's
//                         gradient; decoupled (AdamW): θ *= 1 - lr * weight-decay after
//                         each step, so use a larger value (~0.01; 0.1 slowed the climb)
//   --stall-rule          if >50% of antithetic pairs tie, skip the Adam step and widen
//                         sigma x1.5 for the next generation (at most 4x); reset when not.
//                         Kept only to reproduce Run A; not recommended: in 900-generation
//                         replays (32,32, decoupled wd 0.01) it skipped 65-70% of steps and
//                         escaped the plateau in 0 of 3 seeds.
//   --sigma-scale none|fanin  per-parameter sigma: fanin scales layer l's weights by
//                         sqrt(44 / fanIn_l) (biases keep sigma)
//   --sigma-anneal        sigma *= 0.8 (floor --sigma-min, default 0.02, at most every 10
//                         generations) while >25% of start-0 candidate episodes end below
//                         200 points but the latest validation min is >= 200. The current
//                         sigma is checkpointed; an explicit --sigma on resume overrides it.
//   --pair-episodes       each antithetic pair plays its own episode seeds (and starts);
//                         gradient weights are pair fitness differences / std (ARS V1)
// Speed:
//   --wasm                run the policy network on a WebAssembly SIMD kernel (trainer/wasm-net.js)
//                         in the training, validation and held-out workers. Decisions are
//                         bit-identical to policy.js (test/wasm-net.test.js), so results do not
//                         change; fails at startup if the runtime has no wasm SIMD. Also
//                         accepted by eval and bench. The browser always uses policy.js.
// Boolean options can be turned off on --resume with --no-<option>. Everything except
// --hidden, --action-repeat and --game can change on --resume.
//
// train --auto-export re-exports model.js (and refreshes the report) on every
// new best validation score; --log-every SEC throttles per-generation log lines.
//
// Training is OpenAI-ES (antithetic sampling, centred ranks, Adam) over the
// shared policy MLP, with candidates evaluated in parallel on worker_threads.
const fs = require('fs');
const path = require('path');
const { CFG, createRng, GAME_VERSIONS } = require('../core.js');
const { FORMAT, OBS_DIM, numParams, initParams } = require('../policy.js');
const { modelToCss } = require('./css-model.js');
const { runEpisode, median, windowFitness, validationSeeds, evalSeeds } = require('./env.js');
const { gaussian, sigmaVector, esGradient, adamStep } = require('./es.js');
const {
  valSummary, isBetter, parseStopRule, wantHeldOut, sequentialAllPass, tiedPairs, stallUpdate, annealUpdate,
  gatedAdaptWindow, emptyAdaptAcc, geomeanFitness, splitFolded, clopperPearson, drawGeneration,
} = require('./rules.js');
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
  gameVersion: 'v2',      // game rules trained and evaluated on (core.js GAME_VERSIONS)
  window: 0,              // curriculum: training episode window in points (0 = off: full games)
  startZeroFrac: 0.5,     // curriculum: share of training episodes starting at 0
  startMin: 2000,         // curriculum: lowest late start level (natural speed has settled by here)
  startDist: 'log',       // curriculum: late start distribution (log | points)
  startZeroCap: null,     // curriculum: end of start-0 episodes (null: the window when first set)
  windowAdapt: false,     // curriculum: adapt the window to the late-window pass rate
  windowAdaptEvery: 5,    // curriculum: generations between window adaptation decisions
  fitnessAgg: 'mean',     // curriculum: per-candidate episode aggregate (mean | geomean)
  select: 'median',       // best-model selection (median | count)
  stopRule: 'median',     // held-out stop rule (median | allpass[:N])
  valSeeds: 25,           // validation seeds
  evalWhenFitMax: false,  // also validate when a candidate reached the target
  foldValidation: false,  // validate every generation inside the training batch
  rankTies: 'index',      // centred-rank ties (index | mid)
  decay: 'coupled',       // weight decay (coupled L2 | decoupled AdamW)
  stallRule: false,       // skip the step and widen sigma when most pairs tie
  sigmaScale: 'none',     // per-parameter sigma (none | fanin)
  sigmaAnneal: false,     // shrink sigma while perturbed candidates crash in the opening
  sigmaMin: 0.02,         // sigma-anneal floor
  pairEpisodes: false,    // per-pair episode seeds with pair-difference weights
  wasm: false,            // network on the wasm SIMD kernel (same decisions, faster)
};
// Keys added after the first checkpoints: backfilled from DEFAULTS when resuming
// checkpoints that predate them (their defaults reproduce the old behaviour).
const BACKFILL = ['window', 'startZeroFrac', 'startMin', 'startDist', 'startZeroCap', 'windowAdapt', 'select', 'stopRule',
  'valSeeds', 'evalWhenFitMax', 'foldValidation', 'rankTies', 'decay', 'stallRule', 'sigmaScale', 'sigmaAnneal', 'sigmaMin', 'pairEpisodes',
  'windowAdaptEvery', 'fitnessAgg', 'wasm'];
// Settings that change the parameter layout or the policy's behaviour cannot change on resume.
const STRUCTURAL = ['hidden', 'actionRepeat', 'gameVersion'];

const FLAG_TO_KEY = {
  '--target': ['target', Number], '--hidden': ['hidden', (v) => v.split(',').map(Number)],
  '--action-repeat': ['actionRepeat', Number], '--pop': ['popSize', Number], '--episodes': ['episodes', Number],
  '--sigma': ['sigma', Number], '--lr': ['lr', Number], '--weight-decay': ['weightDecay', Number],
  '--checkpoint-every': ['checkpointEvery', Number], '--eval-every': ['evalEvery', Number],
  '--generations': ['generations', Number], '--seed': ['seed', Number], '--game': ['gameVersion', String],
  '--window': ['window', Number], '--start-zero-frac': ['startZeroFrac', Number], '--start-min': ['startMin', Number],
  '--start-dist': ['startDist', String], '--start-zero-cap': ['startZeroCap', Number],
  '--select': ['select', String], '--stop-rule': ['stopRule', String], '--val-seeds': ['valSeeds', Number],
  '--rank-ties': ['rankTies', String], '--decay': ['decay', String], '--sigma-scale': ['sigmaScale', String],
  '--sigma-min': ['sigmaMin', Number], '--window-adapt-every': ['windowAdaptEvery', Number],
  '--fitness-agg': ['fitnessAgg', String],
};
// Boolean config options: --x sets true, --no-x sets false.
const BOOL_TO_KEY = {
  '--window-adapt': 'windowAdapt', '--eval-when-fit-max': 'evalWhenFitMax', '--fold-validation': 'foldValidation',
  '--stall-rule': 'stallRule', '--sigma-anneal': 'sigmaAnneal', '--pair-episodes': 'pairEpisodes',
  '--wasm': 'wasm',
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
    } else if (BOOL_TO_KEY[a]) {
      out.config[BOOL_TO_KEY[a]] = true;
    } else if (a.startsWith('--no-') && BOOL_TO_KEY['--' + a.slice(5)]) {
      out.config[BOOL_TO_KEY['--' + a.slice(5)]] = false;
    } else if (['--resume', '--force', '--overwrite', '--auto-export', '--no-commit', '--skip-browser'].includes(a)) {
      out.opts[a.slice(2).replace(/-(\w)/g, (m, c) => c.toUpperCase())] = true;
    } else if (['--workers', '--time-budget', '--dir', '--which', '--model', '--out', '--seconds', '--log-every', '--seeds', '--seed-offset'].includes(a)) {
      out.opts[a.slice(2).replace(/-(\w)/g, (m, c) => c.toUpperCase())] = takesValue();
    } else {
      throw new Error(`unknown argument ${a}`);
    }
  }
  return out;
}

const oneOf = (c, key, flag, values) => {
  if (!values.includes(c[key])) throw new Error(`${flag} must be one of ${values.join(', ')}, got ${c[key]}`);
};

// The curriculum is off when window is 0/absent; when on, late starts must be at levels
// where natural speed has settled, and (log starts) every window must fit below target.
function checkCurriculum(c) {
  if (!(c.window > 0)) return;
  oneOf(c, 'startDist', '--start-dist', ['log', 'points']);
  if (!(c.startZeroFrac >= 0 && c.startZeroFrac <= 1)) throw new Error('--start-zero-frac must be in [0, 1]');
  if (!(c.startMin >= 2000)) throw new Error('--start-min must be >= 2000 (below that, natural speed is still ramping)');
  if (c.startDist === 'log' && c.startZeroFrac < 1 && c.startMin + c.window > c.target) {
    throw new Error(`--start-min + --window (${c.startMin + c.window}) must not exceed --target (${c.target})`);
  }
  if (c.startDist === 'points' && c.startZeroFrac < 1 && c.startMin >= c.target) throw new Error('--start-min must be below --target');
  if (c.window > c.target) throw new Error('--window must not exceed --target');
  if (!(c.startZeroCap > 0 && c.startZeroCap <= c.target)) throw new Error('--start-zero-cap must be in (0, target]');
  if (c.windowAdapt && c.startDist !== 'points') throw new Error('--window-adapt needs --start-dist points');
}

function checkConfig(c) {
  if (!GAME_VERSIONS.includes(c.gameVersion)) throw new Error(`--game must be one of ${GAME_VERSIONS.join(', ')}`);
  oneOf(c, 'select', '--select', ['median', 'count']);
  parseStopRule(c.stopRule);
  oneOf(c, 'rankTies', '--rank-ties', ['index', 'mid']);
  oneOf(c, 'decay', '--decay', ['coupled', 'decoupled']);
  oneOf(c, 'sigmaScale', '--sigma-scale', ['none', 'fanin']);
  if (!(Number.isInteger(c.valSeeds) && c.valSeeds > 0)) throw new Error('--val-seeds must be a positive integer');
  if (c.sigmaAnneal && !(c.sigmaMin > 0)) throw new Error('--sigma-min must be > 0');
  oneOf(c, 'fitnessAgg', '--fitness-agg', ['mean', 'geomean']);
  if (c.fitnessAgg === 'geomean' && !(c.window > 0)) throw new Error('--fitness-agg geomean needs the curriculum (--window)');
  if (!(Number.isInteger(c.windowAdaptEvery) && c.windowAdaptEvery > 0)) throw new Error('--window-adapt-every must be a positive integer');
  checkCurriculum(c);
}

const layersOf = (config) => [OBS_DIM, ...config.hidden, 3];
const ts = () => new Date().toISOString();
const log = (...a) => console.log(`[${ts()}]`, ...a);
const fmt = (x) => (Number.isFinite(x) ? Math.round(x * 10) / 10 : x);

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------
// Score fixed seeds with one model (params), in parallel when a pool is given.
// wasm: run the network on trainer/wasm-net.js (same scores).
async function scoreSeeds(pool, model, seeds, target, wasm = false) {
  if (!pool) {
    const makeAgent = wasm ? require('./wasm-net.js').wasmAgents(model) : undefined;
    return seeds.map(s => runEpisode(model, s, target, undefined, makeAgent).score);
  }
  const setup = { theta: model.params, layers: model.layers, actionRepeat: model.actionRepeat, gameVersion: model.gameVersion || 'v1', sigma: 0, target, wasm };
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
      const saved = k === 'gameVersion' ? ck.config[k] || 'v1' : ck.config[k];
      if (k in cli && JSON.stringify(cli[k]) !== JSON.stringify(saved)) throw new Error(`--${k} cannot change on resume`);
    }
    const backfill = Object.fromEntries(BACKFILL.filter(k => !(k in ck.config)).map(k => [k, DEFAULTS[k]]));
    config = { ...ck.config, gameVersion: ck.config.gameVersion || 'v1', ...backfill, ...cli };
    if (config.generations === null) config.generations = Infinity;
    st = {
      generation: ck.generation, theta: Float64Array.from(ck.theta),
      adam: { t: ck.adam.t, m: Float64Array.from(ck.adam.m), v: Float64Array.from(ck.adam.v) },
      rngState: ck.rngState, best: ck.best, cumulativeMs: ck.cumulativeMs, lastValidated: ck.lastValidated ?? -1,
      heldOut: ck.heldOut || null,
      // Adaptive state (absent in older checkpoints): explicit flags override the saved values.
      sigma: 'sigma' in cli || ck.sigma == null ? config.sigma : ck.sigma,
      stallMult: ck.stallMult ?? 1, annealedAt: ck.annealedAt ?? null, lastValMin: ck.lastValMin ?? null,
      window: 'window' in cli || ck.window == null ? config.window : ck.window,
      adaptAcc: ck.adaptAcc ?? emptyAdaptAcc(),
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
      sigma: config.sigma, stallMult: 1, annealedAt: null, lastValMin: null, window: config.window,
      adaptAcc: emptyAdaptAcc(),
    };
    for (const f of ['metrics.jsonl', 'metrics.csv']) fs.rmSync(path.join(dir, f), { force: true });
  }
  // The start-0 episode cap is fixed when the curriculum is first set, so it stays put
  // while --window-adapt moves the window.
  if (config.window > 0 && config.startZeroCap == null) config.startZeroCap = config.window;

  const layers = layersOf(config);
  const workers = Number(opts.workers) || defaultWorkers();
  const budgetMs = opts.timeBudget !== undefined ? Number(opts.timeBudget) * 60000 : Infinity;
  const startedAt = Date.now();
  let lastCommitAt = startedAt;
  const committedMs = () => st.cumulativeMs + (Date.now() - lastCommitAt);

  checkConfig(config);
  if (config.wasm) require('./wasm-net.js').assertAvailable();
  const stopRule = parseStopRule(config.stopRule);
  const valSeedList = validationSeeds(config.valSeeds);
  log(`train: dir=${dir} game=${config.gameVersion} workers=${workers} params=${numParams(layers)} layers=${layers.join('-')} core=${hash.slice(0, 12)}`);
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
      sigma: st.sigma, stallMult: st.stallMult, annealedAt: st.annealedAt, lastValMin: st.lastValMin, window: st.window,
      adaptAcc: st.adaptAcc, cumulativeMs: committedMs(), workers,
    });
    if (reason !== 'periodic') log(`checkpoint saved: generation ${st.generation} (${reason}) -> ${ckptFile}`);
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
    'valMedian', 'valMin', 'valMax', 'bestValMedian', 'frames', 'fps', 'sigma', 'lr', 'thetaNorm',
    'valCount', 'valMeanLog', 'valGeneration', 'valFrames', 'bestValCount', 'tiedPairs', 'stepSkipped', 'sigmaEff', 'window'];
  if (!fs.existsSync(metricsCsv)) fs.writeFileSync(metricsCsv, CSV_COLS.join(',') + '\n');
  // A resumed run keeps its file's header (older runs have fewer columns).
  const csvCols = fs.readFileSync(metricsCsv, 'utf8').split('\n', 1)[0].split(',');
  const modelOf = (params) => ({ layers, params, actionRepeat: config.actionRepeat, gameVersion: config.gameVersion });

  // Record a validation of params (labelled generation gen); replaces best if it is better.
  function recordValidation(scores, params, gen) {
    const s = valSummary(scores, config.target);
    st.lastValidated = gen;
    st.lastValMin = s.min;
    if (isBetter(s, st.best, config.select, scores.length)) {
      st.best = { generation: gen, valMedian: s.median, valMin: s.min, valMax: s.max, valCount: s.count, valMeanLog: s.meanLog, params: Array.from(params) };
      writeJsonAtomic(bestFile, {
        format: FORMAT, layers, actionRepeat: config.actionRepeat, gameVersion: config.gameVersion, params: st.best.params,
        generation: gen, validation: s, coreHash: hash, config,
      });
      if (opts.autoExport) {
        const out = exportModel({ dir, which: 'best' });
        log(`new best (validation median ${s.median}, count ${s.count}/${scores.length}, gen ${gen}): exported ${out.file}`);
        require('./ops.js').writeReport(dir);
      }
    }
    return { ...s, generation: gen };
  }
  async function validate() {
    const scores = await scoreSeeds(pool, modelOf(st.theta), valSeedList, config.target, config.wasm);
    return recordValidation(scores, st.theta, st.generation);
  }

  // Held-out check of best. Returns a stop reason or null; records the attempt in st.heldOut.
  async function heldOutCheck() {
    const best = st.best;
    const attempt = ((st.heldOut && st.heldOut.attempt) || 0) + 1;
    const seeds = evalSeeds(stopRule.seeds);
    const score = (chunk) => scoreSeeds(pool, modelOf(best.params), chunk, config.target, config.wasm);
    if (stopRule.kind === 'median') {
      const scores = await score(seeds);
      const s = summary(scores);
      const passed = scores.filter(x => x >= config.target).length;
      st.heldOut = { generation: best.generation, target: config.target, ...s, seeds: seeds.length, attempt, played: scores.length, passed };
      log(`held-out eval of best (gen ${best.generation}): median ${s.median} min ${s.min} max ${s.max}`);
      return s.median >= config.target ? `held-out median ${s.median} >= target ${config.target}` : null;
    }
    const r = await sequentialAllPass(score, seeds, workers, config.target);
    st.heldOut = {
      generation: best.generation, target: config.target, ...summary(r.scores), seeds: seeds.length, attempt,
      played: r.played, passed: r.passed, firstFailSeed: r.firstFailSeed,
    };
    log(`held-out attempt ${attempt} of best (gen ${best.generation}): ${r.passed}/${r.played} played seeds passed` +
      (r.pass ? ` - all ${seeds.length} pass` : `, first failure seed ${r.firstFailSeed} (aborted after ${r.played} of ${seeds.length})`));
    return r.pass ? `held-out ${r.passed}/${seeds.length} seeds reached target ${config.target}` : null;
  }

  const logEveryMs = (opts.logEvery !== undefined ? Number(opts.logEvery) : 15) * 1000;
  let lastGenLog = 0;
  let stopReason = null;
  while (!stopReason) {
    if (st.generation >= config.generations) { stopReason = `reached ${config.generations} generations`; break; }
    if (Date.now() - startedAt >= budgetMs) { stopReason = 'time budget reached'; break; }

    const t0 = Date.now();
    const rng = createRng(st.rngState);
    const half = config.popSize / 2;
    const cur = config.window > 0 ? { ...config, window: st.window } : null;
    const { noiseSeeds, pairEp } = drawGeneration(rng, config, cur);
    const jobs = [];
    for (let k = 0; k < half; k++) {
      const { seeds, starts, caps } = pairEp[k];
      for (const sign of [1, -1]) jobs.push(starts ? { noiseSeed: noiseSeeds[k], sign, seeds, starts, caps } : { noiseSeed: noiseSeeds[k], sign, seeds });
    }
    // Folded validation: the pre-update θ on the validation seeds, full games (sign 0 = θ itself).
    if (config.foldValidation) for (const s of valSeedList) jobs.push({ noiseSeed: 0, sign: 0, seeds: [s] });
    const sigmaEff = st.sigma * st.stallMult;
    const sigma = sigmaVector(layers, sigmaEff, config.sigmaScale);
    const setup = { theta: st.theta, layers, actionRepeat: config.actionRepeat, gameVersion: config.gameVersion, sigma, target: config.target, wasm: config.wasm };
    const { train: results, val: valResults } = splitFolded(await pool.run(setup, jobs), config.popSize);

    // Fitness: mean capped score, using fractional distance for a finer signal.
    // With the curriculum: mean (or --fitness-agg geomean) progress through each episode's
    // span, in [0, 1]; the start-0/late pass counts feed --window-adapt.
    let latePassed = 0, lateTotal = 0, zeroPassed = 0, zeroTotal = 0, reachedTarget = false;
    const zeroStartScores = [];
    const fitness = results.map((r, i) => {
      const { starts, caps } = pairEp[i >> 1];
      if (!starts) {
        zeroStartScores.push(...r.scores);
        return r.distances.reduce((a, d) => a + Math.min(d * CFG.scoreCoef, config.target), 0) / r.distances.length;
      }
      const tally = (f, j) => {
        if (starts[j] === 0) { zeroStartScores.push(r.scores[j]); zeroTotal++; if (f === 1) zeroPassed++; }
        else { lateTotal++; if (f === 1) latePassed++; }
        if (f === 1 && caps[j] >= config.target) reachedTarget = true;
      };
      if (config.fitnessAgg === 'geomean') {
        const progress = r.distances.map((d, j) => windowFitness(d, starts[j], caps[j] - starts[j]));
        progress.forEach(tally);
        return geomeanFitness(progress, caps.map((c, j) => c - starts[j]));
      }
      return r.distances.reduce((a, d, j) => {
        const f = windowFitness(d, starts[j], caps[j] - starts[j]);
        tally(f, j);
        return a + f;
      }, 0) / r.distances.length;
    });
    const frames = results.reduce((a, r) => a + r.frames, 0);
    const valFrames = valResults.reduce((a, r) => a + r.frames, 0);
    const tied = tiedPairs(fitness);
    const stall = config.stallRule ? stallUpdate(st.stallMult, tied / half) : { skip: false, stallMult: st.stallMult };
    const theta = Float64Array.from(st.theta);
    const adam = { t: st.adam.t, m: Float64Array.from(st.adam.m), v: Float64Array.from(st.adam.v) };
    if (!stall.skip) {
      const grad = esGradient(noiseSeeds, fitness, st.theta.length, sigma,
        { ties: config.rankTies, weights: config.pairEpisodes ? 'pairdiff' : 'ranks' });
      adamStep(theta, grad, adam, config.lr, config.weightDecay, config.decay);
    }

    // Commit (synchronous: a signal handler cannot observe a half-updated state).
    const genMs = Date.now() - t0;
    const preTheta = st.theta;
    st.theta = theta;
    st.adam = adam;
    st.rngState = rng.getState();
    st.generation++;
    st.stallMult = stall.stallMult;
    if (cur && config.windowAdapt) {
      const acc = st.adaptAcc;
      st.adaptAcc = { zeroPassed: acc.zeroPassed + zeroPassed, zeroTotal: acc.zeroTotal + zeroTotal,
        latePassed: acc.latePassed + latePassed, lateTotal: acc.lateTotal + lateTotal };
      if (st.generation % config.windowAdaptEvery === 0) {
        st.window = gatedAdaptWindow(st.window, st.adaptAcc, config.target);
        st.adaptAcc = emptyAdaptAcc();
      }
    }
    let val = null;
    if (config.foldValidation) val = recordValidation(valResults.map(r => r.scores[0]), preTheta, st.generation - 1);
    st.cumulativeMs += Date.now() - lastCommitAt;
    lastCommitAt = Date.now();

    const sorted = [...fitness].sort((a, b) => a - b);
    const fitMax = sorted[sorted.length - 1];
    if (!config.foldValidation) {
      const fitHit = config.evalWhenFitMax && (cur ? reachedTarget : fitMax >= config.target);
      if (st.generation % config.evalEvery === 0 || fitHit) val = await validate();
    }
    if (config.sigmaAnneal) {
      const a = annealUpdate({ sigma: st.sigma, annealedAt: st.annealedAt },
        { zeroStartScores, valMin: st.lastValMin, generation: st.generation, sigmaMin: config.sigmaMin });
      if (a.sigma !== st.sigma) log(`sigma anneal: ${st.sigma} -> ${a.sigma} at gen ${st.generation}`);
      st.sigma = a.sigma;
      st.annealedAt = a.annealedAt;
    }
    if (wantHeldOut(stopRule, val, st.best, st.heldOut, config.target, valSeedList.length)) stopReason = await heldOutCheck();

    const row = {
      generation: st.generation, time: ts(), genMs, cumulativeMs: st.cumulativeMs,
      fitMean: fitness.reduce((a, b) => a + b, 0) / fitness.length, fitMedian: median(sorted),
      fitMax, fitMin: sorted[0],
      valMedian: val ? val.median : null, valMin: val ? val.min : null, valMax: val ? val.max : null,
      bestValMedian: st.best ? st.best.valMedian : null,
      frames, fps: Math.round((frames + valFrames) / (genMs / 1000)), sigma: st.sigma, lr: config.lr,
      thetaNorm: Math.sqrt(st.theta.reduce((a, x) => a + x * x, 0)),
      valCount: val ? val.count : null, valMeanLog: val ? val.meanLog : null, valGeneration: val ? val.generation : null,
      valFrames, bestValCount: st.best ? st.best.valCount ?? null : null,
      tiedPairs: tied, stepSkipped: stall.skip, sigmaEff, window: cur ? cur.window : null,
    };
    fs.appendFileSync(metricsJsonl, JSON.stringify(row) + '\n');
    fs.appendFileSync(metricsCsv, csvCols.map(k => row[k] ?? '').join(',') + '\n');
    if (val || Date.now() - lastGenLog >= logEveryMs || stopReason) {
      lastGenLog = Date.now();
      const fitFmt = cur ? (x) => x.toFixed(3) : fmt; // curriculum fitness is window progress in [0, 1]
      log(`gen ${row.generation}: fit mean ${fitFmt(row.fitMean)} max ${fitFmt(row.fitMax)}` +
      (val ? ` | val median ${val.median} count ${val.count}/${valSeedList.length}${config.foldValidation ? ` (θ of gen ${val.generation})` : ''}` +
        ` (best ${st.best.valMedian} @ gen ${st.best.generation})` : '') +
      (config.stallRule || config.rankTies !== 'index' ? ` | tied ${tied}/${half}${stall.skip ? ' step skipped' : ''}` : '') +
      (cur && config.windowAdapt ? ` | window ${st.window}` : '') +
      ` | ${frames} frames in ${(genMs / 1000).toFixed(1)}s (${row.fps} fps)`);
    }

    if (st.generation % config.checkpointEvery === 0) saveCheckpoint('periodic');
  }

  if (st.lastValidated !== st.generation) {
    const val = await validate();
    log(`final validation: median ${val.median} count ${val.count}/${valSeedList.length} (best ${st.best.valMedian} @ gen ${st.best.generation})`);
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
      model: { format: FORMAT, layers: ck.layers, actionRepeat: ck.config.actionRepeat, gameVersion: ck.config.gameVersion || 'v1', params: ck.theta },
      source: file, target: ck.config.target, meta: { generation: ck.generation, coreHash: ck.coreHash },
    };
  }
  throw new Error(`--which must be best or current`);
}

// Held-out eval (--seeds held-out seeds, default 25, starting at index --seed-offset,
// default 0); appends to the run's evals.jsonl. The train stop check uses the first
// N (default 25) held-out seeds and, under allpass, may retry until they pass, so a
// certification eval of such a run should skip them: --seed-offset 25. Returns the
// record, with the seeds that reached the target (passed) and the two-sided 95%
// Clopper-Pearson interval for the pass rate.
async function evalHeldOut(opts, targetOverride, wasm = false) {
  const { model, source, target: savedTarget } = loadModel(opts);
  const target = targetOverride || savedTarget || DEFAULTS.target;
  const workers = Number(opts.workers) || defaultWorkers();
  const offset = opts.seedOffset !== undefined ? Number(opts.seedOffset) : 0;
  const seeds = evalSeeds(offset + (opts.seeds !== undefined ? Number(opts.seeds) : 25)).slice(offset);
  if (wasm) require('./wasm-net.js').assertAvailable();
  const pool = workers > 1 ? createPool(Math.min(workers, seeds.length)) : null;
  const t0 = Date.now();
  const scores = await scoreSeeds(pool, model, seeds, target, wasm);
  if (pool) await pool.terminate();
  const s = summary(scores);
  const passed = scores.filter(x => x >= target).length;
  const record = { time: ts(), source: path.relative(ROOT, source), generation: model.generation ?? null, target, seeds: seeds.length, seedOffset: offset, ...s, passed, passCI95: clopperPearson(passed, seeds.length), scores };
  record.seconds = (Date.now() - t0) / 1000;
  if (!opts.model) fs.appendFileSync(path.join(path.dirname(source), 'evals.jsonl'), JSON.stringify(record) + '\n');
  return record;
}

async function evalCmd({ config: cli, opts }) {
  const record = await evalHeldOut(opts, cli.target, !!cli.wasm);
  const ci = record.passCI95;
  console.log(`held-out eval (${record.seeds} seeds${record.seedOffset ? ` from index ${record.seedOffset}` : ''}, cap ${record.target}) of ${record.source}:`);
  console.log(`  median ${record.median}  min ${record.min}  max ${record.max}  (${record.seconds.toFixed(1)}s)`);
  console.log(`  passed ${record.passed}/${record.seeds}; pass rate 95% Clopper-Pearson [${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`);
  console.log(`  scores ${record.scores.join(' ')}`);
}

// Write model.js (atomically) from a run's best.json or checkpoint. Returns { file, data }.
function exportModel(opts) {
  const { model, source, meta } = loadModel(opts);
  const out = path.resolve(opts.out || path.join(ROOT, 'model.js'));
  const data = {
    format: FORMAT, layers: model.layers, actionRepeat: model.actionRepeat || 1, gameVersion: model.gameVersion || 'v1',
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
  writeCss(out, data);
  return { file: path.relative(ROOT, out), data };
}

// model.css (the CSS-evaluated network for index-css.html) lives next to its model.js.
function writeCss(modelFile, data) {
  const out = path.join(path.dirname(modelFile), 'model.css');
  writeAtomic(out, modelToCss(data));
  return path.relative(ROOT, out);
}

function exportCssCmd({ opts }) {
  const file = path.resolve(opts.model || path.join(ROOT, 'model.js'));
  const data = require(file);
  const out = writeCss(file, data);
  console.log(`exported ${path.relative(ROOT, file)} (generation ${data.generation}, ${data.params.length} params) -> ${out}`);
}

function exportCmd({ opts }) {
  const { file, data } = exportModel(opts);
  console.log(`exported ${data.source} (generation ${data.generation}, ${data.params.length} params) -> ${file}`);
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------
async function benchCmd({ config: cli, opts }) {
  const { benchSingle } = require('./bench.js');
  const seconds = Number(opts.seconds) || 3;
  const workers = Number(opts.workers) || defaultWorkers();
  const net = { hidden: cli.hidden || [16], wasm: !!cli.wasm };
  if (net.wasm) require('./wasm-net.js').assertAvailable();
  console.log(`bench: ${seconds}s per measurement; workers detected: ${defaultWorkers()} (using ${workers}); ` +
    `net ${layersOf(net).join('-')}${net.wasm ? ' (wasm)' : ''}`);
  const sim = benchSingle('sim', seconds * 1000);
  console.log(`  1 core, simulation only:     ${Math.round(sim.fps).toLocaleString()} frames/s`);
  const pol = benchSingle('policy', seconds * 1000, net);
  console.log(`  1 core, simulation + policy: ${Math.round(pol.fps).toLocaleString()} frames/s`);
  const { Worker } = require('worker_threads');
  const per = await Promise.all(Array.from({ length: workers }, () => new Promise((resolve, reject) => {
    const w = new Worker(path.join(__dirname, 'bench.js'), { workerData: { mode: 'policy', ms: seconds * 1000, net } });
    w.once('message', resolve);
    w.once('error', reject);
  })));
  const total = per.reduce((a, r) => a + r.fps, 0);
  console.log(`  ${workers} workers, simulation + policy: ${Math.round(total).toLocaleString()} frames/s total, ` +
    `${Math.round(total / workers).toLocaleString()} per worker`);
  console.log(JSON.stringify({ seconds, workersDetected: defaultWorkers(), workers, layers: layersOf(net), wasm: net.wasm, simFps: Math.round(sim.fps), policyFps: Math.round(pol.fps), parallelFps: Math.round(total) }));
}

// ---------------------------------------------------------------------------
const ops = () => require('./ops.js');
const COMMANDS = {
  train, eval: evalCmd, bench: benchCmd, export: exportCmd, 'export-css': exportCssCmd,
  report: (a) => ops().reportCmd(a),
  finish: (a) => ops().finishCmd(a),
  session: (a) => ops().sessionCmd(a),
};

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
    console.error('usage: node trainer/cli.js <train|eval|bench|export|export-css|report|finish|session> [options]  (see the header of trainer/cli.js)');
    process.exit(2);
  }
  await cmd(args);
}

module.exports = { exportModel, evalHeldOut, loadModel, DEFAULTS, checkConfig };

if (require.main === module) main().catch((e) => {
  console.error(`error: ${e.message}`);
  process.exit(1);
});
