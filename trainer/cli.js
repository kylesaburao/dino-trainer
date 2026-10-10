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
//   node trainer/cli.js bc-gen   --dir runs/<name> [--seeds 1-40] [--levels 0,2000,50000,300000,900000]
//                              [--span0 3000] [--span 5000] [--keep-boring 10] [--game v2] [--workers N] [--overwrite]
//                              [--teacher jump|duck]
//   node trainer/cli.js bc-train --dir runs/<name> --hidden 32,32 [--backend node|python] [--epochs 15]
//                              [--batch 1024] [--lr 2e-3] [--seed 1] [--demos <dir or demos.json>]
//                              [--target 1000000] [--workers N] [--wasm] [--force] [--overwrite] [--snapshots]
//                              [--distill-from <best.json> [--distill-mode soft|hard] [--tau 1] [--alpha 0]]
//                              [--quant-bits N [--quant-scheme row|layer]]
//   node trainer/cli.js bc-quantize --from <best.json> --dir runs/<name> --quant-bits N [--quant-scheme row|layer]  (refuses another core hash unless --force)
//                              [--target 1000000] [--workers N] [--wasm] [--overwrite]
//
// Behaviour cloning of test/solver.js (trainer/bc.js; supervised, no ES):
//   bc-gen    plays the teacher on each seed (inclusive range) from each start level, for
//             --span0 points from 0 and --span points from the other levels, and writes
//             demos.bin (Float32 rows [policy.observe features, label, weight]; label = the
//             teacher's action: 0 none, 1 jump, 2 duck) and demos.json (manifest: coreHash,
//             teacher, teacherName, counts, solver crashes, sha256). --teacher jump (default;
//             createSolver, never ducks) or duck (createDuckSolver). Grounded no-op frames
//             with the nearest obstacle more than 40 frames
//             away (or none) are kept 1 in --keep-boring with that weight; the counter resets
//             every episode, so the output is the same for any --workers.
//   bc-train  trains a tanh MLP on the demos (--demos, default --dir) with weighted
//             cross-entropy, sqrt-balanced class weights, Adam and a cosine learning rate,
//             then validates it on the 25 validation seeds (full games to --target,
//             default 1000000) and writes best.json (the same schema as ES runs, so eval
//             and export work unchanged; config.teacherName from the demos manifest, 'jump'
//             if absent) and bc-metrics.jsonl (per-epoch loss and recall; recall.duck only
//             when the demos have duck rows).
//             --backend node (default; plain Node) or python (trainer/bc_train.py, needs
//             numpy: the repo's .venv/bin/python, or the PYTHON env var). Each backend is
//             deterministic per --seed on one machine; the two do not match each other.
//             Refuses demos generated on another core hash unless --force, and an
//             existing best.json unless --overwrite.
//             --snapshots (python backend only; off by default) also writes each epoch's
//             params to <dir>/snapshots/epoch-NN.json ({layers, params, epoch}, NN from 01,
//             the policy.createNet flat layout), taken after the epoch's last optimiser
//             step; it does not change training. Replaces any existing <dir>/snapshots/.
//             --distill-from FILE (off by default; both backends) trains on a trained model's
//             logits on the same demo rows instead of only the solver labels (Hinton et al.
//             2015): --distill-mode soft (default; alpha * CE(solver) + (1 - alpha) * tau^2 *
//             CE(softmax(z_t / tau), softmax(z / tau)), --tau default 1, --alpha default 0) or
//             hard (CE on the teacher's argmax). Row weights stay the solver-label class
//             weights. best.json config gains distillFrom {path, paramsSha256, layers,
//             teacherName}, distillMode, tau, alpha; bc-metrics rows gain distillLoss, and
//             `loss` stays the CE against the solver labels. Definitions: trainer/bc.js header.
//             --quant-bits N (2..8; off by default; both backends) is quantization-aware
//             training: symmetric integer weights (qmax 2^(N-1) - 1, so 2 = ternary), scale
//             max|w| / qmax per output row (--quant-scheme row, default) or per layer, fake-
//             quantized in forward and backward with a straight-through estimator; biases
//             stay float. The saved params are the float64 q * scale (still dino-mlp-v1), and
//             config.quant records bits, scheme, mode 'qat', the scales and the levels used;
//             the write fails unless every weight is exactly an integer times its scale.
//   bc-quantize post-training quantization of --from (a best.json) with the same rule,
//             into a new run dir: best.json (config.quant mode 'ptq', quantFrom {path,
//             paramsSha256}), after the same 25-seed validation. No training, no bc-metrics.
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
//   --ramp-frac F         share of training episodes (after the start-0 share) that start
//                         on the natural ramp (default 0 = off): the solver (test/solver.js)
//                         plays from 0 to a level uniform in points on [--ramp-min,
//                         --ramp-max] (default 400..800) and hands over to the candidate at
//                         the first grounded, keys-up frame; the candidate then plays
//                         --window points. Unlike late starts this is faithful below 2000
//                         (late starts begin at the speed cap). Costs the solver's frames
//                         (metric teacherFrames); start-0 + ramp shares must be <= 1.
//                         For --window-adapt, ramp episodes count as late windows (a
//                         window of the same length from a non-zero start); the gate's
//                         opening signal comes from start-0 episodes only, so ramp passes
//                         cannot hide a failing opening (with --start-zero-frac 0 there is
//                         no gate, as before).
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
  rampFrac: 0,            // curriculum: share of episodes starting on the natural ramp (solver handover)
  rampMin: 400,           // curriculum: natural-ramp start levels, uniform in points on [rampMin, rampMax]
  rampMax: 800,
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
  'windowAdaptEvery', 'fitnessAgg', 'wasm', 'rampFrac', 'rampMin', 'rampMax'];
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
  '--fitness-agg': ['fitnessAgg', String], '--ramp-frac': ['rampFrac', Number], '--ramp-min': ['rampMin', Number],
  '--ramp-max': ['rampMax', Number],
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
    } else if (['--resume', '--force', '--overwrite', '--auto-export', '--no-commit', '--skip-browser', '--snapshots'].includes(a)) {
      out.opts[a.slice(2).replace(/-(\w)/g, (m, c) => c.toUpperCase())] = true;
    } else if (['--workers', '--time-budget', '--dir', '--which', '--model', '--out', '--seconds', '--log-every', '--seeds', '--seed-offset',
      '--levels', '--span0', '--span', '--keep-boring', '--backend', '--epochs', '--batch', '--demos', '--teacher',
      '--distill-from', '--tau', '--distill-mode', '--alpha', '--quant-bits', '--quant-scheme', '--from'].includes(a)) {
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
  const late = c.startZeroFrac + (c.rampFrac || 0) < 1;   // any late (startScore) episodes
  if (c.startDist === 'log' && late && c.startMin + c.window > c.target) {
    throw new Error(`--start-min + --window (${c.startMin + c.window}) must not exceed --target (${c.target})`);
  }
  if (c.startDist === 'points' && late && c.startMin >= c.target) throw new Error('--start-min must be below --target');
  if (c.window > c.target) throw new Error('--window must not exceed --target');
  if (!(c.startZeroCap > 0 && c.startZeroCap <= c.target)) throw new Error('--start-zero-cap must be in (0, target]');
  if (c.windowAdapt && c.startDist !== 'points') throw new Error('--window-adapt needs --start-dist points');
  if (c.rampFrac > 0) {
    if (!(c.startZeroFrac + c.rampFrac <= 1)) throw new Error('--start-zero-frac + --ramp-frac must not exceed 1');
    if (!(c.rampMin > 0 && c.rampMin <= c.rampMax)) throw new Error('--ramp-min must be in (0, --ramp-max]');
    // The handover can come up to MAX_OVERSHOOT points after the ramp level.
    const { MAX_OVERSHOOT } = require('./ramp.js');
    if (!(c.rampMax + MAX_OVERSHOOT < c.target)) throw new Error(`--ramp-max + ${MAX_OVERSHOOT} (handover slack) must be below --target`);
  }
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
  if (!(c.rampFrac >= 0 && c.rampFrac <= 1)) throw new Error('--ramp-frac must be in [0, 1]');
  if (c.rampFrac > 0 && !(c.window > 0)) throw new Error('--ramp-frac needs the curriculum (--window)');
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
    'valCount', 'valMeanLog', 'valGeneration', 'valFrames', 'bestValCount', 'tiedPairs', 'stepSkipped', 'sigmaEff', 'window',
    ...(config.rampFrac > 0 ? ['teacherFrames'] : [])];
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
      const { seeds, starts, caps, ramp } = pairEp[k];
      for (const sign of [1, -1]) {
        const job = starts ? { noiseSeed: noiseSeeds[k], sign, seeds, starts, caps } : { noiseSeed: noiseSeeds[k], sign, seeds };
        if (ramp) Object.assign(job, { ramp, window: cur.window });   // natural-ramp episodes
        jobs.push(job);
      }
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
      // Natural-ramp results carry every episode's actual start (the handover score) and cap.
      const { starts, caps } = r.starts ? r : pairEp[i >> 1];
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
    const teacherFrames = results.reduce((a, r) => a + (r.teacherFrames || 0), 0);
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
    if (config.rampFrac > 0) row.teacherFrames = teacherFrames;   // absent when off: rows stay as before
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
      (config.rampFrac > 0 ? ` | teacher ${teacherFrames} frames` : '') +
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
// bc-gen / bc-train: behaviour cloning of test/solver.js (trainer/bc.js)
// ---------------------------------------------------------------------------
const requireDir = (opts, cmd) => {
  if (!opts.dir) throw new Error(`${cmd} needs --dir runs/<name>`);
  return path.resolve(opts.dir);
};

function parseSeedRange(v) {
  const m = /^(\d+)(?:-(\d+))?$/.exec(String(v));
  if (!m) throw new Error(`--seeds must be A-B (inclusive) or a single seed, got ${v}`);
  const a = Number(m[1]), b = m[2] !== undefined ? Number(m[2]) : a;
  if (!(b >= a)) throw new Error(`--seeds ${v}: empty range`);
  return Array.from({ length: b - a + 1 }, (_, i) => a + i);
}

async function bcGenCmd({ config: cli, opts }) {
  const bc = require('./bc.js');
  const dir = requireDir(opts, 'bc-gen');
  const d = bc.GEN_DEFAULTS;
  const o = {
    seeds: parseSeedRange(opts.seeds !== undefined ? opts.seeds : '1-40'),
    levels: opts.levels !== undefined ? opts.levels.split(',').map(Number) : d.levels,
    span0: opts.span0 !== undefined ? Number(opts.span0) : d.span0,
    span: opts.span !== undefined ? Number(opts.span) : d.span,
    keepBoring: opts.keepBoring !== undefined ? Number(opts.keepBoring) : d.keepBoring,
    gameVersion: cli.gameVersion || d.gameVersion,
    teacher: opts.teacher !== undefined ? opts.teacher : d.teacher,
    workers: Number(opts.workers) || defaultWorkers(),
    coreHash: coreHash(),
  };
  if (!GAME_VERSIONS.includes(o.gameVersion)) throw new Error(`--game must be one of ${GAME_VERSIONS.join(', ')}`);
  if (!Object.hasOwn(bc.TEACHERS, o.teacher)) throw new Error(`--teacher must be one of ${Object.keys(bc.TEACHERS).join(', ')}, got ${o.teacher}`);
  bc.resolveTeacher(o.teacher);   // e.g. createDuckSolver not available on this tree
  if (!o.levels.every(l => Number.isInteger(l) && l >= 0)) throw new Error('--levels must be non-negative integers');
  if (!(Number.isInteger(o.keepBoring) && o.keepBoring >= 1)) throw new Error('--keep-boring must be a positive integer');
  if (!(o.span0 > 0 && o.span > 0)) throw new Error('--span0 and --span must be positive');
  const manifestFile = path.join(dir, 'demos.json');
  if (fs.existsSync(manifestFile) && !opts.overwrite) throw new Error(`${manifestFile} exists; pass --overwrite to regenerate`);
  log(`bc-gen: dir=${dir} seeds ${o.seeds[0]}-${o.seeds[o.seeds.length - 1]} levels ${o.levels.join(',')} span0 ${o.span0} span ${o.span} ` +
    `keep-boring ${o.keepBoring} game ${o.gameVersion} teacher ${o.teacher} workers ${o.workers} core=${o.coreHash.slice(0, 12)}`);
  const t0 = Date.now();
  let lastLog = 0;
  o.onProgress = (done, total, rows) => {
    if (done === total || Date.now() - lastLog > 15000) { lastLog = Date.now(); log(`bc-gen: ${done}/${total} seeds, ${rows} rows`); }
  };
  const manifest = await require('./bc.js').generateDemos(dir, o);
  writeJsonAtomic(manifestFile, manifest);
  log(`bc-gen: ${manifest.rows} rows (none ${manifest.counts.rows.none}, jump ${manifest.counts.rows.jump}` +
    `${manifest.counts.rows.duck ? `, duck ${manifest.counts.rows.duck}` : ''}), ` +
    `solver crashes ${manifest.solverCrashes}, ${((Date.now() - t0) / 1000).toFixed(1)}s -> ${path.relative(ROOT, manifestFile)}`);
  if (manifest.solverCrashes) throw new Error(`the solver crashed in ${manifest.solverCrashes} episode(s); the demonstrations are not trustworthy`);
}

// A trained model file (best.json, or a snapshot epoch-NN.json) for --distill-from / --from.
function loadTrainedModel(file, flag) {
  const abs = path.resolve(file);
  if (!abs.endsWith('.json') || !fs.existsSync(abs)) throw new Error(`${flag}: no model JSON at ${file}`);
  const m = JSON.parse(fs.readFileSync(abs, 'utf8'));
  if (m.format && m.format !== FORMAT) throw new Error(`${flag}: unsupported model format ${m.format}`);
  if (!Array.isArray(m.layers) || m.layers[0] !== OBS_DIM || m.layers[m.layers.length - 1] !== 3) throw new Error(`${flag}: layers must be ${OBS_DIM}-...-3`);
  if (!Array.isArray(m.params) || m.params.length !== numParams(m.layers)) throw new Error(`${flag}: params do not match layers`);
  return { abs, model: m };
}

// --distill-from FILE [--distill-mode soft|hard] [--tau T] [--alpha A]; null when off.
function parseDistill(opts) {
  if (opts.distillFrom === undefined) {
    for (const k of ['distillMode', 'tau', 'alpha']) if (opts[k] !== undefined) throw new Error('--distill-mode, --tau and --alpha need --distill-from');
    return null;
  }
  const mode = opts.distillMode || 'soft';
  if (!['soft', 'hard'].includes(mode)) throw new Error(`--distill-mode must be soft or hard, got ${mode}`);
  if (mode === 'hard' && (opts.tau !== undefined || opts.alpha !== undefined)) throw new Error('--tau and --alpha apply to --distill-mode soft only');
  const tau = opts.tau !== undefined ? Number(opts.tau) : 1, alpha = opts.alpha !== undefined ? Number(opts.alpha) : 0;
  if (!(Number.isFinite(tau) && tau > 0)) throw new Error('--tau must be > 0');
  if (!(alpha >= 0 && alpha <= 1)) throw new Error('--alpha must be in [0, 1]');
  const { abs, model } = loadTrainedModel(opts.distillFrom, '--distill-from');
  return { mode, tau, alpha, abs, model };
}

// --quant-bits N [--quant-scheme row|layer]; null when off.
function parseQuant(opts, bc) {
  if (opts.quantBits === undefined) {
    if (opts.quantScheme !== undefined) throw new Error('--quant-scheme needs --quant-bits');
    return null;
  }
  const q = { bits: Number(opts.quantBits), scheme: opts.quantScheme || 'row' };
  bc.checkQuantOptions(q);
  return q;
}

const distillConfig = (dz, bc) => ({
  distillFrom: { path: path.relative(ROOT, dz.abs), paramsSha256: bc.paramsSha256(dz.model.params), layers: dz.model.layers,
    teacherName: (dz.model.config && dz.model.config.teacherName) || null },
  distillMode: dz.mode, ...(dz.mode === 'soft' ? { tau: dz.tau, alpha: dz.alpha } : {}),
});
const distillText = (dz) => ` distill=${dz.mode}${dz.mode === 'soft' ? ` tau=${dz.tau} alpha=${dz.alpha}` : ''} from=${path.relative(ROOT, dz.abs)}`;
const quantText = (q) => ` quant=int${q.bits}/${q.scheme}`;

async function bcTrainCmd({ config: cli, opts }) {
  const bc = require('./bc.js');
  const dir = requireDir(opts, 'bc-train');
  if (!cli.hidden) throw new Error('bc-train needs --hidden (e.g. --hidden 32,32)');
  const backend = opts.backend || 'node';
  if (!['node', 'python'].includes(backend)) throw new Error(`--backend must be node or python, got ${backend}`);
  if (opts.snapshots && backend !== 'python') throw new Error('--snapshots is only supported with --backend python');
  const dz = parseDistill(opts);
  const quant = parseQuant(opts, bc);
  const d = bc.TRAIN_DEFAULTS;
  const o = {
    hidden: cli.hidden,
    epochs: opts.epochs !== undefined ? Number(opts.epochs) : d.epochs,
    batch: opts.batch !== undefined ? Number(opts.batch) : d.batch,
    lr: cli.lr !== undefined ? cli.lr : d.lr,
    seed: cli.seed !== undefined ? cli.seed : d.seed,
  };
  if (!o.hidden.every(h => Number.isInteger(h) && h > 0)) throw new Error('--hidden must be positive integers');
  if (!(Number.isInteger(o.epochs) && o.epochs > 0 && Number.isInteger(o.batch) && o.batch > 0)) throw new Error('--epochs and --batch must be positive integers');
  const target = cli.target || 1000000;
  fs.mkdirSync(dir, { recursive: true });
  const bestFile = path.join(dir, 'best.json');
  if (fs.existsSync(bestFile) && !opts.overwrite) throw new Error(`${bestFile} exists; pass --overwrite to replace it`);
  const hash = coreHash();
  const demos = bc.resolveDemos(path.resolve(opts.demos || dir));
  const sha = bc.checkDemos(demos, hash, !!opts.force, (m) => log(m));
  const release = acquireLock(dir);
  process.on('exit', release);
  const snapshotDir = opts.snapshots ? path.join(dir, 'snapshots') : null;
  if (snapshotDir) fs.rmSync(snapshotDir, { recursive: true, force: true });
  const gameVersion = demos.manifest.gameVersion;
  if (dz && dz.model.gameVersion && dz.model.gameVersion !== gameVersion) throw new Error(`--distill-from: teacher plays ${dz.model.gameVersion}, demos are ${gameVersion}`);
  log(`bc-train: dir=${dir} backend=${backend} layers=${[OBS_DIM, ...o.hidden, 3].join('-')} epochs ${o.epochs} batch ${o.batch} lr ${o.lr} seed ${o.seed} ` +
    `demos ${path.relative(ROOT, demos.manifestFile)} (${demos.manifest.rows} rows, sha ${sha.slice(0, 12)}) core=${hash.slice(0, 12)}` +
    `${dz ? distillText(dz) : ''}${quant ? quantText(quant) : ''}`);
  // Teacher logits on every demo row, straight from the Float32 rows (no Float64 copy).
  let logits = null;
  if (dz) {
    const buf = fs.readFileSync(demos.binFile);
    const f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    logits = bc.teacherLogits(dz.model, { X: f, n: f.length / bc.ROW, dim: OBS_DIM, stride: bc.ROW });
    const ya = bc.argmaxLabels(logits);
    let agree = 0;
    for (let i = 0; i < ya.length; i++) if (ya[i] === f[i * bc.ROW + OBS_DIM]) agree++;
    log(`distill: teacher ${path.relative(ROOT, dz.abs)} (${dz.model.layers.join('-')}) argmax = solver label on ${agree}/${ya.length} rows`);
  }

  const metricsFile = path.join(dir, 'bc-metrics.jsonl');
  fs.rmSync(metricsFile, { force: true });
  const onEpoch = (row) => {
    fs.appendFileSync(metricsFile, JSON.stringify({ backend, ...row }) + '\n');
    log(`epoch ${row.epoch}: loss ${row.loss.toFixed(4)} recall none ${row.recall.none.toFixed(4)} jump ${row.recall.jump.toFixed(4)}` +
      `${row.recall.duck !== undefined ? ` duck ${row.recall.duck.toFixed(4)}` : ''} (${row.seconds.toFixed(1)}s)`);
  };
  const t0 = Date.now();
  let result;
  const extra = { ...(quant ? { quant } : {}) };
  if (backend === 'node') {
    if (dz) extra.distill = { logits, mode: dz.mode, tau: dz.tau, alpha: dz.alpha };
    result = bc.trainNode(bc.loadDemos(demos.binFile), { ...o, ...extra, onEpoch });
  } else {
    const outFile = path.join(dir, `bc-python.${process.pid}.tmp`);
    const logitsFile = path.join(dir, `teacher-logits.${process.pid}.tmp`);
    try {
      if (dz) { bc.writeLogits(logitsFile, logits); extra.distill = { logitsFile, mode: dz.mode, tau: dz.tau, alpha: dz.alpha }; }
      result = await bc.trainPython(demos.binFile, outFile, { ...o, ...(snapshotDir ? { snapshotDir } : {}), ...extra }, ROOT, (line) => {
        if (line.startsWith('{')) onEpoch(JSON.parse(line)); else if (line) log(line);
      });
    } finally {
      fs.rmSync(outFile, { force: true });
      fs.rmSync(logitsFile, { force: true });
    }
  }
  logits = null;
  const trainSeconds = (Date.now() - t0) / 1000;
  let quantConfig = null;
  if (quant) {
    quantConfig = { bits: quant.bits, scheme: quant.scheme, mode: 'qat', biases: 'float', scales: result.scales };
    quantConfig.levels = bc.checkQuantized(result.layers, result.params, quantConfig);   // throws unless exactly k * scale
  }
  const model = { format: FORMAT, layers: result.layers, params: result.params, actionRepeat: 1, gameVersion };

  const valSeedList = validationSeeds(25);
  const workers = Number(opts.workers) || defaultWorkers();
  const pool = createPool(Math.min(workers, valSeedList.length));
  log(`validation: ${valSeedList.length} seeds to ${target} (${gameVersion})`);
  const scores = await scoreSeeds(pool, model, valSeedList, target, !!cli.wasm);
  await pool.terminate();
  const validation = valSummary(scores, target);
  const last = result.log[result.log.length - 1];
  const config = {
    target, hidden: o.hidden, actionRepeat: 1, gameVersion,
    method: 'behaviour cloning of test/solver.js (supervised; no ES)',
    trainer: `trainer/cli.js bc-train --backend ${backend}`, backend,
    teacherName: demos.manifest.teacherName || 'jump',
    epochs: o.epochs, batch: o.batch, lr: o.lr, seed: o.seed,
    demos: path.relative(ROOT, demos.manifestFile), demosSha256: sha, demosRows: demos.manifest.rows,
    trainSeconds, final: last ? { loss: last.loss, recall: last.recall } : null,
    ...(snapshotDir ? { snapshots: path.relative(ROOT, snapshotDir) } : {}),
    ...(dz ? distillConfig(dz, bc) : {}),
    ...(quantConfig ? { quant: quantConfig } : {}),
  };
  writeJsonAtomic(bestFile, bc.assembleBest({ layers: result.layers, params: result.params, validation, coreHash: hash, config }));
  release();
  log(`validation: median ${validation.median} min ${validation.min} count ${validation.count}/${valSeedList.length}; ` +
    `trained in ${trainSeconds.toFixed(1)}s -> ${path.relative(ROOT, bestFile)}`);
}

// bc-quantize: post-training quantization of a trained model into a new run dir.
async function bcQuantizeCmd({ config: cli, opts }) {
  const bc = require('./bc.js');
  const dir = requireDir(opts, 'bc-quantize');
  if (!opts.from) throw new Error('bc-quantize needs --from <best.json>');
  const quant = parseQuant(opts, bc);
  if (!quant) throw new Error('bc-quantize needs --quant-bits N');
  const { abs, model: src } = loadTrainedModel(opts.from, '--from');
  const hash = coreHash();
  if (src.coreHash && src.coreHash !== hash) {
    if (!opts.force) throw new Error(`core hash mismatch: --from was trained on ${src.coreHash.slice(0, 12)}, current core/policy is ${hash.slice(0, 12)}; pass --force to quantize it anyway`);
    log(`WARNING: --from core hash ${src.coreHash.slice(0, 12)} differs from the current ${hash.slice(0, 12)}, continuing because of --force`);
  }
  const target = cli.target || 1000000;
  fs.mkdirSync(dir, { recursive: true });
  const bestFile = path.join(dir, 'best.json');
  if (fs.existsSync(bestFile) && !opts.overwrite) throw new Error(`${bestFile} exists; pass --overwrite to replace it`);
  const release = acquireLock(dir);
  process.on('exit', release);
  const gameVersion = src.gameVersion || (src.config && src.config.gameVersion) || 'v2';
  log(`bc-quantize: dir=${dir} from=${path.relative(ROOT, abs)} layers=${src.layers.join('-')}${quantText(quant)}`);
  const q = bc.quantizeParams(src.layers, src.params, quant);
  const quantConfig = { bits: quant.bits, scheme: quant.scheme, mode: 'ptq', biases: 'float', scales: q.scales };
  quantConfig.levels = bc.checkQuantized(src.layers, q.params, quantConfig);
  let changed = 0, maxErr = 0;
  q.params.forEach((v, i) => { if (v !== src.params[i]) changed++; maxErr = Math.max(maxErr, Math.abs(v - src.params[i])); });
  log(`bc-quantize: ${changed}/${q.params.length} params changed, max |change| ${maxErr.toExponential(3)}, ${quantConfig.levels} integer levels`);
  const model = { format: FORMAT, layers: src.layers, params: q.params, actionRepeat: 1, gameVersion };
  const valSeedList = validationSeeds(25);
  const workers = Number(opts.workers) || defaultWorkers();
  const pool = createPool(Math.min(workers, valSeedList.length));
  log(`validation: ${valSeedList.length} seeds to ${target} (${gameVersion})`);
  const scores = await scoreSeeds(pool, model, valSeedList, target, !!cli.wasm);
  await pool.terminate();
  const validation = valSummary(scores, target);
  const sc = src.config || {};
  const config = {
    target, hidden: src.layers.slice(1, -1), actionRepeat: 1, gameVersion,
    method: 'behaviour cloning of test/solver.js, then post-training quantization (no training)',
    trainer: 'trainer/cli.js bc-quantize', teacherName: sc.teacherName || 'jump',
    ...(sc.demos ? { demos: sc.demos, demosSha256: sc.demosSha256, demosRows: sc.demosRows } : {}),
    quantFrom: { path: path.relative(ROOT, abs), paramsSha256: bc.paramsSha256(src.params), method: sc.method || null, seed: sc.seed ?? null },
    quant: quantConfig,
  };
  writeJsonAtomic(bestFile, bc.assembleBest({ layers: src.layers, params: q.params, validation, coreHash: src.coreHash || hash, config }));
  release();
  log(`validation: median ${validation.median} min ${validation.min} count ${validation.count}/${valSeedList.length} -> ${path.relative(ROOT, bestFile)}`);
}

// ---------------------------------------------------------------------------
const ops = () => require('./ops.js');
const COMMANDS = {
  train, eval: evalCmd, bench: benchCmd, export: exportCmd, 'export-css': exportCssCmd,
  report: (a) => ops().reportCmd(a),
  finish: (a) => ops().finishCmd(a),
  session: (a) => ops().sessionCmd(a),
  'bc-gen': bcGenCmd, 'bc-train': bcTrainCmd, 'bc-quantize': bcQuantizeCmd,
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
    console.error('usage: node trainer/cli.js <train|eval|bench|export|export-css|report|finish|session|bc-gen|bc-train|bc-quantize> [options]  (see the header of trainer/cli.js)');
    process.exit(2);
  }
  await cmd(args);
}

module.exports = { exportModel, evalHeldOut, loadModel, DEFAULTS, checkConfig };

if (require.main === module) main().catch((e) => {
  console.error(`error: ${e.message}`);
  process.exit(1);
});
