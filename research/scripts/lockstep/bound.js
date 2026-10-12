'use strict';
// Stage 1 bound (research/prereg-lockstep-tier-a.md §7): single-thread split of a full rules-v2
// game to 1M into sim (core step + observe + replay loop) and the wasm net (matmul, tanh), per shape.
// Usage: node bound.js [--trials 5] [--cap 1000000] [--seed-index 0] [--only full] [model.json ...]
const L = require('./common.js');

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const TRIALS = Number(opt('--trials', 5));
const CAP = Number(opt('--cap', 1000000));
const SEED = L.VALIDATION_SEEDS[Number(opt('--seed-index', 0))];
const ONLY = opt('--only', null);
const SAMPLE_EVERY = 8;
const files = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const MODELS = files.length ? files : [
  'runs/exp-bc-duck/v2-bc-duck-d2w32-s1/best.json',
  'runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json',
  'runs/exp-distill/v2-bc-duck-d1w6-s1/best.json',
];

const ns = (sec, n) => (sec * 1e9) / n;
const out = [];
for (const file of MODELS) {
  const model = L.loadModel(file);
  const shape = model.layers.join('-');
  // Record the game once (actions, sampled observations).
  const actions = [];
  const obs = [];
  let f = 0;
  const { makeAgent, state } = L.hashingAgents(model, (idx, game) => {
    actions.push(idx);
    if (game.state === 'running' && (f++ % SAMPLE_EVERY) === 0) obs.push(L.observe(game, new Float64Array(L.OBS_DIM)));
  });
  const ref = L.runEpisode(model, SEED, CAP, undefined, makeAgent);
  const F = ref.frames;
  const decisions = actions.length - 1;   // every running frame decides (actionRepeat 1); the idle frame does not
  if (model.actionRepeat !== 1) throw new Error('bound.js assumes actionRepeat 1');

  const full = L.minOf(TRIALS, () => L.runEpisode(model, SEED, CAP, undefined, L.wasmAgents(model)));
  const row = { shape, file, frames: F, score: ref.score, full_ns: ns(full.min, F), full_med_ns: ns(full.median, F) };
  if (ONLY !== 'full') {
    const act = Uint8Array.from(actions);
    const x = new Float64Array(L.OBS_DIM);
    const sim = L.minOf(TRIALS, () => {
      const game = L.createGame({ rng: L.createRng(SEED), version: model.gameVersion });
      for (let t = 0; t < F; t++) {
        if (game.state === 'running') L.observe(game, x);
        game.step(L.ACTIONS[act[t]]);
      }
    });
    const stepOnly = L.minOf(TRIALS, () => {
      const game = L.createGame({ rng: L.createRng(SEED), version: model.gameVersion });
      for (let t = 0; t < F; t++) game.step(L.ACTIONS[act[t]]);
    });
    // Net on recorded observations: full decide, matmul only, copy-in only.
    const n = obs.length;
    const flat = new Float64Array(n * L.OBS_DIM);
    obs.forEach((o, i) => flat.set(o, i * L.OBS_DIM));
    const netT = L.createProbeNet(model.layers, model.params, { tanh: true });
    const netM = L.createProbeNet(model.layers, model.params, { tanh: false });
    let sink = 0;
    const run = (net) => () => { for (let i = 0; i < n; i++) { net.input.set(flat.subarray(i * L.OBS_DIM, (i + 1) * L.OBS_DIM)); sink += net.decide(); } };
    const copyOnly = () => { for (let i = 0; i < n; i++) { netT.input.set(flat.subarray(i * L.OBS_DIM, (i + 1) * L.OBS_DIM)); sink += netT.input[3]; } };
    const tDec = L.minOf(TRIALS * 4, run(netT));
    const tMm = L.minOf(TRIALS * 4, run(netM));
    const tCopy = L.minOf(TRIALS * 4, copyOnly);
    // tanh alone on the recorded hidden pre-activations.
    const pre = [];
    for (let i = 0; i < Math.min(n, 20000); i++) { netT.input.set(flat.subarray(i * L.OBS_DIM, (i + 1) * L.OBS_DIM)); netT.capturePre(pre); }
    const preFlat = Float64Array.from(pre.flatMap((a) => Array.from(a)));
    const tmp = new Float64Array(preFlat.length);
    const tTanh = L.minOf(TRIALS * 4, () => { for (let q = 0; q < preFlat.length; q++) tmp[q] = Math.tanh(preFlat[q]); });
    const hidden = model.layers.slice(1, -1).reduce((a, b) => a + b, 0);
    const decideNs = ns(tDec.min - tCopy.min, n);
    const mmNs = ns(tMm.min - tCopy.min, n);
    const tanhNs = (ns(tTanh.min, preFlat.length)) * hidden;
    // Throughput bound of the matmul: one f64x2 mul + one f64x2 add per 2 MACs (padded units),
    // at 4 vector ops per cycle, 4.5 GHz (optimistic; stated in the prereg §2).
    let vops = 0;
    for (let l = 1; l < model.layers.length; l++) vops += model.layers[l - 1] * ((model.layers[l] + 1) & ~1);
    const mmBoundNs = vops / 4 / 4.5;
    Object.assign(row, {
      decisions, sim_ns: ns(sim.min, F), step_ns: ns(stepOnly.min, F), observe_ns: ns(sim.min - stepOnly.min, F),
      decide_ns: decideNs, matmul_ns: mmNs, tanh_ns_per_decision: tanhNs, tanh_ns_each: ns(tTanh.min, preFlat.length),
      copy_ns: ns(tCopy.min, n), argmax_rest_ns: decideNs - mmNs - tanhNs, mm_bound_ns: mmBoundNs, sink,
    });
    // Best-case per-thread speedup: sim + tanh + argmax/JS rest unchanged, matmul to the bound,
    // scatter free. Per frame = per decision here (actionRepeat 1, decisions/F ~ 1).
    const dpf = decisions / F;
    const c1 = row.sim_ns + dpf * decideNs;
    const cBest = row.sim_ns + dpf * (decideNs - mmNs + mmBoundNs);
    const cZero = row.sim_ns + dpf * (decideNs - mmNs);
    Object.assign(row, { c1_ns: c1, mlp_share: (dpf * decideNs) / c1, S_bound: c1 / cBest, S_mm_zero: c1 / cZero });
  }
  out.push(row);
  console.log(JSON.stringify(row));
}
