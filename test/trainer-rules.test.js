'use strict';
// Trainer options beyond the original ES loop: selection and stop rules, ES
// hygiene (tie ranks, decoupled decay, stall rule), sigma control, curriculum v2,
// per-pair episodes and folded validation. All pure helpers; no workers.
const test = require('node:test');
const assert = require('node:assert');
const { createRng, CFG } = require('../core.js');
const { numParams } = require('../policy.js');
const env = require('../trainer/env.js');
const es = require('../trainer/es.js');
const rules = require('../trainer/rules.js');

const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);

// --- seed sets ---------------------------------------------------------------
test('extended seed sets keep their first 25, have no duplicates and stay in disjoint ranges', () => {
  // The original 25-seed formulas, written out independently of env.js.
  const h = (i, salt) => (Math.imul((i + 1) ^ salt, 0x9E3779B1) ^ 0x85EBCA6B) >>> 0;
  const val25 = Array.from({ length: 25 }, (_, i) => 0x40000000 + (h(i, 0x51) & 0x3FFFFFFF));
  const eval25 = Array.from({ length: 25 }, (_, i) => (0x80000000 | h(i, 0xE7)) >>> 0);
  assert.deepStrictEqual(env.VALIDATION_SEEDS, val25);
  assert.deepStrictEqual(env.EVAL_SEEDS, eval25);
  const val = env.validationSeeds(500), held = env.evalSeeds(500);
  assert.deepStrictEqual(val.slice(0, 25), val25);
  assert.deepStrictEqual(held.slice(0, 25), eval25);
  assert.strictEqual(new Set(val).size, 500);
  assert.strictEqual(new Set(held).size, 500);
  assert.ok(val.every(s => s >= 2 ** 30 && s < 2 ** 31), 'validation in [2^30, 2^31)');
  assert.ok(held.every(s => s >= 2 ** 31 && s < 2 ** 32), 'held-out in [2^31, 2^32)');
  const rng = createRng(1);
  for (let i = 0; i < 10000; i++) assert.ok(env.trainSeed(rng) < 2 ** 30, 'training in [0, 2^30)');
});

// --- selection and stop rule -------------------------------------------------
test('count selection ranks (count, mean log score) and replaces a saturated best on a tie', () => {
  const T = 100;
  const s = (scores) => rules.valSummary(scores, T);
  const best = (v) => ({ valMedian: v.median, valCount: v.count, valMeanLog: v.meanLog });
  const a = s([100, 100, 50]), b = s([100, 100, 60]), c = s([100, 99, 99]);
  assert.strictEqual(a.count, 2);
  close(a.meanLog, (2 * Math.log(101) + Math.log(51)) / 3);
  assert.ok(rules.isBetter(b, best(a), 'count', 3), 'same count, higher meanLog');
  assert.ok(!rules.isBetter(a, best(b), 'count', 3));
  assert.ok(!rules.isBetter(a, best(a), 'count', 3), 'unsaturated tie does not replace');
  assert.ok(!rules.isBetter(c, best(a), 'count', 3), 'count dominates meanLog');
  const full = s([100, 100, 100]);
  assert.ok(rules.isBetter(full, best(full), 'count', 3), 'saturated tie replaces (newest all-pass θ)');
  // A best from a run that used the median rule (no count fields) is replaced.
  assert.ok(rules.isBetter(a, { valMedian: 100 }, 'count', 3));
  // median: the original strict-median rule.
  assert.ok(!rules.isBetter(full, best(full), 'median', 3));
  assert.ok(rules.isBetter(s([1, 2, 3]), null, 'median', 3));
});

test('stop rule parsing', () => {
  assert.deepStrictEqual(rules.parseStopRule('median'), { kind: 'median', seeds: 25 });
  assert.deepStrictEqual(rules.parseStopRule('allpass'), { kind: 'allpass', seeds: 25 });
  assert.deepStrictEqual(rules.parseStopRule('allpass:59'), { kind: 'allpass', seeds: 59 });
  assert.throws(() => rules.parseStopRule('median:5'));
  assert.throws(() => rules.parseStopRule('any'));
});

test('deadlock regression: a frozen best is never held-out tested twice', () => {
  const T = 1000;
  const rule = rules.parseStopRule('median');
  const best = { generation: 40, valMedian: T };
  const val = rules.valSummary(Array(25).fill(T), T);
  assert.ok(rules.wantHeldOut(rule, val, best, null, T, 25), 'first check runs');
  const heldOut = { generation: 40, target: T, seeds: 25, median: 500 };   // ...and failed
  // best stays frozen at the capped median: later validations must not retest it.
  for (let g = 0; g < 5; g++) assert.ok(!rules.wantHeldOut(rule, val, best, heldOut, T, 25));
  // A new best, a new target or a different held-out size is tested again.
  assert.ok(rules.wantHeldOut(rule, val, { ...best, generation: 45 }, heldOut, T, 25));
  assert.ok(rules.wantHeldOut(rule, rules.valSummary(Array(25).fill(2000), 2000), best, heldOut, 2000, 25));
  assert.ok(rules.wantHeldOut(rules.parseStopRule('allpass:59'), val, best, heldOut, T, 25));
  // A held-out record from an older checkpoint (no seeds field) allows a retest.
  assert.ok(rules.wantHeldOut(rule, val, best, { generation: 40, target: T, median: 500 }, T, 25));
  // allpass triggers only when every validation seed passes.
  const allpass = rules.parseStopRule('allpass:25');
  assert.ok(!rules.wantHeldOut(allpass, rules.valSummary([...Array(24).fill(T), T - 1], T), best, null, T, 25));
  assert.ok(rules.wantHeldOut(allpass, val, best, null, T, 25));
  assert.ok(!rules.wantHeldOut(rule, null, best, null, T, 25), 'no validation this generation');
});

test('allpass held-out check plays rounds and aborts after the first round with a failure', async () => {
  const T = 10;
  const seeds = Array.from({ length: 25 }, (_, i) => 1000 + i);
  const played = [];
  const scorer = (failing) => async (chunk) => { played.push(chunk.length); return chunk.map(s => (failing.has(s) ? 3 : T)); };
  let r = await rules.sequentialAllPass(scorer(new Set()), seeds, 8, T);
  assert.deepStrictEqual(played, [8, 8, 8, 1]);
  assert.deepStrictEqual({ played: r.played, passed: r.passed, pass: r.pass, firstFailSeed: r.firstFailSeed }, { played: 25, passed: 25, pass: true, firstFailSeed: null });
  played.length = 0;
  r = await rules.sequentialAllPass(scorer(new Set([1010, 1012, 1020])), seeds, 8, T);
  assert.deepStrictEqual(played, [8, 8], 'stops after the round holding the first failure');
  assert.deepStrictEqual({ played: r.played, passed: r.passed, pass: r.pass, firstFailSeed: r.firstFailSeed }, { played: 16, passed: 14, pass: false, firstFailSeed: 1010 });
});

test('folded validation results split into the training candidates and the pre-update θ games', () => {
  const results = Array.from({ length: 16 + 5 }, (_, i) => ({ id: i }));
  const { train, val } = rules.splitFolded(results, 16);
  assert.deepStrictEqual(train.map(r => r.id), Array.from({ length: 16 }, (_, i) => i));
  assert.deepStrictEqual(val.map(r => r.id), [16, 17, 18, 19, 20]);
  assert.strictEqual(rules.splitFolded(results.slice(0, 16), 16).val.length, 0, 'no folding: nothing extra');
});

// --- ES hygiene --------------------------------------------------------------
test('mid ranks: a tied pair gets weight 0 and an all-tied generation gives a zero gradient', () => {
  const r = es.centredRanks([5, 1, 5, 3], 'mid');
  assert.deepStrictEqual(Array.from(r), [(2.5 / 3) - 0.5, -0.5, (2.5 / 3) - 0.5, 1 / 3 - 0.5]);
  assert.strictEqual(r[0] - r[2], 0);
  // Index ties (the original rule) give a tied pair a nonzero weight.
  const ri = es.centredRanks([5, 5], 'index');
  assert.notStrictEqual(ri[0] - ri[1], 0);
  const n = 50, seeds = [11, 22, 33, 44];
  const tiedPairs = es.esGradient(seeds, [7, 7, 9, 9, 7, 7, 1, 1], n, 0.1, { ties: 'mid' });
  assert.ok(tiedPairs.every(x => x === 0), 'every pair tied within itself');
  const allTied = es.esGradient(seeds, Array(8).fill(54), n, 0.1, { ties: 'mid' });
  assert.ok(allTied.every(x => x === 0));
  assert.ok(es.esGradient(seeds, Array(8).fill(54), n, 0.1).some(x => x !== 0), 'index ties: pure noise');
  assert.strictEqual(rules.tiedPairs([7, 7, 9, 9, 7, 8, 1, 1]), 3);
});

test('decoupled weight decay is plain Adam followed by theta *= 1 - lr * wd', () => {
  const theta0 = Float64Array.from([0.5, -1.2, 2]);
  const grad = Float64Array.from([0.3, 0, -0.7]);
  const lr = 0.03, wd = 0.1;
  const fresh = () => ({ t: 0, m: new Float64Array(3), v: new Float64Array(3) });
  const t = Float64Array.from(theta0), st = fresh();
  es.adamStep(t, grad, st, lr, wd, 'decoupled');
  const plain = Float64Array.from(theta0), sp = fresh();
  es.adamStep(plain, grad, sp, lr, 0, 'coupled');
  for (let i = 0; i < 3; i++) {
    // First Adam step: m/c1 = g, v/c2 = g^2, so the step is lr * g / (|g| + eps).
    const step = grad[i] === 0 ? 0 : lr * grad[i] / (Math.abs(grad[i]) + 1e-8);
    close(t[i], (theta0[i] + step) * (1 - lr * wd));
    close(t[i], plain[i] * (1 - lr * wd));
  }
  assert.deepStrictEqual(Array.from(st.m), Array.from(sp.m), 'decay never enters the moments');
  // Coupled decay: L2 inside the gradient (the original rule).
  const c = Float64Array.from(theta0), sc = fresh();
  es.adamStep(c, grad, sc, lr, wd);
  close(sc.m[1], 0.1 * (0 - wd * theta0[1]));
});

test('stall rule: skip and widen sigma while most pairs tie, reset otherwise', () => {
  let m = 1;
  const seen = [];
  for (let g = 0; g < 6; g++) {
    const u = rules.stallUpdate(m, 0.6);
    assert.strictEqual(u.skip, true);
    m = u.stallMult;
    seen.push(m);
  }
  assert.deepStrictEqual(seen, [1.5, 2.25, 3.375, 4, 4, 4]);
  assert.deepStrictEqual(rules.stallUpdate(m, 0.5), { skip: false, stallMult: 1 });
});

// --- sigma control -----------------------------------------------------------
test('fan-in sigma vector follows the policy.js layout for any shape', () => {
  for (const layers of [[44, 32, 32, 3], [44, 64, 3], [44, 8, 4, 3]]) {
    const s = es.sigmaVector(layers, 0.1, 'fanin');
    assert.strictEqual(s.length, numParams(layers));
    let k = 0;
    for (let l = 1; l < layers.length; l++) {
      const w = 0.1 * Math.sqrt(44 / layers[l - 1]);
      for (let i = 0; i < layers[l] * layers[l - 1]; i++) assert.strictEqual(s[k++], w, `layer ${l} weight`);
      for (let i = 0; i < layers[l]; i++) assert.strictEqual(s[k++], 0.1, `layer ${l} bias`);
    }
    assert.strictEqual(s[0], 0.1, 'first layer keeps sigma');
  }
  assert.strictEqual(es.sigmaVector([44, 32, 3], 0.07, 'none'), 0.07);
});

test('perturb and esGradient use the same per-parameter sigma', () => {
  const layers = [44, 6, 3], n = numParams(layers);
  const sigma = es.sigmaVector(layers, 0.1, 'fanin');
  const theta = new Float64Array(n).fill(0.25);
  const eps = es.noise(77, n);
  const p = es.perturb(theta, 77, 1, sigma), q = es.perturb(theta, 77, -1, sigma);
  for (let i = 0; i < n; i++) {
    close(p[i], 0.25 + sigma[i] * eps[i]);
    close((p[i] - q[i]) / (2 * sigma[i]), eps[i], 1e-9);
  }
  assert.deepStrictEqual(Array.from(es.perturb(theta, 77, 0, sigma)), Array.from(theta), 'sign 0 is θ itself');
  // One pair, fitness (1, 0): ranks give w = 1, so g_i = eps_i / (2 sigma_i).
  const g = es.esGradient([77], [1, 0], n, sigma);
  for (let i = 0; i < n; i++) close(g[i], eps[i] / (2 * sigma[i]));
  // A scalar sigma is the same as a constant vector.
  const gs = es.esGradient([77], [1, 0], n, 0.1), gv = es.esGradient([77], [1, 0], n, new Float64Array(n).fill(0.1));
  for (let i = 0; i < n; i++) close(gs[i], gv[i]);
});

test('sigma anneal: shrink while candidates crash in the opening and the centre survives, with cooldown and floor', () => {
  const crashy = [...Array(30).fill(50), ...Array(70).fill(900)];   // 30% below 200
  const safe = [...Array(20).fill(50), ...Array(80).fill(900)];     // 20%
  const args = (o) => ({ zeroStartScores: crashy, valMin: 300, generation: 50, sigmaMin: 0.02, ...o });
  let a = rules.annealUpdate({ sigma: 0.1, annealedAt: null }, args());
  close(a.sigma, 0.08);
  assert.strictEqual(a.annealedAt, 50);
  assert.deepStrictEqual(rules.annealUpdate(a, args({ generation: 59 })), a, 'cooldown');
  close(rules.annealUpdate(a, args({ generation: 60 })).sigma, 0.064);
  assert.strictEqual(rules.annealUpdate({ sigma: 0.1, annealedAt: null }, args({ zeroStartScores: safe })).sigma, 0.1);
  assert.strictEqual(rules.annealUpdate({ sigma: 0.1, annealedAt: null }, args({ valMin: 150 })).sigma, 0.1, 'centre fails the opening');
  assert.strictEqual(rules.annealUpdate({ sigma: 0.1, annealedAt: null }, args({ valMin: null })).sigma, 0.1, 'no validation yet');
  assert.strictEqual(rules.annealUpdate({ sigma: 0.021, annealedAt: null }, args()).sigma, 0.02, 'floor');
});

// --- curriculum v2 -----------------------------------------------------------
const seedsFor = (g, n) => Array.from({ length: n }, (_, j) => (g * 7919 + j * 104729 + 1) & 0x3FFFFFFF);

test('points starts are uniform on [startMin, target) and the log rule is unchanged', () => {
  const c = { target: 1000000, window: 25000, startZeroFrac: 0.25, startMin: 2000, startDist: 'points', startZeroCap: 5000 };
  const late = [];
  for (let g = 0; g < 2500; g++) {
    const s = env.startScores(seedsFor(g, 8), c);
    assert.deepStrictEqual(s.slice(0, 2), [0, 0], '0.25 of 8 = 2 zero starts');
    late.push(...s.slice(2));
  }
  assert.ok(late.every(x => Number.isInteger(x) && x >= c.startMin && x < c.target));
  const bins = Array(10).fill(0);
  for (const x of late) bins[Math.floor((x - c.startMin) / (c.target - c.startMin) * 10)]++;
  for (const b of bins) assert.ok(Math.abs(b / late.length - 0.1) < 0.01, `decile share ${b / late.length}`);
  // The top 2.5% of starts (where windows reach 1M) is sampled ~ uniformly, unlike log starts.
  assert.ok(Math.abs(late.filter(x => x >= 975000).length / late.length - 0.0245) < 0.006);
  // Same seeds, log rule: values from the original formula.
  const logC = { ...c, startDist: 'log' }, seeds = seedsFor(3, 8);
  const hi = logC.target - logC.window;
  assert.ok(env.startScores(seeds, logC).slice(2).every(x => x >= 2000 && x <= hi));
  assert.deepStrictEqual(env.startScores(seeds, logC), env.startScores(seeds, { ...logC, startDist: undefined }));
});

test('episode caps: start-0 episodes end at the zero cap, a top window is clipped and counts by its own span', () => {
  const c = { target: 1000000, window: 25000, startZeroCap: 5000 };
  const starts = [0, 2000, 500000, 990000];
  const caps = env.episodeCaps(starts, c);
  assert.deepStrictEqual(caps, [5000, 27000, 525000, 1000000]);
  const d = (score) => score / CFG.scoreCoef;
  // Clipped window: 10000 points long, so reaching the target is fitness 1 and halfway is 0.5.
  assert.strictEqual(env.windowFitness(d(1000000), 990000, caps[3] - 990000), 1);
  close(env.windowFitness(d(995000), 990000, caps[3] - 990000), 0.5, 1e-9);
  close(env.windowFitness(d(2500), 0, caps[0]), 0.5, 1e-9);
  assert.strictEqual(env.windowFitness(d(5000), 0, caps[0]), 1);
  // Log rule with the cap defaulting to the window: every episode ends at start + window.
  assert.deepStrictEqual(env.episodeCaps([0, 3000], { target: 1000000, window: 50000, startZeroCap: 50000 }), [50000, 53000]);
});

test('window adaptation keeps the late pass rate between 30% and 60%, clamped to [5000, target]', () => {
  assert.strictEqual(rules.adaptWindow(25000, 0.61, 1e6), 37500);
  assert.strictEqual(rules.adaptWindow(25000, 0.29, 1e6), 16667);
  assert.strictEqual(rules.adaptWindow(25000, 0.45, 1e6), 25000);
  assert.strictEqual(rules.adaptWindow(6000, 0.1, 1e6), 5000);
  assert.strictEqual(rules.adaptWindow(900000, 0.9, 1e6), 1e6);
  assert.strictEqual(rules.adaptWindow(3000, 0.1, 4000), 4000, 'a target below 5000 is the floor');
});

// --- per-pair episodes -------------------------------------------------------
test('generation draws: per-pair episodes are deterministic, shared within a pair, and off by default', () => {
  const c = { popSize: 8, episodes: 3, pairEpisodes: true };
  const draw = (cfg, cur) => {
    const rng = createRng(1234);
    const out = rules.drawGeneration(rng, cfg, cur);
    return { ...out, state: rng.getState() };
  };
  const a = draw(c, null), b = draw(c, null);
  assert.deepStrictEqual(a, b, 'same RNG state, same draws');
  assert.strictEqual(new Set(a.pairEp.map(p => p.seeds.join())).size, 4, 'each pair has its own seeds');
  // Off: the original stream (noise seeds, then one shared seed set).
  const off = draw({ ...c, pairEpisodes: false }, null);
  const rng = createRng(1234);
  const noise = Array.from({ length: 4 }, () => Math.floor(rng() * 4294967296));
  const shared = Array.from({ length: 3 }, () => env.trainSeed(rng));
  assert.deepStrictEqual(off.noiseSeeds, noise);
  assert.ok(off.pairEp.every(p => p.seeds === off.pairEp[0].seeds));
  assert.deepStrictEqual(off.pairEp[0].seeds, shared);
  assert.strictEqual(off.state, rng.getState());
  assert.deepStrictEqual(a.noiseSeeds, noise, 'noise seeds are drawn first either way');
  // With the curriculum, each pair's starts are its own seeds' starts.
  const cur = { target: 1000000, window: 25000, startZeroFrac: 0.34, startMin: 2000, startDist: 'points', startZeroCap: 5000 };
  const withCur = draw(c, cur);
  for (const p of withCur.pairEp) {
    assert.deepStrictEqual(p.starts, env.startScores(p.seeds, cur));
    assert.deepStrictEqual(p.caps, env.episodeCaps(p.starts, cur));
  }
});

test('pair-difference weights: D_k over the std of all fitness values', () => {
  const f = [3, 1, 2, 2, 10, 4];
  const mean = 22 / 6, std = Math.sqrt(f.reduce((a, x) => a + (x - mean) ** 2, 0) / 6);
  const w = es.pairDiffWeights(f);
  close(w[0], 2 / std); assert.strictEqual(w[1], 0); close(w[2], 6 / std);
  const g = es.esGradient([5, 6, 7], f, 10, 0.1, { weights: 'pairdiff' });
  const e5 = es.noise(5, 10), e7 = es.noise(7, 10);
  for (let i = 0; i < 10; i++) close(g[i], (w[0] * e5[i] + w[2] * e7[i]) / (6 * 0.1), 1e-12);
});

test('Clopper-Pearson interval', () => {
  const r = rules.clopperPearson(25, 25);
  close(r.lower, Math.pow(0.025, 1 / 25), 1e-9);
  assert.strictEqual(r.upper, 1);
  const r2 = rules.clopperPearson(24, 25);
  close(r2.lower, 0.7965, 1e-4); close(r2.upper, 0.9990, 1e-4);
});

// --- gated curriculum ------------------------------------------------------------
test('gated window adaptation: shrink while the opening fails, late rule once it passes, hold between', () => {
  const T = 1e6;
  const acc = (zp, zt, lp, lt) => ({ zeroPassed: zp, zeroTotal: zt, latePassed: lp, lateTotal: lt });
  // Opening failing: shrink even though every late window passed.
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(1, 10, 10, 10), T), 20000);
  // Opening passed: the late rule applies (grow / shrink / hold).
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(7, 10, 10, 10), T), 45000);
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(7, 10, 1, 10), T), 20000);
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(7, 10, 5, 10), T), 30000);
  // In between (30%..60%): hold.
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(5, 10, 10, 10), T), 30000);
  // No start-0 episodes: the late rule alone; no episodes at all: hold.
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(0, 0, 10, 10), T), 45000);
  assert.strictEqual(rules.gatedAdaptWindow(30000, acc(0, 0, 0, 0), T), 30000);
  // A shrink never grows a window already below the 5000 floor.
  assert.strictEqual(rules.gatedAdaptWindow(500, acc(0, 10, 0, 10), T), 500);
  assert.strictEqual(rules.gatedAdaptWindow(500, acc(10, 10, 0, 10), T), 500);
  assert.strictEqual(rules.gatedAdaptWindow(6000, acc(0, 10, 0, 0), T), 5000);
  assert.deepStrictEqual(rules.emptyAdaptAcc(), acc(0, 0, 0, 0));
});

test('gated window adaptation: never grows while the opening fails; w_t <= w0 * 1.5^(t/N) with every N', () => {
  const T = 1e7, N = 5, w0 = 5000;
  const rng = createRng(7);
  for (let trial = 0; trial < 200; trial++) {
    let w = w0, decisions = 0, bound = w0;
    for (let t = 1; t <= 300; t++) {
      if (t % N) continue;
      const zt = Math.floor(rng() * 20), lt = Math.floor(rng() * 20);
      const a = { zeroTotal: zt, zeroPassed: Math.floor(rng() * (zt + 1)), lateTotal: lt, latePassed: Math.floor(rng() * (lt + 1)) };
      const next = rules.gatedAdaptWindow(w, a, T);
      if (a.zeroTotal > 0 && a.zeroPassed / a.zeroTotal < 0.3) assert.ok(next <= w, 'grew while the opening failed');
      decisions++;
      bound = Math.round(bound * 1.5);   // the largest window reachable after `decisions` decisions
      assert.ok(next <= Math.min(T, bound), `window ${next} above bound ${bound}`);
      assert.ok(next <= w0 * 1.5 ** (t / N) + decisions, 'rate limit (allowing 0.5 per rounding)');
      w = next;
    }
  }
});

test('geometric-mean fitness: equals the mean when all stages are equal; a failing stage dominates; floor applied', () => {
  close(rules.geomeanFitness([0.4, 0.4, 0.4, 0.4], [500, 500, 500, 500]), 0.4, 1e-12);
  close(rules.geomeanFitness([1, 1, 1, 1], [500, 500, 500, 500]), 1, 1e-12);
  // One stage at zero, three passed: floored at 1/span, F = (1/span)^(1/E), below the mean 0.75.
  const one = rules.geomeanFitness([0, 1, 1, 1], [500, 2000, 2000, 2000]);
  close(one, (1 / 500) ** (1 / 4), 1e-12);
  assert.ok(one < 0.75);
  // Several failing stages drive it towards 0.
  assert.ok(rules.geomeanFitness([0, 0, 0, 1], [5000, 5000, 5000, 5000]) < 0.002);
  // The floor is per episode (its own span) and only applies below it.
  close(rules.geomeanFitness([0, 0], [100, 400]), Math.sqrt(1 / 100 * 1 / 400), 1e-12);
  close(rules.geomeanFitness([0.5, 0.002], [100, 400]), Math.sqrt(0.5 * 0.0025), 1e-12);
  assert.ok(Number.isFinite(rules.geomeanFitness([0, 0], [1, 1])));
});

test('checkConfig: --fitness-agg and --window-adapt-every', () => {
  const { DEFAULTS, checkConfig } = require('../trainer/cli.js');
  const cur = { window: 500, target: 3000, startDist: 'points', startZeroCap: 500 };
  checkConfig({ ...DEFAULTS, ...cur, fitnessAgg: 'geomean', windowAdapt: true, windowAdaptEvery: 3 });
  assert.throws(() => checkConfig({ ...DEFAULTS, fitnessAgg: 'geomean' }), /geomean needs/);
  assert.throws(() => checkConfig({ ...DEFAULTS, ...cur, fitnessAgg: 'median' }), /fitness-agg/);
  assert.throws(() => checkConfig({ ...DEFAULTS, windowAdaptEvery: 0 }), /window-adapt-every/);
  assert.throws(() => checkConfig({ ...DEFAULTS, windowAdaptEvery: 2.5 }), /window-adapt-every/);
});
