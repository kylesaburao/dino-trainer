'use strict';
// Training-only windowed random-start curriculum (trainer/env.js).
const test = require('node:test');
const assert = require('node:assert');
const { CFG } = require('../core.js');
const { startScores, windowFitness, runEpisode } = require('../trainer/env.js');
const model = require('../model.js');

const C = { target: 1000000, window: 50000, startZeroFrac: 0.5, startMin: 2000 };
const seedsFor = (g, n) => Array.from({ length: n }, (_, j) => (g * 7919 + j * 104729 + 1) & 0x3FFFFFFF);

test('start levels are a pure function of the episode seeds', () => {
  const seeds = seedsFor(3, 4);
  assert.deepStrictEqual(startScores(seeds, C), startScores([...seeds], { ...C }));
  // Each late start depends only on its own seed, not on its neighbours.
  const a = startScores([seeds[0], seeds[1], 11, 12], { ...C, startZeroFrac: 0 });
  const b = startScores([seeds[0], seeds[1], 13, 14], { ...C, startZeroFrac: 0 });
  assert.deepStrictEqual(a.slice(0, 2), b.slice(0, 2));
});

test('zero starts are stratified and late starts are log-uniform in [startMin, target - window]', () => {
  let zeros = 0, total = 0, lowHalf = 0, late = 0;
  for (let g = 0; g < 2000; g++) {
    const s = startScores(seedsFor(g, 4), C);
    const z = s.filter(x => x === 0).length;
    assert.strictEqual(z, 2, 'frac 0.5 of 4 episodes is exactly 2 zero starts');
    assert.deepStrictEqual(s.slice(0, z), Array(z).fill(0));
    for (const x of s.slice(z)) {
      assert.ok(Number.isInteger(x) && x >= C.startMin && x <= C.target - C.window, `start ${x}`);
      late++;
      if (x < Math.sqrt(C.startMin * (C.target - C.window))) lowHalf++;
    }
    zeros += z; total += s.length;
  }
  assert.strictEqual(zeros / total, 0.5);
  assert.ok(Math.abs(lowHalf / late - 0.5) < 0.03, `log-midpoint split ${lowHalf / late}`);
  // A fractional share is met in expectation: 0.3 of 4 episodes = 1 or 2 zero starts.
  let z3 = 0;
  for (let g = 0; g < 4000; g++) {
    const z = startScores(seedsFor(g, 4), { ...C, startZeroFrac: 0.3 }).filter(x => x === 0).length;
    assert.ok(z === 1 || z === 2);
    z3 += z;
  }
  assert.ok(Math.abs(z3 / 4000 / 4 - 0.3) < 0.01, `zero share ${z3 / 16000}`);
});

test('window fitness is progress through the window, capped at 1', () => {
  const d = (score) => score / CFG.scoreCoef;
  assert.strictEqual(windowFitness(d(0), 0, 50000), 0);
  assert.strictEqual(windowFitness(d(25000), 0, 50000), 0.5);
  assert.strictEqual(windowFitness(d(225000), 200000, 50000), 0.5);
  assert.strictEqual(windowFitness(d(260000), 200000, 50000), 1);
  assert.strictEqual(windowFitness(d(200000) - 1e-9, 200000, 50000), 0);
});

test('a late-start episode starts at its level and stops at the end of its window', () => {
  const start = 20000, window = 2000;
  const r = runEpisode(model, 12345, start + window, start);
  assert.ok(r.distance * CFG.scoreCoef >= start, 'starts at the start level');
  assert.ok(r.score <= start + window && r.score >= start);
  assert.ok(windowFitness(r.distance, start, window) <= 1);
  // Without a start level the episode is the full game, unchanged.
  assert.deepStrictEqual(runEpisode(model, 12345, 500), runEpisode(model, 12345, 500, 0));
});
