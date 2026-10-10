'use strict';
// Game v2: progressively harder up to 1,000,000 points, still always survivable.
const test = require('node:test');
const assert = require('node:assert');
const C = require('../core.js');
const { snapshot, decodeKeys, readTraces } = require('./golden-format.js');
const { createSolver } = require('./solver.js');

function play(version, seed, capScore, startScore) {
  const g = C.createGame({ rng: C.createRng(seed), version, startScore });
  const act = createSolver();
  const capDistance = capScore / C.CFG.scoreCoef;
  do g.step(act(g)); while (g.state === 'running' && g.distance < capDistance);
  return g;
}

test('v1 is the default; unknown versions are rejected', () => {
  assert.strictEqual(C.createGame({ rng: C.createRng(1) }).version, 'v1');
  assert.strictEqual(C.createGame({ rng: C.createRng(1), version: 'v2' }).version, 'v2');
  assert.throws(() => C.createGame({ rng: C.createRng(1), version: 'v3' }), /unknown game version/);
});

test('v2 is exactly v1 below 1,000 points (golden traces replayed on v2)', () => {
  let compared = 0;
  for (const trace of readTraces()) {
    const base = C.createRng(trace.seed);
    let calls = 0;
    const game = C.createGame({ rng: () => { calls++; return base(); }, version: 'v2' });
    const view = { get state() { return game.state; }, dino: game.dino, get speed() { return game.speed; },
      get distance() { return game.distance; }, get obstacles() { return game.obstacles; }, get rngCalls() { return calls; } };
    for (let f = 0; f < trace.rows.length; f++) {
      game.step(decodeKeys(trace.inputs[f]));
      if (Math.floor(trace.rows[f][7] * C.CFG.scoreCoef) >= 1000) break;
      assert.deepStrictEqual(snapshot(view), trace.rows[f], `${trace.name} frame ${f}`);
      compared++;
    }
  }
  assert.ok(compared > 30000, `compared ${compared} frames`);   // 34,670 frames below 1,000 points
});

test('difficulty schedule: log scale from 1k to 1M points, rules interpolate v1 -> v2', () => {
  const d = (s) => C.difficulty('v2', s);
  assert.strictEqual(d(0), 0);
  assert.strictEqual(d(1000), 0);
  assert.ok(Math.abs(d(10000) - 1 / 3) < 1e-12);
  assert.ok(Math.abs(d(100000) - 2 / 3) < 1e-12);
  assert.strictEqual(d(1000000), 1);
  assert.strictEqual(d(5000000), 1);
  assert.strictEqual(C.difficulty('v1', 1000000), 0);
  const r0 = C._internal.rulesAt(0, {}), r1 = C._internal.rulesAt(1, {});
  assert.deepStrictEqual([r0.maxSpeed, r0.gapCoef, r0.maxGapCoef, r0.minWindowFrames, r0.minGroundFrames, r0.pteroWeight, r0.pteroSpeedOffset, r0.maxGroup],
    [C.CFG.maxSpeed, C.CFG.gapCoef, C.CFG.maxGapCoef, C.CFG.minWindowFrames, C.CFG.minGroundFrames, 1, 0.8, 3]);
  assert.deepStrictEqual([r1.maxSpeed, r1.gapCoef, r1.maxGapCoef, r1.minWindowFrames, r1.minGroundFrames, r1.pteroWeight, r1.pteroSpeedOffset, r1.maxGroup],
    [C.V2.maxSpeed, C.V2.gapCoef, C.V2.maxGapCoef, C.V2.minWindowFrames, C.V2.minGroundFrames, C.V2.pteroWeight, C.V2.pteroSpeedOffset, 4]);
});

test('v2 ramps in play: level and speed rise with score, level 0 below 1,000', () => {
  const low = play('v2', 3, 999);
  assert.strictEqual(low.level, 0);
  assert.ok(low.speed <= C.CFG.maxSpeed);
  // After 2,000 points from each start, speed has caught up with that level's cap.
  const at = (score) => { const g = play('v2', 3, score + 2000, score); return { speed: g.speed, level: g.level }; };
  const pts = [at(1000), at(30000), at(300000), at(1000000)];
  for (let i = 1; i < pts.length; i++) {
    assert.ok(pts[i].speed > pts[i - 1].speed || pts[i].speed === C.V2.maxSpeed, JSON.stringify(pts));
    assert.ok(pts[i].level > pts[i - 1].level || pts[i].level === 1, JSON.stringify(pts));
  }
  assert.strictEqual(pts[3].speed, C.V2.maxSpeed);
  assert.strictEqual(pts[3].level, 1);
});

// The exact fairness-model controller must never crash: every v2 sequence is survivable.
test('v2 survivable: exact solver plays 0 -> 1M points without a crash', () => {
  for (const seed of [1, 2]) {
    const g = play('v2', seed, 1000000);
    assert.strictEqual(g.state, 'running', `seed ${seed} crashed at ${g.score()}`);
  }
});

test('v2 survivable at every difficulty level (exact solver, 12 seeds x 4 start levels)', () => {
  for (const start of [10000, 31623, 300000, 1000000]) {
    for (let seed = 1; seed <= 12; seed++) {
      const g = play('v2', 500 + seed, start + 20000, start);
      assert.strictEqual(g.state, 'running', `start ${start} seed ${seed} crashed at ${g.score()}`);
    }
  }
});

test('v1 also never crashes under the exact solver', () => {
  for (let seed = 1; seed <= 6; seed++) {
    const g = play('v1', seed, 30000);
    assert.strictEqual(g.state, 'running', `seed ${seed} crashed at ${g.score()}`);
  }
});
