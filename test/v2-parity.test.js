'use strict';
// Rules-v2 parity gate: core.js must reproduce, exactly, the v2 golden traces recorded
// from commit 10cd0e0 (before rules v3; see test/record-golden-v2.js). Rules v3 shares
// the v2 code paths, so this proves the v3 change left v2 untouched. Rules v3 must also
// reproduce every v2 frame below 100,000 points.
const test = require('node:test');
const assert = require('node:assert');
const C = require('../core.js');
const { snapshot, fieldName, decodeKeys, readTraces, GOLDEN_V2_DIR } = require('./golden-format.js');

const traces = readTraces(GOLDEN_V2_DIR);

function replay(trace, version, stopAt = Infinity) {
  const base = C.createRng(trace.seed);
  let calls = 0;
  const game = C.createGame({ rng: () => { calls++; return base(); }, version, startScore: trace.startScore });
  const view = { get state() { return game.state; }, dino: game.dino, get speed() { return game.speed; },
    get distance() { return game.distance; }, get obstacles() { return game.obstacles; }, get rngCalls() { return calls; } };
  let compared = 0;
  for (let f = 0; f < trace.rows.length; f++) {
    game.step(decodeKeys(trace.inputs[f]));
    if (Math.floor(trace.rows[f][7] * C.CFG.scoreCoef) >= stopAt) break;
    const got = snapshot(view);
    const want = trace.rows[f];
    if (got.length !== want.length || got.some((x, i) => x !== want[i])) {
      const i = got.findIndex((x, k) => x !== want[k]);
      assert.fail(`${trace.name} on ${version}, frame ${f}: ${i < 0 ? 'length' : fieldName(i)} differs: got ${got[i]}, want ${want[i]}`);
    }
    compared++;
  }
  return compared;
}

test('v2 golden traces exist and cover the v2 schedule', () => {
  assert.ok(traces.length >= 7, 'expected the recorded v2 trace set');
  assert.ok(traces.every(t => t.version === 'v2' && t.baseline === '10cd0e0'));
  const all = traces.flatMap(t => t.rows);
  const has = (pred) => all.some(pred);
  const score = (r) => Math.floor(r[7] * C.CFG.scoreCoef);
  assert.ok(has(r => r[3] === 1), 'jumps');
  assert.ok(has(r => r[4] === 1), 'ducks');
  assert.ok(has(r => r[5] === 1), 'fast-drops');
  assert.ok(has(r => r[0] === 2), 'crashes');
  assert.ok(has(r => r[6] === C.V2.maxSpeed), 'v2 max speed');
  assert.ok(has(r => { for (let i = 10; i < r.length; i += 5) if (r[i] === 2) return true; return false; }), 'pterodactyls');
  assert.ok(has(r => { for (let i = 10; i < r.length; i += 5) if (r[i + 3] === 4) return true; return false; }), 'cactus groups of 4');
  assert.ok(has(r => score(r) > 1000 && score(r) < 2000), 'the 1k onset');
  assert.ok(has(r => score(r) >= 150000 && score(r) < 151000), 'through 150k');
  assert.ok(has(r => score(r) >= 1000000), 'past 1M (difficulty capped)');
});

for (const trace of traces) {
  test(`v2 parity: ${trace.name} (${trace.rows.length} frames)`, () => {
    assert.strictEqual(replay(trace, 'v2'), trace.rows.length);
  });
}
