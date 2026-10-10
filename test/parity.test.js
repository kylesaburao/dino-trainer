'use strict';
// Parity gate: the refactored core must reproduce, exactly, the golden traces
// recorded from baseline commit 2da34de (see test/record-golden.js).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const DinoCore = require('../core.js');
const { snapshot, fieldName, decodeKeys, readTraces } = require('./golden-format.js');

const traces = readTraces();

test('golden traces exist and cover the required behaviour', () => {
  assert.ok(traces.length >= 10, 'expected the recorded trace set');
  const all = traces.flatMap(t => t.rows);
  const has = (pred) => all.some(pred);
  assert.ok(has(r => r[3] === 1), 'jumps');
  assert.ok(has(r => r[4] === 1), 'ducks');
  assert.ok(has(r => r[5] === 1), 'fast-drops');
  assert.ok(has(r => r[0] === 2), 'crashes');
  assert.ok(has(r => r[6] === 13), 'max speed');
  assert.ok(has(r => { for (let i = 10; i < r.length; i += 5) if (r[i] === 2) return true; return false; }), 'pterodactyls');
  assert.ok(has(r => { for (let i = 10; i < r.length; i += 5) if (r[i + 3] > 1) return true; return false; }), 'cactus groups');
  assert.ok(has(r => Math.floor(r[7] * 0.025) >= 700), 'night mode (score >= 700)');
  // a restart after a crash
  assert.ok(traces.some(t => t.rows.some((r, i) => i > 0 && t.rows[i - 1][0] === 2 && r[0] === 1)), 'restarts');
});

for (const trace of traces) {
  test(`parity: ${trace.name} (${trace.rows.length} frames)`, () => {
    const base = DinoCore.createRng(trace.seed);
    let calls = 0;
    const game = DinoCore.createGame({ rng: () => { calls++; return base(); } });
    const view = { get state() { return game.state; }, dino: game.dino, get speed() { return game.speed; },
      get distance() { return game.distance; }, get obstacles() { return game.obstacles; }, get rngCalls() { return calls; } };
    for (let f = 0; f < trace.rows.length; f++) {
      game.step(decodeKeys(trace.inputs[f]));
      const got = snapshot(view);
      const want = trace.rows[f];
      if (got.length !== want.length || got.some((x, i) => x !== want[i])) {
        const i = got.findIndex((x, k) => x !== want[k]);
        assert.fail(`frame ${f}: ${i < 0 ? 'length' : fieldName(i)} differs: got ${got[i]}, want ${want[i]}`);
      }
    }
  });
}

test('sizes come from pixel-map rows exactly as the baseline sprite canvases', () => {
  assert.strictEqual(DinoCore.STAND_TOP, 98);
  assert.strictEqual(DinoCore.DUCK_H, 24);
  assert.strictEqual(DinoCore.CLOUD_W, 46);
  const t = Object.fromEntries(DinoCore.OBSTACLE_TYPES.map(o => [o.name, [o.w, o.h]]));
  assert.deepStrictEqual(t, { small: [16, 34], large: [24, 50], ptero: [46, 20] });
});

test('core and policy contain no DOM, clock or Math.random access', () => {
  for (const f of ['core.js', 'policy.js']) {
    const src = fs.readFileSync(path.join(__dirname, '..', f), 'utf8')
      .replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
    for (const word of ['Math.random', 'document.', 'window.', 'performance.', 'localStorage', 'AudioContext', 'Date.now', 'addEventListener', 'requestAnimationFrame']) {
      assert.ok(!src.includes(word), `${f} must not reference ${word}`);
    }
  }
});

test('independent games do not share state', () => {
  const a = DinoCore.createGame({ rng: DinoCore.createRng(1) });
  const b = DinoCore.createGame({ rng: DinoCore.createRng(1) });
  a.step({ jump: true });
  for (let i = 0; i < 300; i++) a.step({});
  b.step({});
  assert.strictEqual(b.state, 'idle');
  assert.strictEqual(b.distance, 0);
  assert.notStrictEqual(a.distance, 0);
});
