'use strict';
// netview.js: the network view's forward pass must pick the same action as
// policy.js createNet for every bundled model (any shape), so the view's
// mismatch warning only fires on a real disagreement.
const test = require('node:test');
const assert = require('node:assert');
const DinoPolicy = require('../policy.js');
const DinoNetView = require('../netview.js');
const MODELS = require('../models.js');

function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

test('netview: forward pass argmax equals policy.js createNet on every bundled model', () => {
  const r = rng(7);
  const x = new Float64Array(DinoPolicy.OBS_DIM);
  for (const m of MODELS) {
    const net = DinoPolicy.createNet(m.layers, m.params);
    const fwd = DinoNetView.createForward(m);
    assert.deepStrictEqual(fwd.acts.map(a => a.length), m.layers, m.id);
    for (let n = 0; n < 500; n++) {
      for (let i = 0; i < x.length; i++) x[i] = r() * 4 - 2;
      assert.strictEqual(fwd.run(x), net(x), `${m.id} input ${n}`);
    }
  }
});

test('netview: deep and narrow shapes up to the 16 x 32 cap', () => {
  const r = rng(11);
  const gauss = () => { const u = Math.max(r(), 1e-12); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const x = new Float64Array(DinoPolicy.OBS_DIM);
  for (const hidden of [[1], [32], [5, 7], Array(16).fill(32), Array(16).fill(3)]) {
    const layers = [DinoPolicy.OBS_DIM, ...hidden, 3];
    const params = DinoPolicy.initParams(layers, gauss);
    for (let i = 0; i < params.length; i++) params[i] += gauss() * 0.1;   // nonzero biases
    const net = DinoPolicy.createNet(layers, params);
    const fwd = DinoNetView.createForward({ layers, params });
    for (let n = 0; n < 200; n++) {
      for (let i = 0; i < x.length; i++) x[i] = r() * 4 - 2;
      assert.strictEqual(fwd.run(x), net(x), `${layers.join('-')} input ${n}`);
    }
  }
});

test('netview: input groups cover the observation in order', () => {
  let next = 0;
  for (const g of DinoNetView.GROUPS) { assert.strictEqual(g.from, next); next += g.names.length; }
  assert.strictEqual(next, DinoPolicy.OBS_DIM);
});
