'use strict';
// trainer/wasm-net.js (train --wasm) must be the same function as policy.js: every raw
// logit bit-for-bit, every decision, and whole episodes frame for frame.
const test = require('node:test');
const assert = require('node:assert');
const { createRng } = require('../core.js');
const P = require('../policy.js');
const { gaussian, perturb } = require('../trainer/es.js');
const { runEpisode } = require('../trainer/env.js');
const W = require('../trainer/wasm-net.js');

// policy.js's unit sums, one row at a time, returning the raw output layer.
function referenceLogits(layers, params, x) {
  let input = x, k = 0;
  for (let l = 1; l < layers.length; l++) {
    const nIn = layers[l - 1], nOut = layers[l], out = new Float64Array(nOut), bBase = k + nOut * nIn;
    for (let i = 0; i < nOut; i++) {
      let s = params[bBase + i];
      for (let j = 0; j < nIn; j++) s += params[k + i * nIn + j] * input[j];
      out[i] = l === layers.length - 1 ? s : Math.tanh(s);
    }
    k = bBase + nOut;
    input = out;
  }
  return input;
}

// Odd widths, widths below / at / above the 8-unit block, and 1-4 hidden layers.
const HIDDEN = [[16], [32, 32], [33, 17], [64, 64, 64], [8, 8, 8, 8], [1], [5, 9], [7]];

test('wasm SIMD is available', () => { W.assertAvailable(); });

test('wasm net: raw logits bit-equal and decisions equal to policy.js (random weights and inputs)', () => {
  const g = gaussian(createRng(23));
  const u = createRng(29);
  for (const hidden of HIDDEN) {
    const layers = [P.OBS_DIM, ...hidden, 3];
    const params = P.initParams(layers, g);
    for (let i = 0; i < params.length; i++) params[i] = params[i] * (1 + 2 * u()) + 0.3 * g();   // biases, large weights
    const js = P.createNet(layers, params), wasm = W.createWasmNet(layers, params);
    const x = new Float64Array(P.OBS_DIM);
    for (let n = 0; n < 4000; n++) {
      const scale = [0.5, 2, 30][n % 3];   // 30 saturates tanh
      for (let j = 0; j < x.length; j++) x[j] = u() < 0.2 ? 0 : (3 * u() - 1) * scale;
      wasm.input.set(x);
      const a = wasm.decide();
      const ref = referenceLogits(layers, params, x), got = wasm.logits();
      for (let i = 0; i < ref.length; i++) {
        if (!Object.is(ref[i], got[i])) assert.fail(`layers ${layers.join('-')}, input ${n}, logit ${i}: ${got[i]} != ${ref[i]}`);
      }
      assert.strictEqual(a, js(x), `layers ${layers.join('-')}, input ${n}`);
    }
  }
});

test('wasm net: ties in the output resolve to the lowest index, like policy.js', () => {
  const layers = [P.OBS_DIM, 3];
  const params = new Float64Array(P.numParams(layers));   // all-zero logits
  const wasm = W.createWasmNet(layers, params), js = P.createNet(layers, params);
  assert.strictEqual(wasm.decide(), 0);
  assert.strictEqual(js(wasm.input), 0);
  params[P.OBS_DIM * 3 + 1] = params[P.OBS_DIM * 3 + 2] = 1;   // logits 0, 1, 1
  assert.strictEqual(W.createWasmNet(layers, params).decide(), 1);
  assert.strictEqual(P.createNet(layers, params)(wasm.input), 1);
});

// Plays one episode, recording the action index of every frame.
function playRecorded(model, seed, cap, start, makeAgent) {
  const actions = [];
  const record = (agent) => ({ act: (g) => { const a = agent.act(g); actions.push(P.ACTIONS.indexOf(a)); return a; } });
  const r = runEpisode(model, seed, cap, start, (m) => record(makeAgent(m)));
  return { ...r, actions };
}

function assertSameEpisode(model, seed, cap, start, label) {
  const js = playRecorded(model, seed, cap, start, P.createAgent);
  const wasm = playRecorded(model, seed, cap, start, W.wasmAgents(model));
  assert.ok(!js.actions.includes(-1), 'agent returned a non-ACTIONS object');
  assert.strictEqual(wasm.frames, js.frames, `${label}: frames`);
  assert.strictEqual(wasm.score, js.score, `${label}: score`);
  assert.ok(Object.is(wasm.distance, js.distance), `${label}: distance`);
  assert.deepStrictEqual(wasm.actions, js.actions, `${label}: actions`);
  return js;
}

test('wasm agent plays whole episodes identically: the shipped model', () => {
  const model = require('../model.js');
  let frames = 0;
  for (const seed of [1, 7, 4242]) frames += assertSameEpisode(model, seed, 3000, undefined, `model.js seed ${seed}`).frames;
  frames += assertSameEpisode(model, 99, 6000, 3000, 'model.js late start').frames;
  assert.ok(frames > 20000, `only ${frames} frames played`);
});

test('wasm agent plays whole episodes identically: perturbed trained model, action repeat 1 and 3', () => {
  // A fixed 44-32-32-3 trained model, not the root model.js: the frame-count floor
  // below needs a model that survives sigma-0.1 noise, which a small shipped net may not.
  const shipped = require('../runs/bc-v2-h32x32/best.json');
  let frames = 0;
  for (const [noiseSeed, repeat] of [[1, 1], [2, 3], [3, 1]]) {
    const model = { ...shipped, params: perturb(Float64Array.from(shipped.params), noiseSeed, 1, 0.1), actionRepeat: repeat };
    for (const seed of [5, 6]) frames += assertSameEpisode(model, seed, 3000, undefined, `perturbed ${noiseSeed} x${repeat} seed ${seed}`).frames;
  }
  assert.ok(frames > 10000, `only ${frames} frames played`);
});

test('wasm agent plays whole episodes identically: random params, action repeat, late starts', () => {
  const g = gaussian(createRng(5));
  const cases = [{ hidden: [32, 32], repeat: 1 }, { hidden: [33, 17], repeat: 3 }, { hidden: [8, 8, 8, 8], repeat: 2 }];
  for (const { hidden, repeat } of cases) {
    const layers = [P.OBS_DIM, ...hidden, 3];
    const model = { layers, params: P.initParams(layers, g), actionRepeat: repeat, gameVersion: 'v2' };
    for (const seed of [3, 11]) {
      assertSameEpisode(model, seed, 2000, undefined, `${layers.join('-')} x${repeat} seed ${seed}`);
      assertSameEpisode(model, seed, 2600, 2100, `${layers.join('-')} x${repeat} seed ${seed} start 2100`);
    }
  }
});

test('wasm agent validates the model like policy.js createAgent', () => {
  assert.throws(() => W.createWasmAgent({ layers: [3, 3], params: new Float64Array(12) }), /inputs/);
  assert.throws(() => W.createWasmNet([P.OBS_DIM, 3], new Float64Array(5)), /params/);
  const layers = [P.OBS_DIM, 3];
  assert.throws(() => W.createWasmAgent({ format: 'x', layers, params: new Float64Array(P.numParams(layers)) }), /format/);
});
