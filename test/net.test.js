'use strict';
// createNet's blocked loop must decide exactly like the plain one-row-at-a-time
// MLP it replaced, so models trained before the change play identically.
const test = require('node:test');
const assert = require('node:assert');
const { createGame, createRng } = require('../core.js');
const P = require('../policy.js');
const { gaussian } = require('../trainer/es.js');

// The original createNet loop, kept here as the reference.
function referenceNet(layers, params) {
  const bufs = layers.map(n => new Float64Array(n));
  return function decide(x) {
    let input = x;
    let k = 0;
    for (let l = 1; l < layers.length; l++) {
      const nIn = layers[l - 1], nOut = layers[l], outB = bufs[l];
      const last = l === layers.length - 1;
      const bBase = k + nOut * nIn;
      for (let i = 0; i < nOut; i++) {
        let s = params[bBase + i];
        const row = k + i * nIn;
        for (let j = 0; j < nIn; j++) s += params[row + j] * input[j];
        outB[i] = last ? s : Math.tanh(s);
      }
      k = bBase + nOut;
      input = outB;
    }
    let best = 0;
    for (let i = 1; i < input.length; i++) if (input[i] > input[best]) best = i;
    return best;
  };
}

// Shapes cover full blocks of 4, remainders of 1-3 and outputs narrower than a block.
const SHAPES = [[P.OBS_DIM, 3], [P.OBS_DIM, 16, 3], [P.OBS_DIM, 32, 3], [P.OBS_DIM, 32, 32, 3],
  [P.OBS_DIM, 5, 3], [P.OBS_DIM, 7, 6, 3], [P.OBS_DIM, 9, 4], [P.OBS_DIM, 1, 3]];

test('blocked createNet decides exactly like the reference MLP (random weights and inputs)', () => {
  const g = gaussian(createRng(11));
  for (const layers of SHAPES) {
    const params = P.initParams(layers, g);
    for (let i = 0; i < params.length; i++) params[i] += 0.5 * g();   // non-zero biases, larger weights
    const net = P.createNet(layers, params), ref = referenceNet(layers, params);
    const x = new Float64Array(P.OBS_DIM);
    for (let n = 0; n < 5000; n++) {
      for (let j = 0; j < x.length; j++) x[j] = 2 * g();
      assert.strictEqual(net(x), ref(x), `layers ${layers.join('-')}, input ${n}`);
    }
  }
});

test('the shipped model decides exactly like the reference MLP over real play', () => {
  const model = require('../model.js');
  const net = P.createNet(model.layers, model.params), ref = referenceNet(model.layers, model.params);
  const x = new Float64Array(P.OBS_DIM);
  let frames = 0;
  for (let seed = 1; seed <= 5; seed++) {
    const game = createGame({ rng: createRng(seed), version: model.gameVersion || 'v1' });
    game.step(P.ACTIONS[1]);
    while (game.state === 'running' && frames < 20000 * seed) {
      const a = ref(P.observe(game, x));
      assert.strictEqual(net(x), a, `seed ${seed}, frame ${frames}`);
      game.step(P.ACTIONS[a]);
      frames++;
    }
  }
  assert.ok(frames > 50000, `only ${frames} frames played`);
});
