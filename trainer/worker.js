'use strict';
// Worker thread: evaluates jobs { id, noiseSeed, sign, seeds[, starts, caps] } for the current
// setup { theta, layers, actionRepeat, gameVersion, sigma, target } and replies with scores.
// sigma is a number or a per-parameter vector (es.js sigmaVector).
// With starts (training curriculum), episode j starts at starts[j] and ends at caps[j].
const { parentPort } = require('worker_threads');
const { runEpisode } = require('./env.js');
const { perturb } = require('./es.js');

let setup = null;

parentPort.on('message', (msg) => {
  if (msg.type === 'setup') {
    setup = msg.setup;
    return;
  }
  if (msg.type === 'job') {
    const { id, noiseSeed, sign, seeds, starts, caps } = msg.job;
    const params = perturb(setup.theta, noiseSeed, sign, setup.sigma);
    const model = { layers: setup.layers, params, actionRepeat: setup.actionRepeat, gameVersion: setup.gameVersion };
    const scores = [], distances = [];
    let frames = 0;
    for (let j = 0; j < seeds.length; j++) {
      const r = starts ? runEpisode(model, seeds[j], caps[j], starts[j]) : runEpisode(model, seeds[j], setup.target);
      scores.push(r.score);
      distances.push(r.distance);
      frames += r.frames;
    }
    parentPort.postMessage({ type: 'result', result: { id, scores, distances, frames } });
  }
});
