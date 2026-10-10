'use strict';
// Worker thread: evaluates jobs { id, noiseSeed, sign, seeds } for the current
// setup { theta, layers, actionRepeat, gameVersion, sigma, target } and replies with scores.
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
    const { id, noiseSeed, sign, seeds } = msg.job;
    const params = perturb(setup.theta, noiseSeed, sign, setup.sigma);
    const model = { layers: setup.layers, params, actionRepeat: setup.actionRepeat, gameVersion: setup.gameVersion };
    const scores = [], distances = [];
    let frames = 0;
    for (const seed of seeds) {
      const r = runEpisode(model, seed, setup.target);
      scores.push(r.score);
      distances.push(r.distance);
      frames += r.frames;
    }
    parentPort.postMessage({ type: 'result', result: { id, scores, distances, frames } });
  }
});
