'use strict';
// Worker thread: evaluates jobs { id, noiseSeed, sign, seeds[, starts, caps] } for the current
// setup { theta, layers, actionRepeat, gameVersion, sigma, target[, wasm] } and replies with scores.
// With setup.wasm the network runs on wasm-net.js (bit-identical decisions, built once per job).
// sigma is a number or a per-parameter vector (es.js sigmaVector).
// With starts (training curriculum), episode j starts at starts[j] and ends at caps[j].
// With ramp (natural-ramp starts), episodes with ramp[j] are played by the solver up to
// starts[j] (trainer/ramp.js), then by the candidate for `window` points; the result
// then also carries every episode's actual starts and caps and the teacher's frames.
const { parentPort } = require('worker_threads');
const { runEpisode } = require('./env.js');
const { perturb } = require('./es.js');

let setup = null;
let wasmAgents = null;
let ramps = null;

parentPort.on('message', (msg) => {
  if (msg.type === 'setup') {
    setup = msg.setup;
    if (setup.wasm && !wasmAgents) wasmAgents = require('./wasm-net.js').wasmAgents;
    return;
  }
  if (msg.type === 'job') {
    const { id, noiseSeed, sign, seeds, starts, caps, ramp, window } = msg.job;
    if (ramp && !ramps) ramps = require('./ramp.js');
    const params = perturb(setup.theta, noiseSeed, sign, setup.sigma);
    const model = { layers: setup.layers, params, actionRepeat: setup.actionRepeat, gameVersion: setup.gameVersion };
    const makeAgent = setup.wasm ? wasmAgents(model) : undefined;
    const scores = [], distances = [];
    let frames = 0, teacherFrames = 0;
    const actualStarts = [], actualCaps = [];
    for (let j = 0; j < seeds.length; j++) {
      const r = ramp && ramp[j] ? ramps.runRampEpisode(model, seeds[j], starts[j], window, setup.target, makeAgent)
        : starts ? runEpisode(model, seeds[j], caps[j], starts[j], makeAgent)
          : runEpisode(model, seeds[j], setup.target, undefined, makeAgent);
      scores.push(r.score);
      distances.push(r.distance);
      frames += r.frames;
      if (ramp) {
        actualStarts.push(ramp[j] ? r.start : starts[j]);
        actualCaps.push(ramp[j] ? r.cap : caps[j]);
        teacherFrames += r.teacherFrames || 0;
      }
    }
    const result = ramp ? { id, scores, distances, frames, starts: actualStarts, caps: actualCaps, teacherFrames }
      : { id, scores, distances, frames };
    parentPort.postMessage({ type: 'result', result });
  }
});
