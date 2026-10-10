'use strict';
// Throughput benchmark. Replays the recorded inputs of the long golden trace
// (9000 frames: all speeds up to max, pterodactyls, night) so the measurement
// covers representative gameplay rather than early crashes. In 'policy' mode the
// shared policy also observes and runs the MLP every frame (its choice is
// computed but the recorded inputs are applied, keeping the workload fixed).
// net = { hidden, wasm } picks the hidden layers (default [16]) and the wasm-net.js kernel.
const { isMainThread, parentPort, workerData } = require('worker_threads');
const { createGame, createRng } = require('../core.js');
const { OBS_DIM, observe, initParams, createNet } = require('../policy.js');
const { gaussian } = require('./es.js');
const { readTraces, decodeKeys } = require('../test/golden-format.js');

function benchSingle(mode, ms, net = {}) {
  const trace = readTraces().find(t => t.name === '10-solver-long');
  const keys = Array.from(trace.inputs, decodeKeys);
  const layers = [OBS_DIM, ...(net.hidden || [16]), 3];
  const params = initParams(layers, gaussian(createRng(7)));
  let x, decide;
  if (net.wasm) {
    const w = require('./wasm-net.js').createWasmNet(layers, params);
    x = w.input;
    decide = w.decide;
  } else {
    decide = createNet(layers, params);
    x = new Float64Array(OBS_DIM);
  }
  let frames = 0, sink = 0;
  const t0 = process.hrtime.bigint();
  const deadline = t0 + BigInt(ms) * 1000000n;
  while (process.hrtime.bigint() < deadline) {
    const game = createGame({ rng: createRng(trace.seed) });
    for (let f = 0; f < keys.length; f++) {
      if (mode === 'policy') sink += decide(observe(game, x));
      game.step(keys[f]);
    }
    frames += keys.length;
  }
  const secs = Number(process.hrtime.bigint() - t0) / 1e9;
  return { fps: frames / secs, frames, secs, sink };
}

if (!isMainThread && workerData) parentPort.postMessage(benchSingle(workerData.mode, workerData.ms, workerData.net));

module.exports = { benchSingle };
