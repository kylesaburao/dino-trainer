'use strict';
// Shared helpers for the lockstep prototype (research/prereg-lockstep-tier-a.md).
// Research code only: reads core.js / policy.js / trainer/wasm-net.js, changes none of them.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const { createGame, createRng, CFG } = require(path.join(ROOT, 'core.js'));
const { ACTIONS, OBS_DIM, observe, numParams } = require(path.join(ROOT, 'policy.js'));
const { wasmAgents, moduleBytes } = require(path.join(ROOT, 'trainer', 'wasm-net.js'));
const { runEpisode, VALIDATION_SEEDS, validationSeeds } = require(path.join(ROOT, 'trainer', 'env.js'));

const loadModel = (file) => JSON.parse(fs.readFileSync(path.resolve(ROOT, file), 'utf8'));

// FNV-1a over action indices (one byte per frame).
const fnvInit = () => 0x811c9dc5;
const fnvStep = (h, b) => Math.imul(h ^ b, 0x01000193) >>> 0;

// A makeAgent for env.js runEpisode that wraps the wasm agent and records each frame's action
// index into the FNV hash (and optionally into `rec`). Returns { makeAgent, state }.
function hashingAgents(model, rec) {
  const inner = wasmAgents(model);
  const state = { hash: fnvInit(), frames: 0 };
  const makeAgent = (m) => {
    const a = inner(m);
    return {
      act(game) {
        const k = a.act(game);
        const idx = ACTIONS.indexOf(k);
        state.hash = fnvStep(state.hash, idx);
        if (rec) rec(idx, game);
        state.frames++;
        return k;
      },
    };
  };
  return { makeAgent, state };
}

// The reference path: env.js runEpisode with the wasm agent; returns score, frames, action hash.
function referenceEpisode(model, seed, cap) {
  const { makeAgent, state } = hashingAgents(model);
  const r = runEpisode(model, seed, cap, undefined, makeAgent);
  return { score: r.score, frames: r.frames, hash: state.hash };
}

// createWasmNet's layout and calls, with switches for the bound probes (tanh off).
function createProbeNet(layers, params, { tanh = true } = {}) {
  if (params.length !== numParams(layers)) throw new Error('params length');
  const nL = layers.length - 1;
  const meta = [];
  let maxPad = 2, wBytes = 0;
  for (let l = 1; l <= nL; l++) {
    const nIn = layers[l - 1], nOut = layers[l], nPad = (nOut + 1) & ~1;
    meta.push({ nIn, nOut, nPad, w: 0 });
    maxPad = Math.max(maxPad, nPad);
    wBytes += (nIn * nPad + nPad) * 8;
  }
  const align16 = (b) => (b + 15) & ~15;
  const X = 0, A = align16(layers[0] * 8), B = A + maxPad * 8, W0 = B + maxPad * 8;
  const pages = Math.max(1, Math.ceil((W0 + wBytes) / 65536));
  const mem = new WebAssembly.Memory({ initial: pages, maximum: pages });
  const { layerT } = new WebAssembly.Instance(new WebAssembly.Module(moduleBytes()), { env: { mem } }).exports;
  const f64 = new Float64Array(mem.buffer);
  let k = 0, w = W0;
  for (const m of meta) {
    const { nIn, nOut, nPad } = m;
    const base = w / 8, bBase = k + nOut * nIn;
    for (let i = 0; i < nOut; i++) {
      for (let j = 0; j < nIn; j++) f64[base + j * nPad + i] = params[k + i * nIn + j];
      f64[base + nIn * nPad + i] = params[bBase + i];
    }
    m.w = w;
    w += (nIn * nPad + nPad) * 8;
    k = bBase + nOut;
  }
  const nOutLast = layers[nL];
  const outAt = (l) => (l % 2 === 0 ? A : B);
  const outLast = outAt(nL - 1) / 8;
  const input = f64.subarray(0, layers[0]);
  const pre = [];   // hidden pre-activations of the last decide (for the tanh probe)
  return {
    input,
    decide() {
      let src = X;
      for (let l = 0; l < nL; l++) {
        const m = meta[l], dst = outAt(l);
        layerT(m.nIn, m.nPad, m.w, src, dst);
        if (tanh && l < nL - 1) {
          const q0 = dst / 8, q1 = q0 + m.nOut;
          for (let q = q0; q < q1; q++) f64[q] = Math.tanh(f64[q]);
        }
        src = dst;
      }
      let best = 0;
      for (let i = 1; i < nOutLast; i++) if (f64[outLast + i] > f64[outLast + best]) best = i;
      return best;
    },
    capturePre(out) {   // re-run, copying hidden pre-activations into out (array of Float64Array)
      let src = X;
      for (let l = 0; l < nL; l++) {
        const m = meta[l], dst = outAt(l);
        layerT(m.nIn, m.nPad, m.w, src, dst);
        if (l < nL - 1) {
          const q0 = dst / 8;
          out.push(f64.slice(q0, q0 + m.nOut));
          for (let q = q0; q < q0 + m.nOut; q++) f64[q] = Math.tanh(f64[q]);
        }
        src = dst;
      }
    },
  };
}

const now = () => Number(process.hrtime.bigint()) / 1e9;
function minOf(trials, fn) {
  const ts = [];
  for (let t = 0; t < trials; t++) { const t0 = now(); fn(); ts.push(now() - t0); }
  ts.sort((a, b) => a - b);
  return { min: ts[0], median: ts[ts.length >> 1], all: ts };
}

module.exports = {
  ROOT, createGame, createRng, CFG, ACTIONS, OBS_DIM, observe, numParams, runEpisode, VALIDATION_SEEDS, validationSeeds,
  loadModel, fnvInit, fnvStep, hashingAgents, referenceEpisode, createProbeNet, now, minOf, wasmAgents,
};
