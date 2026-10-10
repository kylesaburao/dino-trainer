'use strict';
// OpenAI-ES pieces: seeded Gaussian noise, centred ranks, Adam.
const { createRng } = require('../core.js');

// Standard normal sampler (Box-Muller) on a seeded mulberry32 stream.
function gaussian(rng) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = rng();
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

// The perturbation for a noise seed; identical in the main thread and workers.
function noise(seed, n, out) {
  const g = gaussian(createRng(seed));
  const eps = out || new Float64Array(n);
  for (let i = 0; i < n; i++) eps[i] = g();
  return eps;
}

// theta + sign * sigma * eps(seed)
function perturb(theta, seed, sign, sigma) {
  const p = Float64Array.from(theta);
  if (sign === 0) return p;
  const eps = noise(seed, theta.length);
  for (let i = 0; i < p.length; i++) p[i] += sign * sigma * eps[i];
  return p;
}

// Centred ranks in [-0.5, 0.5] (ties broken by index).
function centredRanks(xs) {
  const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
  const r = new Float64Array(xs.length);
  const n = xs.length;
  idx.forEach((i, k) => { r[i] = n > 1 ? k / (n - 1) - 0.5 : 0; });
  return r;
}

// Antithetic ES gradient estimate. fitness[2k] is +eps_k, fitness[2k+1] is -eps_k.
function esGradient(noiseSeeds, fitness, n, sigma) {
  const ranks = centredRanks(fitness);
  const g = new Float64Array(n);
  const eps = new Float64Array(n);
  for (let k = 0; k < noiseSeeds.length; k++) {
    const w = ranks[2 * k] - ranks[2 * k + 1];
    if (w === 0) continue;
    noise(noiseSeeds[k], n, eps);
    for (let i = 0; i < n; i++) g[i] += w * eps[i];
  }
  const scale = 1 / (fitness.length * sigma);
  for (let i = 0; i < n; i++) g[i] *= scale;
  return g;
}

// Adam ascent step with L2 weight decay. Mutates theta and state {m, v, t}.
function adamStep(theta, grad, state, lr, weightDecay, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  state.t++;
  const c1 = 1 - Math.pow(b1, state.t), c2 = 1 - Math.pow(b2, state.t);
  for (let i = 0; i < theta.length; i++) {
    const g = grad[i] - weightDecay * theta[i];
    state.m[i] = b1 * state.m[i] + (1 - b1) * g;
    state.v[i] = b2 * state.v[i] + (1 - b2) * g * g;
    theta[i] += lr * (state.m[i] / c1) / (Math.sqrt(state.v[i] / c2) + eps);
  }
}

module.exports = { gaussian, noise, perturb, centredRanks, esGradient, adamStep };
