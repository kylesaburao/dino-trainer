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

// Perturbation scale: a number (the same sigma for every parameter), or with
// mode 'fanin' a per-parameter vector in the policy.js layout (per layer: nOut x nIn
// weights row-major, then nOut biases). A weight's noise reaches its unit's
// pre-activation summed over fanIn inputs, so layer l's weights get
// sigma * sqrt(layers[0] / fanIn_l) (the first layer keeps sigma). Biases are not
// summed over inputs, so they keep sigma.
function sigmaVector(layers, sigma, mode) {
  if (mode === 'none') return sigma;
  if (mode !== 'fanin') throw new Error(`unknown --sigma-scale ${mode}`);
  let n = 0;
  for (let l = 1; l < layers.length; l++) n += layers[l] * layers[l - 1] + layers[l];
  const s = new Float64Array(n);
  let k = 0;
  for (let l = 1; l < layers.length; l++) {
    const w = sigma * Math.sqrt(layers[0] / layers[l - 1]);
    for (let i = 0; i < layers[l] * layers[l - 1]; i++) s[k++] = w;
    for (let i = 0; i < layers[l]; i++) s[k++] = sigma;
  }
  return s;
}

// theta + sign * sigma * eps(seed); sigma is a number or a per-parameter vector.
function perturb(theta, seed, sign, sigma) {
  const p = Float64Array.from(theta);
  if (sign === 0) return p;
  const eps = noise(seed, theta.length);
  if (typeof sigma === 'number') {
    for (let i = 0; i < p.length; i++) p[i] += sign * sigma * eps[i];
  } else {
    for (let i = 0; i < p.length; i++) p[i] += sign * sigma[i] * eps[i];
  }
  return p;
}

// Centred ranks in [-0.5, 0.5]. ties 'index' breaks ties by index (the original
// rule); 'mid' gives tied values the mean of their ranks, so a tied antithetic pair
// gets weight exactly 0.
function centredRanks(xs, ties = 'index') {
  const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
  const r = new Float64Array(xs.length);
  const n = xs.length;
  if (ties === 'index') {
    idx.forEach((i, k) => { r[i] = n > 1 ? k / (n - 1) - 0.5 : 0; });
    return r;
  }
  if (ties !== 'mid') throw new Error(`unknown --rank-ties ${ties}`);
  for (let a = 0; a < n;) {
    let b = a + 1;
    while (b < n && xs[idx[b]] === xs[idx[a]]) b++;
    const k = (a + b - 1) / 2;
    for (let j = a; j < b; j++) r[idx[j]] = n > 1 ? k / (n - 1) - 0.5 : 0;
    a = b;
  }
  return r;
}

// Pair weights for per-pair episodes (ARS V1 style, no top-b): each pair's fitness
// difference over the std of all fitness values. Pairs play different episodes, so
// global ranks would mostly weight pairs by their level, not their difference.
function pairDiffWeights(fitness) {
  const n = fitness.length;
  const mean = fitness.reduce((a, x) => a + x, 0) / n;
  const std = Math.sqrt(fitness.reduce((a, x) => a + (x - mean) * (x - mean), 0) / n);
  const w = new Float64Array(n / 2);
  for (let k = 0; k < w.length; k++) w[k] = (fitness[2 * k] - fitness[2 * k + 1]) / (std + 1e-12);
  return w;
}

// Antithetic ES gradient estimate. fitness[2k] is +eps_k, fitness[2k+1] is -eps_k.
// g_i = sum_k w_k eps_ki / (fitness.length * sigma_i), with w_k the centred-rank
// difference of pair k (opts.ties), or with opts.weights 'pairdiff' pairDiffWeights.
function esGradient(noiseSeeds, fitness, n, sigma, opts = {}) {
  let w;
  if (opts.weights === 'pairdiff') {
    w = pairDiffWeights(fitness);
  } else {
    const ranks = centredRanks(fitness, opts.ties);
    w = noiseSeeds.map((s, k) => ranks[2 * k] - ranks[2 * k + 1]);
  }
  const g = new Float64Array(n);
  const eps = new Float64Array(n);
  for (let k = 0; k < noiseSeeds.length; k++) {
    if (w[k] === 0) continue;
    noise(noiseSeeds[k], n, eps);
    for (let i = 0; i < n; i++) g[i] += w[k] * eps[i];
  }
  if (typeof sigma === 'number') {
    const scale = 1 / (fitness.length * sigma);
    for (let i = 0; i < n; i++) g[i] *= scale;
  } else {
    for (let i = 0; i < n; i++) g[i] *= 1 / (fitness.length * sigma[i]);
  }
  return g;
}

// Adam ascent step. Mutates theta and state {m, v, t}.
//   decay 'coupled' (the original): L2 inside the gradient, g = grad - wd * theta,
//     which Adam's normalisation makes nearly inert.
//   decay 'decoupled' (AdamW): plain Adam on grad, then theta *= 1 - lr * wd, so wd
//     is a per-step shrink rate (~0.1 is sensible, vs ~0.005 coupled).
function adamStep(theta, grad, state, lr, weightDecay, decay = 'coupled', b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  if (decay !== 'coupled' && decay !== 'decoupled') throw new Error(`unknown --decay ${decay}`);
  state.t++;
  const c1 = 1 - Math.pow(b1, state.t), c2 = 1 - Math.pow(b2, state.t);
  const l2 = decay === 'coupled' ? weightDecay : 0;
  const shrink = decay === 'decoupled' ? 1 - lr * weightDecay : 1;
  for (let i = 0; i < theta.length; i++) {
    const g = grad[i] - l2 * theta[i];
    state.m[i] = b1 * state.m[i] + (1 - b1) * g;
    state.v[i] = b2 * state.v[i] + (1 - b2) * g * g;
    theta[i] += lr * (state.m[i] / c1) / (Math.sqrt(state.v[i] / c2) + eps);
    if (shrink !== 1) theta[i] *= shrink;
  }
}

module.exports = { gaussian, noise, sigmaVector, perturb, centredRanks, pairDiffWeights, esGradient, adamStep };
