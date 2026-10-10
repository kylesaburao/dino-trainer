// Shared AI policy: observation extraction + MLP inference + action mapping.
//
// The ONE implementation used by both the Node trainer and the browser AI mode,
// so the browser sees exactly what training saw. Classic <script> (sets the
// global `DinoPolicy`, needs core.js loaded first) and Node require().
//
// Observations are limited to what a player can see on the 600px canvas:
//   - dino height, vertical velocity, jumping/ducking flags, game speed
//   - the nearest OBS_K obstacles with x < W that the dino has not passed:
//     relative x, hitbox y/w/h, type, group size, pterodactyl height class and
//     on-screen velocity (speed + speedOffset, which a player sees as motion).
// Never read: obstacles at x >= W (placeFairly parks future obstacles there),
// o.gap, spawn history, the fairness solver, or RNG state.
(function (root) {
'use strict';

const Core = (typeof module !== 'undefined' && module.exports) ? require('./core.js') : root.DinoCore;
const { W, H, CFG, STAND_TOP, DINO_X, OBSTACLE_TYPES, obstacleBox, standBox } = Core;

const FORMAT = 'dino-mlp-v1';
const OBS_K = 3;               // nearest visible obstacles observed
const DINO_FEATURES = 5;
const PER_OBSTACLE = 13;
const OBS_DIM = DINO_FEATURES + OBS_K * PER_OBSTACLE;

// Action index -> per-frame key state (applied through the real input functions).
const ACTIONS = [
  { jump: false, down: false },  // 0: none
  { jump: true, down: false },   // 1: jump held
  { jump: false, down: true },   // 2: down held (duck / fast-drop)
];
const NONE = ACTIONS[0], START = ACTIONS[1];

const TYPE_INDEX = Object.fromEntries(OBSTACLE_TYPES.map((t, i) => [t.name, i]));
const PTERO_YS = OBSTACLE_TYPES[TYPE_INDEX.ptero].ys;
const DINO_FRONT = standBox(STAND_TOP).x + standBox(STAND_TOP).w;

const near = [];               // scratch for observe(); emptied on every call

// Fill `out` (length OBS_DIM) from the game's visible state. Returns out.
function observe(game, out) {
  const d = game.dino;
  out[0] = (STAND_TOP - d.y) / 100;
  out[1] = d.vy / 10;
  out[2] = d.jumping ? 1 : 0;
  out[3] = d.ducking ? 1 : 0;
  out[4] = (game.speed - CFG.speed) / (CFG.maxSpeed - CFG.speed);

  // Pick the OBS_K nearest visible, not-yet-passed obstacles by x. Only x < W
  // is ever considered, before any other field is read.
  near.length = 0;
  for (const o of game.obstacles) {
    if (!(o.x < W)) continue;
    if (o.x + o.w <= DINO_X) continue;
    let i = near.length;
    near.push(o);
    while (i > 0 && near[i - 1].x > o.x) { near[i] = near[i - 1]; i--; }
    near[i] = o;
  }

  for (let k = 0; k < OBS_K; k++) {
    const b = DINO_FEATURES + k * PER_OBSTACLE;
    const o = near[k];
    for (let j = 0; j < PER_OBSTACLE; j++) out[b + j] = 0;
    if (!o) { out[b + 1] = 1; continue; }      // absent: present = 0, "one screen away"
    const box = obstacleBox(o);
    const type = TYPE_INDEX[o.t.name];
    out[b] = 1;
    out[b + 1] = (box.x - DINO_FRONT) / W;
    out[b + 2] = box.y / H;
    out[b + 3] = box.w / 100;
    out[b + 4] = box.h / 50;
    out[b + 5 + type] = 1;
    out[b + 8] = o.size / 3;
    if (type === TYPE_INDEX.ptero) out[b + 9 + PTERO_YS.indexOf(o.y)] = 1;
    out[b + 12] = (game.speed + o.speedOffset) / CFG.maxSpeed;
  }
  return out;
}

// ---------------------------------------------------------------------------
// MLP: tanh hidden layers, linear output, argmax action.
// Flat parameter layout per layer: weights (out x in, row-major) then biases.
// ---------------------------------------------------------------------------
function numParams(layers) {
  let n = 0;
  for (let l = 1; l < layers.length; l++) n += layers[l] * layers[l - 1] + layers[l];
  return n;
}

// Scaled-normal init (std 1/sqrt(fan_in)), zero biases. `gauss` is () => N(0,1).
function initParams(layers, gauss) {
  const p = new Float64Array(numParams(layers));
  let k = 0;
  for (let l = 1; l < layers.length; l++) {
    const s = 1 / Math.sqrt(layers[l - 1]);
    for (let i = 0; i < layers[l] * layers[l - 1]; i++) p[k++] = gauss() * s;
    k += layers[l];
  }
  return p;
}

// Returns a function x -> action index, with preallocated buffers.
// Rows are computed four at a time (each input read once per block), ~1.5-1.7x
// faster. Every unit still sums bias + w0*x0 + w1*x1 + ... in index order, so
// results are bit-identical to the one-row-at-a-time loop (see test/net.test.js).
function createNet(layers, params) {
  if (params.length !== numParams(layers)) throw new Error(`expected ${numParams(layers)} params, got ${params.length}`);
  const bufs = layers.map(n => new Float64Array(n));
  return function decide(x) {
    let input = x;
    let k = 0;
    for (let l = 1; l < layers.length; l++) {
      const nIn = layers[l - 1], nOut = layers[l], outB = bufs[l];
      const last = l === layers.length - 1;
      const bBase = k + nOut * nIn;
      let i = 0;
      for (; i + 4 <= nOut; i += 4) {
        let s0 = params[bBase + i], s1 = params[bBase + i + 1], s2 = params[bBase + i + 2], s3 = params[bBase + i + 3];
        const r0 = k + i * nIn, r1 = r0 + nIn, r2 = r1 + nIn, r3 = r2 + nIn;
        for (let j = 0; j < nIn; j++) {
          const v = input[j];
          s0 += params[r0 + j] * v;
          s1 += params[r1 + j] * v;
          s2 += params[r2 + j] * v;
          s3 += params[r3 + j] * v;
        }
        if (last) { outB[i] = s0; outB[i + 1] = s1; outB[i + 2] = s2; outB[i + 3] = s3; }
        else { outB[i] = Math.tanh(s0); outB[i + 1] = Math.tanh(s1); outB[i + 2] = Math.tanh(s2); outB[i + 3] = Math.tanh(s3); }
      }
      for (; i < nOut; i++) {
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

// An agent drives one game through per-frame key states. `model` is the exported
// form ({ format, layers, params, actionRepeat }) or anything with those fields.
//   idle    -> jump (starts a run, exactly like a human's first press)
//   over    -> no keys (restarting is the caller's choice)
//   running -> policy decision, re-evaluated every actionRepeat frames
function createAgent(model) {
  if (model.format && model.format !== FORMAT) throw new Error(`unsupported model format ${model.format}`);
  if (model.layers[0] !== OBS_DIM) throw new Error(`model expects ${model.layers[0]} inputs, policy provides ${OBS_DIM}`);
  const decide = createNet(model.layers, model.params);
  const repeat = Math.max(1, model.actionRepeat | 0);
  const x = new Float64Array(OBS_DIM);
  let wait = 0, current = NONE;
  return {
    act(game) {
      if (game.state === 'idle') { wait = 0; current = NONE; return START; }
      if (game.state !== 'running') { wait = 0; current = NONE; return NONE; }
      if (wait <= 0) {
        current = ACTIONS[decide(observe(game, x))];
        wait = repeat;
      }
      wait--;
      return current;
    },
  };
}

const DinoPolicy = { FORMAT, OBS_K, OBS_DIM, ACTIONS, observe, numParams, initParams, createNet, createAgent };

root.DinoPolicy = DinoPolicy;
if (typeof module !== 'undefined' && module.exports) module.exports = DinoPolicy;
})(typeof globalThis !== 'undefined' ? globalThis : this);
