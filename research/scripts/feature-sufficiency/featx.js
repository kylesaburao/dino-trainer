'use strict';
// Shared pieces for Probe B: extended observations (allowed information only),
// the exact solver's decision as a per-state expert label, and variant column maps.
const R = '/Users/kyle/Desktop/dino-game/';
const Core = require(R + 'core.js');
const P = require(R + 'policy.js');
const { W, STAND_TOP, obstacleBox, standBox, _internal: { requirement } } = Core;
const FRONT = standBox(STAND_TOP).x + standBox(STAND_TOP).w;

// Row layout: 44 base features (policy.observe) + EXTRA below.
//   44,45,46 tau_k   frames until obstacle k's hitbox reaches the dino front, /30 (absent: 80/30)
//   47       gap01   frames between obstacle 0's back edge and obstacle 1's front, /30 (absent: 80/30)
//   48       keyJump the agent's own jump key is down (game.keys.jump)
//   49       tLand   frames until a free fall lands (0 when grounded), /30
//   50       speed22 (speed - 6) / 16          (rescaled out[4])
//   51,52,53 v22_k   obstacle k on-screen velocity / 22 (rescaled out[b+12])
// Every extra is a function of the 44 base features (plus the agent's own key state),
// so it adds no new information except keyJump.
const EXTRA = 10, ROW = P.OBS_DIM + EXTRA;
const ABSENT_TAU = 80 / 30;

function extend(game, x) {               // x: Float32Array/Float64Array of length ROW, base already filled
  for (let k = 0; k < 3; k++) {
    const b = 5 + 13 * k;
    x[44 + k] = x[b] ? Math.min(80, (x[b + 1] * W) / (x[b + 12] * 13)) / 30 : ABSENT_TAU;
  }
  if (x[5] && x[18]) {
    const gapPx = x[19] * W - (x[6] * W + x[8] * 100);
    x[47] = Math.min(80, gapPx / (x[30] * 13)) / 30;
  } else x[47] = ABSENT_TAU;
  x[48] = game.keys.jump ? 1 : 0;
  let y = STAND_TOP - x[0] * 100, vy = x[1] * 10, t = 0;
  if (x[2]) while (y < STAND_TOP && t < 80) { y += vy; vy += 0.6; t++; }
  x[49] = t / 30;
  x[50] = (x[4] * 7 + 6 - 6) / 16;
  for (let k = 0; k < 3; k++) x[51 + k] = x[5 + 13 * k + 12] * 13 / 22;
  return x;
}

function observeRow(game, x) {
  P.observe(game, x);
  return extend(game, x);
}

// Feature-set variants: list of row columns fed to the net.
const BASE = Array.from({ length: 44 }, (_, i) => i);
const rescaled = BASE.map(i => (i === 4 ? 50 : i === 17 ? 51 : i === 30 ? 52 : i === 43 ? 53 : i));
const VARIANTS = {
  base: BASE,
  rescale: rescaled,
  tau: [...BASE, 44, 45],
  tau_gap: [...BASE, 44, 45, 47],
  key: [...BASE, 48],
  tau_gap_key: [...BASE, 44, 45, 47, 48],
  all: [...rescaled, 44, 45, 46, 47, 48, 49],
};

// The exact solver's choice in this state (test/solver.js logic). Grounded states
// are stateless in the solver; airborne states need the frames since takeoff `k`.
function expert(game, k) {
  if (game.state !== 'running') return 0;
  const d = game.dino;
  if (d.jumping) {
    let best = null;
    for (const o of game.obstacles) {
      const r = requirement(o, game.speed);
      if (!r || r.grounded || r.to <= 0) continue;
      if (!best || r.from < best.from) best = r;
    }
    if (!best) return 0;
    if (best.release === Infinity) return d.vy > 0 ? 0 : 1;
    return k < best.release - 1 ? 1 : 0;
  }
  if (game.keys.jump) return 0;
  let next = null;
  for (const o of game.obstacles) {
    const v = game.speed + o.speedOffset;
    const tau = (obstacleBox(o).x - FRONT) / v;
    if (tau > 80) continue;
    const r = requirement(o, game.speed);
    if (!r || r.grounded || r.to <= 0) continue;
    if (!next || r.from < next.from) next = { ...r, o };
  }
  return next && next.from <= 0.5 ? 1 : 0;
}

// The obstacle the solver would take off for now (for the visibility check).
function pressTarget(game) {
  let next = null;
  for (const o of game.obstacles) {
    const tau = (obstacleBox(o).x - FRONT) / (game.speed + o.speedOffset);
    if (tau > 80) continue;
    const r = requirement(o, game.speed);
    if (!r || r.grounded || r.to <= 0) continue;
    if (!next || r.from < next.from) next = { from: r.from, o };
  }
  return next && next.o;
}

module.exports = { ROW, EXTRA, observeRow, extend, VARIANTS, expert, pressTarget };
