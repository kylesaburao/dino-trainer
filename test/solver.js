'use strict';
// Exact fairness-model controller, for the tests and as the behaviour-cloning
// teacher (trainer/bc.js, bc-gen). It reads the planner, which the policy must
// never do; the clone's inputs come only from policy.observe. It executes the canonical strategy the fairness
// model guarantees exists: for each obstacle that must be jumped, take off on
// the whole frame nearest the centre of its takeoff window, using that
// obstacle's frozen requirement (release frame), and stay grounded otherwise.
// If the fairness model is right, this never crashes.
const { _internal: { requirement, jumpArc }, standBox, obstacleBox, STAND_TOP, GROUND, DUCK_H, W, MAX_PTERO_SPEED_OFFSET } = require('../core.js');

const NONE = { jump: false, down: false }, JUMP = { jump: true, down: false };
const FRONT = standBox(STAND_TOP).x + standBox(STAND_TOP).w;

// Lookahead prune: an obstacle with tau > horizon cannot be due (takeoff from <= 0.5).
// Every takeoff lies within the longest airtime before tau (from >= tau - b, b <= airtime;
// rules-v3 late anchors are later still), so any horizon >= that airtime is exact. The
// longest airtime is ~33 frames at every speed (jumpVelocity grows with speed, but
// maxJumpY / dropVel cap the arc), so the horizon is 80 frames on rules v1, v2 and v3.
const horizon = (speed) => Math.max(80, jumpArc(speed, Infinity).length + 2);

function createSolver() {
  let hold = 0;                  // frames the jump key stays down after the press frame
  return function act(game) {
    if (game.state === 'idle') return JUMP;
    if (game.state !== 'running') return NONE;
    const d = game.dino;
    if (d.jumping) {
      // A full-height jump (release = Infinity) keeps the key down until the dino
      // falls; releasing while falling cannot change the arc (vy > 0 is never cut),
      // and leaves the key up so the next jump can be pressed right after landing.
      if (hold === Infinity) return d.vy > 0 ? NONE : JUMP;
      if (hold > 0) { hold--; return JUMP; }
      return NONE;
    }
    if (game.keys.jump) return NONE;
    let next = null, nextFrom = Infinity;
    const far = horizon(game.speed);
    for (const o of game.obstacles) {
      const v = game.speed + o.speedOffset;
      const tau = (obstacleBox(o).x - FRONT) / v;
      if (tau > far) continue;                        // takeoff is far away (from <= tau)
      const r = requirement(o, game.speed);
      if (!r || r.grounded || r.to <= 0) continue;
      if (r.from < nextFrom) { next = r; nextFrom = r.from; }
    }
    if (next && next.from <= 0.5) {
      // Key down on jump steps 1 .. release-1, released before step `release`.
      hold = next.release === Infinity ? Infinity : next.release - 2;
      return JUMP;
    }
    return NONE;
  };
}

// teacher-duck (research/prereg-teacher-duck.md): createSolver's plan, except that an
// obstacle the fairness model says must be jumped, but whose box clears the ducking
// box (mid pterodactyls), is passed by staying grounded and holding DOWN through its
// overlap with the standing box. The choice is made on the frame the jump would take
// off. It ducks only if (1) no other planned takeoff comes before the duck ends,
// (2) no obstacle that hits the ducking box (wider than the standing box) overlaps
// it while ducking, and (3) the duck ends before any obstacle not yet spawned could
// reach it. Otherwise it falls back to the planned jump (counted). DOWN is never
// pressed while airborne (that would fast-drop). Releasing DOWN and pressing jump
// on the same frame works: applyKeys releases before it presses.
//
// act.stats (per solver, so one solver per game): midEncountered (jump-required
// obstacles that clear the ducking box, counted at their decision frame) = midDucked +
// fallback; fallbackWhy {takeoff, obstacle, newcomer}; downFrames (frames returning
// DOWN); conflicts (a takeoff came due during a duck; expected 0).
const DOWN = { jump: false, down: true };
const DUCK_MARGIN = 1;          // frames of extra duck before and after the predicted overlap
const MAX_PTERO_OFFSET = MAX_PTERO_SPEED_OFFSET;   // the largest pterodactyl speedOffset (1.6 on rules v2 and v3); bounds newcomers

// The ducking hitbox, as core.js dinoBox() builds it (core.js does not export it).
const duckBox = (dino) => ({ x: dino.x + 4, y: GROUND - DUCK_H + 4, w: 50, h: 18 });

function createDuckSolver() {
  let hold = 0;
  const plan = new WeakMap();     // obstacle -> 'duck' | 'jump'
  const stats = { midEncountered: 0, midDucked: 0, fallback: 0, fallbackWhy: { takeoff: 0, obstacle: 0, newcomer: 0 }, downFrames: 0, conflicts: 0 };
  const stand = standBox(STAND_TOP);

  // Steps k (k = 1 is the step about to run) during which the standing box would
  // overlap o horizontally, widened by DUCK_MARGIN.
  function duckSteps(o, speed) {
    const b = obstacleBox(o), v = speed + o.speedOffset;
    const tau = (b.x - (stand.x + stand.w)) / v, end = (b.x + b.w - stand.x) / v;
    return { d0: Math.max(1, Math.floor(tau) + 1 - DUCK_MARGIN), d1: Math.ceil(end) - 1 + DUCK_MARGIN };
  }
  // Why P cannot be ducked (null if it can).
  function blocker(game, P, w) {
    const duck = duckBox(game.dino);
    for (const o of game.obstacles) {
      if (o === P || plan.get(o) === 'duck') continue;
      const r = requirement(o, game.speed);
      // (1) another takeoff (a planned or possible jump) before the duck ends.
      if (r && !r.grounded && r.to > 0 && Math.ceil(r.from - 0.5) + 1 <= w.d1) return 'takeoff';
      // (2) an obstacle that hits the ducking box overlaps it while ducking.
      const b = obstacleBox(o), v = game.speed + o.speedOffset;
      if (b.y < duck.y + duck.h && b.y + b.h > duck.y) {
        const from = (b.x - (duck.x + duck.w)) / v, to = (b.x + b.w - duck.x) / v;
        if (from < w.d1 && to > w.d0 - 1) return 'obstacle';
      }
    }
    // (3) an obstacle spawned from the next step on starts at x >= W.
    if (w.d1 >= 1 + (W - (duck.x + duck.w)) / (game.speed + MAX_PTERO_OFFSET + 0.1)) return 'newcomer';
    return null;
  }

  function act(game) {
    if (game.state === 'idle') return JUMP;
    if (game.state !== 'running') return NONE;
    const d = game.dino;
    if (d.jumping) {
      if (hold === Infinity) return d.vy > 0 ? NONE : JUMP;
      if (hold > 0) { hold--; return JUMP; }
      return NONE;
    }
    let ducking = false;
    for (const o of game.obstacles) {
      if (plan.get(o) !== 'duck') continue;
      const w = duckSteps(o, game.speed);
      if (w.d0 <= 1 && w.d1 >= 1) ducking = true;
    }
    for (;;) {
      if (!ducking && game.keys.jump) return NONE;
      let next = null, nextFrom = Infinity, nextO = null;
      const far = horizon(game.speed);
      for (const o of game.obstacles) {
        if (plan.get(o) === 'duck') continue;
        const v = game.speed + o.speedOffset;
        const tau = (obstacleBox(o).x - FRONT) / v;
        if (tau > far) continue;
        const r = requirement(o, game.speed);
        if (!r || r.grounded || r.to <= 0) continue;
        if (r.from < nextFrom) { next = r; nextFrom = r.from; nextO = o; }
      }
      if (next && next.from <= 0.5) {
        const b = obstacleBox(nextO);
        if (!plan.has(nextO) && b.y + b.h <= duckBox(d).y) {
          stats.midEncountered++;
          const why = blocker(game, nextO, duckSteps(nextO, game.speed));
          if (!why) { plan.set(nextO, 'duck'); stats.midDucked++; continue; }   // re-plan without it
          plan.set(nextO, 'jump'); stats.fallback++; stats.fallbackWhy[why]++;
        }
        if (ducking) { stats.conflicts++; break; }
        hold = next.release === Infinity ? Infinity : next.release - 2;
        return JUMP;
      }
      break;
    }
    if (ducking) { stats.downFrames++; return DOWN; }
    return NONE;
  }
  act.stats = stats;
  return act;
}

module.exports = { createSolver, createDuckSolver, duckBox };
