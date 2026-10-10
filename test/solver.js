'use strict';
// Exact fairness-model controller, for the tests and as the behaviour-cloning
// teacher (trainer/bc.js, bc-gen). It reads the planner, which the policy must
// never do; the clone's inputs come only from policy.observe. It executes the canonical strategy the fairness
// model guarantees exists: for each obstacle that must be jumped, take off on
// the whole frame nearest the centre of its takeoff window, using that
// obstacle's frozen requirement (release frame), and stay grounded otherwise.
// If the fairness model is right, this never crashes.
const { _internal: { requirement }, standBox, obstacleBox, STAND_TOP } = require('../core.js');

const NONE = { jump: false, down: false }, JUMP = { jump: true, down: false };
const FRONT = standBox(STAND_TOP).x + standBox(STAND_TOP).w;

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
    for (const o of game.obstacles) {
      const v = game.speed + o.speedOffset;
      const tau = (obstacleBox(o).x - FRONT) / v;
      if (tau > 80) continue;                         // takeoff is far away (from <= tau)
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

module.exports = { createSolver };
