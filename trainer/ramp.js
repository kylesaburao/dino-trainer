'use strict';
// Natural-ramp training starts (--ramp-frac): the solver (test/solver.js, the same
// teacher as bc.js) plays the real game from 0 until the score reaches the episode's
// ramp level, then a fresh student agent takes over. Below ~2000 points a
// createGame({ startScore }) start is not faithful (it starts at the speed cap while
// natural speed is still ramping), but while the dino survives, the world (speed,
// distance, obstacles, RNG) depends only on the seed and the frame, never on the
// controller's actions; so the handed-over world is exactly the natural one, and only
// the dino's own state comes from the teacher. The handover waits for the first frame
// at or after the ramp level with the dino grounded, not ducking and no key down, so
// the student never inherits a jump or a held key. Loaded only when a job has ramp
// episodes (training only; validation, held-out, eval and export never use it).
const { createGame, createRng, CFG, V2 } = require('../core.js');
const { createAgent } = require('../policy.js');
const { createSolver } = require('../test/solver.js');

// Frames allowed after reaching the ramp level to find a grounded, keys-up frame
// (a solver jump lasts well under 100), and the most points that can add (at v2's top speed).
const HANDOVER_SLACK = 600;
const MAX_OVERSHOOT = Math.ceil(HANDOVER_SLACK * V2.maxSpeed * CFG.scoreCoef);

// Play the solver from a fresh game until the handover frame. Returns the running game
// and the frames the teacher played (= game.frameIndex). Throws if the teacher crashes
// or never reaches a clean handover state.
function rampHandover(seed, version, rampScore) {
  const game = createGame({ rng: createRng(seed), version });
  const teacher = createSolver();
  let frames = 0, reachedAt = -1;
  for (;;) {
    if (game.state === 'running' && game.score() >= rampScore) {
      if (reachedAt < 0) reachedAt = frames;
      const d = game.dino;
      if (!d.jumping && !d.ducking && !game.keys.jump && !game.keys.down) return { game, teacherFrames: frames };
      if (frames - reachedAt > HANDOVER_SLACK) throw new Error(`ramp: no clean handover within ${HANDOVER_SLACK} frames of ${rampScore} points (seed ${seed})`);
    }
    game.step(teacher(game));
    frames++;
    if (game.state === 'over') throw new Error(`ramp: the teacher crashed at ${game.score()} points (seed ${seed})`);
  }
}

// One natural-ramp episode: teacher to the handover, then the student until it crashes
// or passes cap = min(handover score + window, capMax). Like runEpisode, plus the
// handover score (start, an integer score like every curriculum start), the cap and
// the teacher's frames; frames counts the student's frames only.
function runRampEpisode(model, seed, rampScore, window, capMax, makeAgent = createAgent) {
  const { game, teacherFrames } = rampHandover(seed, model.gameVersion || 'v1', rampScore);
  const start = game.score();
  const cap = Math.min(start + window, capMax);
  if (!(cap > start)) throw new Error(`ramp: handover at ${start} points leaves no window below ${capMax} (seed ${seed})`);
  const agent = makeAgent(model);
  const capDistance = cap / CFG.scoreCoef;
  let frames = 0;
  do {
    game.step(agent.act(game));
    frames++;
  } while (game.state === 'running' && game.distance < capDistance);
  return { score: Math.min(game.score(), cap), distance: game.distance, frames, teacherFrames, start, cap };
}

module.exports = { rampHandover, runRampEpisode, HANDOVER_SLACK, MAX_OVERSHOOT };
