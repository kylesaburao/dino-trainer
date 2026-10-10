'use strict';
// Headless episodes: the real core driven by the shared policy agent, one key
// state per frame through the real input functions (game.step -> applyKeys).
const { createGame, createRng, CFG } = require('../core.js');
const { createAgent } = require('../policy.js');

// Seed ranges are disjoint by construction:
//   training   [0, 2^30)      drawn from the trainer's RNG every generation
//   validation [2^30, 2^31)   25 fixed seeds, used to pick the best model
//   held-out   [2^31, 2^32)   25 fixed seeds, only for `eval` and the stop check
const hashSeed = (i, salt) => (Math.imul((i + 1) ^ salt, 0x9E3779B1) ^ 0x85EBCA6B) >>> 0;
const VALIDATION_SEEDS = Array.from({ length: 25 }, (_, i) => 0x40000000 + (hashSeed(i, 0x51) & 0x3FFFFFFF));
const EVAL_SEEDS = Array.from({ length: 25 }, (_, i) => (0x80000000 | hashSeed(i, 0xE7)) >>> 0);
const trainSeed = (rng) => Math.floor(rng() * 0x40000000);

// Play one episode from a fresh game until crash or `capScore` points.
// The agent's first action presses jump from idle (the run's start).
function runEpisode(model, seed, capScore) {
  const game = createGame({ rng: createRng(seed) });
  const agent = createAgent(model);
  const capDistance = capScore / CFG.scoreCoef;
  let frames = 0;
  do {
    game.step(agent.act(game));
    frames++;
  } while (game.state === 'running' && game.distance < capDistance);
  return { score: Math.min(game.score(), capScore), distance: game.distance, frames };
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

module.exports = { runEpisode, median, trainSeed, VALIDATION_SEEDS, EVAL_SEEDS };
