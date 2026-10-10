'use strict';
// Headless episodes: the real core driven by the shared policy agent, one key
// state per frame through the real input functions (game.step -> applyKeys).
const { createGame, createRng, CFG } = require('../core.js');
const { createAgent } = require('../policy.js');

// Seed ranges are disjoint by construction:
//   training   [0, 2^30)      drawn from the trainer's RNG every generation
//   validation [2^30, 2^31)   fixed seeds, used to pick the best model
//   held-out   [2^31, 2^32)   fixed seeds, only for `eval` and the stop check
// validationSeeds(n) / evalSeeds(n) extend the 25-seed sets with the same formula,
// so their first 25 are VALIDATION_SEEDS / EVAL_SEEDS.
const hashSeed = (i, salt) => (Math.imul((i + 1) ^ salt, 0x9E3779B1) ^ 0x85EBCA6B) >>> 0;
const validationSeeds = (n) => Array.from({ length: n }, (_, i) => 0x40000000 + (hashSeed(i, 0x51) & 0x3FFFFFFF));
const evalSeeds = (n) => Array.from({ length: n }, (_, i) => (0x80000000 | hashSeed(i, 0xE7)) >>> 0);
const VALIDATION_SEEDS = validationSeeds(25);
const EVAL_SEEDS = evalSeeds(25);
const trainSeed = (rng) => Math.floor(rng() * 0x40000000);

// Windowed random-start curriculum (training only; off unless window > 0).
// Start levels are a pure function of the generation's episode seeds, so they
// take no draws from the trainer RNG and every candidate gets the same
// (seed, startScore) pairs. A share startZeroFrac of the episodes start at 0
// (stratified: the first floor(frac*n) episodes, plus one more with probability
// equal to the fractional part); the rest start at a late level drawn by c.startDist:
//   'log'    log-uniform in [startMin, target - window], so every window ends at or
//            before target (the original rule);
//   'points' uniform in points on [startMin, target); a window that would pass the
//            target is clipped there (see episodeCaps).
// Levels must be 0 or >= ~2000: below that, natural speed is still ramping and
// a late start would not match natural play.
const START_SALT = 0x5EED5A17;
const uniform = (seed, k) => createRng((hashSeed(seed, START_SALT) + Math.imul(k, 0x632BE5AB)) >>> 0)();
function startScores(seeds, c) {
  const want = c.startZeroFrac * seeds.length;
  const zeros = Math.floor(want) + (uniform(seeds[0], 0) < want - Math.floor(want) ? 1 : 0);
  if (c.startDist === 'points') {
    return seeds.map((s, j) => (j < zeros ? 0 : Math.min(c.target - 1, Math.round(c.startMin + uniform(s, 1) * (c.target - c.startMin)))));
  }
  const hi = c.target - c.window;
  return seeds.map((s, j) => (j < zeros ? 0 : Math.round(c.startMin * Math.pow(hi / c.startMin, uniform(s, 1)))));
}

// Absolute end score of each curriculum episode: start-0 episodes end at
// startZeroCap; late episodes at start + window, clipped to target.
const episodeCaps = (starts, c) => starts.map(s => (s === 0 ? c.startZeroCap : Math.min(s + c.window, c.target)));

// Per-episode curriculum fitness: progress through the episode's span
// (cap - start: the window, or a clipped final window's own length), in [0, 1].
// Uses fractional distance (like the full-game fitness) for a finer signal.
const windowFitness = (distance, startScore, span) =>
  Math.min(1, Math.max(0, distance * CFG.scoreCoef - startScore) / span);

// Play one episode from a fresh game until crash or `capScore` points, on the
// game version the model is for (model.gameVersion; 'v1' if absent).
// The agent's first action presses jump from idle (the run's start).
// startScore (training curriculum only) starts the run at that score; capScore
// is absolute, so a window episode passes startScore + window.
// makeAgent (default policy.js createAgent) builds the agent from the model; the
// trainer's --wasm path passes a wasm-net.js factory with the same decisions.
function runEpisode(model, seed, capScore, startScore, makeAgent = createAgent) {
  const gameOpts = { rng: createRng(seed), version: model.gameVersion || 'v1' };
  if (startScore) gameOpts.startScore = startScore;
  const game = createGame(gameOpts);
  const agent = makeAgent(model);
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

module.exports = { runEpisode, median, trainSeed, startScores, episodeCaps, windowFitness, validationSeeds, evalSeeds, VALIDATION_SEEDS, EVAL_SEEDS };
