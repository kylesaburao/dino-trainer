'use strict';
// Observation rules: only what is on screen (x < 600) and never planner internals.
const test = require('node:test');
const assert = require('node:assert');
const { createGame, createRng, OBSTACLE_TYPES, W } = require('../core.js');
const P = require('../policy.js');
const { runEpisode } = require('../trainer/env.js');
const { gaussian } = require('../trainer/es.js');

// Play a seeded game with simple scripted inputs and call visit(game) every frame.
function play(seed, frames, visit) {
  const game = createGame({ rng: createRng(seed) });
  const keyRng = createRng(seed + 1);
  let code = 0;
  for (let f = 0; f < frames; f++) {
    if (f % 7 === 0) code = keyRng() < 0.3 ? 1 : keyRng() < 0.1 ? 2 : 0;
    game.step({ jump: code === 1 || f === 0, down: code === 2 });
    if (game.state !== 'running') break;
    visit(game, f);
  }
}

const obs = (game) => Array.from(P.observe(game, new Float64Array(P.OBS_DIM)));

test('observations ignore obstacles at x >= 600 and planner internals', () => {
  let checked = 0, sawOffscreen = 0;
  for (let seed = 1; seed <= 40; seed++) {
    play(seed, 3000, (game) => {
      const before = obs(game);
      if (game.obstacles.some(o => o.x >= W)) sawOffscreen++;
      // Tamper with everything the policy must not see.
      const saved = game.obstacles.map(o => ({ o, x: o.x, gap: o.gap, size: o.size, y: o.y }));
      for (const o of game.obstacles) {
        o.gap = -12345;
        if (o.x >= W) { o.x += 500; o.size = 3; o.y = 58; }
      }
      const fake = { t: OBSTACLE_TYPES[2], size: 1, frame: 0, timer: 0, x: W, y: 58, w: 46, h: 20, speedOffset: 0.8, gap: 0 };
      game.obstacles.push(fake, { ...fake, x: W + 100, t: OBSTACLE_TYPES[1] });
      const after = obs(game);
      game.obstacles.length -= 2;
      for (const s of saved) Object.assign(s.o, { x: s.x, gap: s.gap, size: s.size, y: s.y });
      assert.deepStrictEqual(after, before);
      checked++;
    });
  }
  assert.ok(checked > 1000, 'enough frames checked');
  assert.ok(sawOffscreen > 0, 'placeFairly-parked obstacles (x >= 600) occurred and were ignored');
});

test('an obstacle at x < 600 is observed', () => {
  const game = createGame({ rng: createRng(5) });
  game.step({ jump: true });
  const empty = obs(game);
  game.obstacles.push({ t: OBSTACLE_TYPES[0], size: 1, frame: 0, timer: 0, x: W - 1, y: 106, w: 16, h: 34, speedOffset: 0 });
  assert.notDeepStrictEqual(obs(game), empty);
  assert.strictEqual(obs(game)[5], 1);
});

test('features are normalised (|x| <= ~2) over real play', () => {
  let maxAbs = 0;
  for (let seed = 1; seed <= 20; seed++) play(seed, 3000, (g) => { for (const v of obs(g)) maxAbs = Math.max(maxAbs, Math.abs(v)); });
  assert.ok(maxAbs <= 2, `max |feature| ${maxAbs}`);
});

test('agent episodes are deterministic; an idle game is started with a jump press', () => {
  const layers = [P.OBS_DIM, 8, 3];
  const model = { layers, params: P.initParams(layers, gaussian(createRng(3))), actionRepeat: 1 };
  const a = runEpisode(model, 42, 10000), b = runEpisode(model, 42, 10000);
  assert.deepStrictEqual(a, b);
  assert.ok(a.frames > 1);
  const game = createGame({ rng: createRng(1) });
  const agent = P.createAgent(model);
  assert.deepStrictEqual(agent.act(game), P.ACTIONS[1], 'idle -> press jump to start');
});

test('model shape is validated', () => {
  assert.throws(() => P.createAgent({ layers: [3, 3], params: new Float64Array(12) }), /inputs/);
  assert.throws(() => P.createNet([P.OBS_DIM, 3], new Float64Array(5)), /params/);
});
