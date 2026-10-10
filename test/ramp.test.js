'use strict';
// Natural-ramp training starts (trainer/ramp.js, --ramp-frac): the solver plays from 0
// and hands over to the student. The gate is faithfulness: the world the student
// receives is exactly the world of pure solver play at that frame.
const test = require('node:test');
const assert = require('node:assert');
const { createGame, createRng, CFG } = require('../core.js');
const { createAgent, initParams, OBS_DIM } = require('../policy.js');
const { createSolver } = require('./solver.js');
const { gaussian } = require('../trainer/es.js');
const env = require('../trainer/env.js');
const rules = require('../trainer/rules.js');
const { rampHandover, runRampEpisode, HANDOVER_SLACK, MAX_OVERSHOOT } = require('../trainer/ramp.js');

const SEEDS = [1, 7, 42, 1234, 99991, 2 ** 29 + 5];
const LEVELS = [400, 550, 700, 800];

// Everything about the world a frame can carry (obstacles hold their type object; JSON keeps its fields).
const world = (g) => JSON.stringify({
  state: g.state, frameIndex: g.frameIndex, distance: g.distance, speed: g.speed, score: g.score(), level: g.level,
  obstacles: g.obstacles, clouds: g.clouds, groundX: g.groundX, nightAlpha: g.nightAlpha, moonX: g.moonX, stars: g.stars,
});
const dinoState = (g) => JSON.stringify({ dino: g.dino, keys: g.keys });

// Pure solver play of a seed for exactly n frames.
function solverPlay(seed, version, n) {
  const g = createGame({ rng: createRng(seed), version });
  const act = createSolver();
  for (let i = 0; i < n; i++) g.step(act(g));
  return g;
}

test('faithfulness gate: the handed-over world and dino are exactly pure solver play at that frame', () => {
  for (const version of ['v2', 'v1']) {
    for (const seed of SEEDS) {
      for (const level of LEVELS) {
        const { game, teacherFrames } = rampHandover(seed, version, level);
        const ref = solverPlay(seed, version, teacherFrames);
        assert.strictEqual(game.frameIndex, teacherFrames);
        assert.strictEqual(world(game), world(ref), `world, ${version} seed ${seed} level ${level}`);
        assert.strictEqual(dinoState(game), dinoState(ref), `dino, ${version} seed ${seed} level ${level}`);
        // Clean handover: running, at or past the level, grounded, not ducking, no key down.
        assert.strictEqual(game.state, 'running');
        assert.ok(game.score() >= level);
        assert.ok(!game.dino.jumping && !game.dino.ducking && !game.keys.jump && !game.keys.down);
      }
    }
  }
});

test('faithfulness gate end to end: a solver student after the handover replays pure solver play exactly', () => {
  // A fresh solver is equivalent at a grounded, keys-up handover (its hold state is
  // only read mid-jump), so the whole episode must match pure solver play, including
  // every later spawn (which would expose any drift in the hidden RNG or committed state).
  const solverStudent = () => ({ act: createSolver() });
  for (const seed of SEEDS) {
    for (const level of [400, 800]) {
      const r = runRampEpisode({ gameVersion: 'v2' }, seed, level, 1500, 1e9, solverStudent);
      // Pure solver play from 0 until it passes the same cap, counted independently.
      const ref = createGame({ rng: createRng(seed), version: 'v2' });
      const act = createSolver();
      let n = 0;
      do { ref.step(act(ref)); n++; } while (ref.state !== 'over' && (ref.state === 'idle' || ref.distance < r.cap / CFG.scoreCoef));
      assert.strictEqual(ref.state, 'running', 'pure solver play survives to the cap');
      assert.strictEqual(r.teacherFrames + r.frames, n, `frames, seed ${seed} level ${level}`);
      assert.strictEqual(r.distance, ref.distance, `distance, seed ${seed} level ${level}`);
      assert.strictEqual(r.score, r.cap, 'the solver student reaches the cap');
    }
  }
});

test('handover comes soon after the level, never far past it', () => {
  let worst = 0;
  for (let seed = 1; seed <= 40; seed++) {
    for (const level of LEVELS) {
      const { game, teacherFrames } = rampHandover(seed, 'v2', level);
      // First frame at or past the level (the speed ramp is the same for every controller).
      const first = solverPlay(seed, 'v2', 0);
      const act = createSolver();
      let n = 0;
      while (!(first.state === 'running' && first.score() >= level)) { first.step(act(first)); n++; }
      worst = Math.max(worst, teacherFrames - n);
      assert.ok(teacherFrames - n <= HANDOVER_SLACK);
      assert.ok(game.score() - level <= MAX_OVERSHOOT);
    }
  }
  assert.ok(worst < 100, `worst handover delay ${worst} frames`);
});

test('the student decides first on the handover frame, then once per frame', () => {
  const layers = [OBS_DIM, 16, 3];
  const model = { layers, params: initParams(layers, gaussian(createRng(3))), actionRepeat: 1, gameVersion: 'v2' };
  for (const seed of SEEDS) {
    const calls = [];
    const spy = (m) => {
      const inner = createAgent(m);
      return {
        act(g) {
          calls.push({ frame: g.frameIndex, state: g.state, score: g.score(), clean: !g.dino.jumping && !g.dino.ducking && !g.keys.jump && !g.keys.down });
          return inner.act(g);
        },
      };
    };
    const r = runRampEpisode(model, seed, 600, 2000, 1e9, spy);
    assert.strictEqual(calls[0].frame, r.teacherFrames, 'first decision on the handover frame');
    assert.strictEqual(calls[0].state, 'running');
    assert.ok(calls[0].clean && calls[0].score >= 600 && calls[0].score === r.start);
    assert.strictEqual(calls.length, r.frames);
    calls.forEach((c, i) => assert.strictEqual(c.frame, r.teacherFrames + i));
    // The spy changes nothing.
    assert.deepStrictEqual(runRampEpisode(model, seed, 600, 2000, 1e9), r);
  }
});

test('ramp episodes work for any --hidden and for the wasm agent, with the same results', () => {
  const { wasmAgents } = require('../trainer/wasm-net.js');
  for (const hidden of [[8], [16, 16, 16], [32, 32]]) {
    const layers = [OBS_DIM, ...hidden, 3];
    const model = { layers, params: initParams(layers, gaussian(createRng(hidden.length * 31 + hidden[0]))), actionRepeat: 1, gameVersion: 'v2' };
    for (const seed of [5, 77]) {
      const js = runRampEpisode(model, seed, 500, 800, 1e9);
      assert.deepStrictEqual(runRampEpisode(model, seed, 500, 800, 1e9, wasmAgents(model)), js, `hidden ${hidden}`);
      assert.ok(js.frames >= 1 && js.start >= 500 && js.cap === js.start + 800);
    }
  }
});

test('a ramp episode that survives to its cap has window fitness exactly 1; the cap clips at capMax', () => {
  const solverStudent = () => ({ act: createSolver() });
  for (const seed of SEEDS) {
    const r = runRampEpisode({ gameVersion: 'v2' }, seed, 700, 300, 1e9, solverStudent);
    assert.strictEqual(Number.isInteger(r.start), true);
    assert.strictEqual(env.windowFitness(r.distance, r.start, r.cap - r.start), 1);
    const clipped = runRampEpisode({ gameVersion: 'v2' }, seed, 700, 300, r.start + 100, solverStudent);
    assert.strictEqual(clipped.cap, r.start + 100);
  }
  assert.throws(() => runRampEpisode({ gameVersion: 'v2' }, 1, 700, 300, 700, solverStudent), /no window/);
});

test('ramp strata: after the zeros, uniform on [rampMin, rampMax]; the other episodes are unchanged', () => {
  const C = { target: 100000, window: 2000, startZeroFrac: 0.25, startMin: 2000, startDist: 'points', rampMin: 400, rampMax: 800 };
  for (let g = 0; g < 200; g++) {
    const seeds = Array.from({ length: 8 }, (_, j) => (g * 977 + j * 131) >>> 0);
    const off = env.startScores(seeds, C);
    assert.deepStrictEqual(env.startScores(seeds, { ...C, rampFrac: 0 }), off, 'rampFrac 0 = no ramp key');
    assert.strictEqual(env.rampMask(seeds, { ...C, rampFrac: 0 }), null);
    const on = env.startScores(seeds, { ...C, rampFrac: 0.5 });
    const mask = env.rampMask(seeds, { ...C, rampFrac: 0.5 });
    assert.strictEqual(mask.filter(Boolean).length, 4);
    const zeros = off.filter(x => x === 0).length;
    on.forEach((s, j) => {
      if (j < zeros) assert.strictEqual(s, 0);
      else if (mask[j]) assert.ok(s >= 400 && s <= 800 && j < zeros + 4);
      else assert.strictEqual(s, off[j], 'late starts keep their levels');
    });
    for (const dist of ['log', 'points']) {
      const c = { ...C, startDist: dist, rampFrac: 0.3 };
      assert.deepStrictEqual(env.startScores(seeds, c), env.startScores([...seeds], { ...c }), 'pure function of the seeds');
    }
  }
  // Fractional shares are stratified like startZeroFrac, and ramps never exceed what is left.
  const seeds = [3, 4, 5, 6];
  assert.strictEqual(env.rampMask(seeds, { ...C, startZeroFrac: 1, rampFrac: 0.5 }), null);
  const counts = new Set();
  for (let s = 0; s < 200; s++) counts.add(env.rampMask([s, s + 1, s + 2, s + 3], { ...C, startZeroFrac: 0, rampFrac: 0.3 }).filter(Boolean).length);
  assert.deepStrictEqual([...counts].sort(), [1, 2]);
});

test('generation draws carry the ramp mask only when some episode starts on the ramp', () => {
  const base = { popSize: 8, episodes: 4, pairEpisodes: false };
  const cur = { target: 100000, window: 2000, startZeroFrac: 0.25, startMin: 2000, startDist: 'points', startZeroCap: 2000, rampMin: 400, rampMax: 800 };
  const off = rules.drawGeneration(createRng(9), base, { ...cur, rampFrac: 0 });
  assert.ok(off.pairEp.every(ep => !('ramp' in ep)));
  assert.deepStrictEqual(rules.drawGeneration(createRng(9), base, cur), off, 'no rampFrac key = off');
  const on = rules.drawGeneration(createRng(9), base, { ...cur, rampFrac: 0.5 });
  assert.deepStrictEqual(on.noiseSeeds, off.noiseSeeds, 'no extra trainer RNG draws');
  assert.deepStrictEqual(on.pairEp[0].ramp, [false, true, true, false]);
  assert.deepStrictEqual(on.pairEp[0].seeds, off.pairEp[0].seeds);
});

test('checkConfig: --ramp-frac needs the curriculum, a valid range and room below --target', () => {
  const { DEFAULTS, checkConfig } = require('../trainer/cli.js');
  const cur = { window: 2000, target: 100000, startDist: 'points', startZeroCap: 2000 };
  const ok = (o) => checkConfig({ ...DEFAULTS, ...cur, ...o });
  ok({ rampFrac: 0.25 });
  ok({ rampFrac: 0.5, startZeroFrac: 0.5 });                 // no late episodes left
  ok({ rampFrac: 0.5, startZeroFrac: 0.5, startMin: 2000, target: 2500, startDist: 'log' });   // late rule not applied
  assert.throws(() => checkConfig({ ...DEFAULTS, rampFrac: 0.25 }), /needs the curriculum/);
  assert.throws(() => ok({ rampFrac: 1.5 }), /ramp-frac/);
  assert.throws(() => ok({ rampFrac: -0.1 }), /ramp-frac/);
  assert.throws(() => ok({ rampFrac: 0.6, startZeroFrac: 0.5 }), /must not exceed 1/);
  assert.throws(() => ok({ rampFrac: 0.25, rampMin: 0 }), /ramp-min/);
  assert.throws(() => ok({ rampFrac: 0.25, rampMin: 900, rampMax: 800 }), /ramp-min/);
  assert.throws(() => ok({ rampFrac: 0.25, rampMax: 800, target: 800 + MAX_OVERSHOOT, window: 500, startZeroCap: 500, startMin: 2000, startZeroFrac: 0.75 }), /handover slack/);
  // Off by default: defaults never load ramp.js's teacher.
  assert.strictEqual(DEFAULTS.rampFrac, 0);
});
