'use strict';
// teacher-duck (test/solver.js createDuckSolver; research/prereg-teacher-duck.md §4).
const test = require('node:test');
const assert = require('node:assert');
const { createGame, createRng, standBox, obstacleBox, STAND_TOP } = require('../core.js');
const { createSolver, createDuckSolver, duckBox } = require('./solver.js');

const SEEDS = [40100000, 40100001, 40100002, 7, 1234];
const stand = standBox(STAND_TOP);
const overlapsStandX = (o) => { const b = obstacleBox(o); return b.x < stand.x + stand.w && b.x + b.w > stand.x; };

test('applyKeys: releasing DOWN and pressing jump on the same frame jumps (release before press)', () => {
  const g = createGame({ rng: createRng(1), version: 'v2' });
  g.step({ jump: true, down: false });                       // start the run
  while (g.dino.jumping || g.keys.jump) g.step({ jump: false, down: false });
  g.step({ jump: false, down: true });
  assert.ok(g.dino.ducking && !g.dino.jumping);
  g.step({ jump: true, down: false });
  assert.ok(!g.dino.ducking && g.dino.jumping, 'jump taken on the frame DOWN is released');
});

test('the duck box is the ducking hitbox: it clears mid pterodactyls and not low ones', () => {
  const g = createGame({ rng: createRng(1), version: 'v2' });
  const db = duckBox(g.dino);
  assert.deepStrictEqual(db, { x: g.dino.x + 4, y: 120, w: 50, h: 18 });
  const ptero = (y) => obstacleBox({ t: { name: 'ptero' }, x: 0, y });
  assert.ok(ptero(82).y + ptero(82).h <= db.y, 'mid clears');
  assert.ok(ptero(82).y + ptero(82).h > stand.y, 'mid hits the standing box');
  assert.ok(ptero(108).y + ptero(108).h > db.y, 'low hits the duck box');
});

test('the duck solver acts exactly like createSolver until its first mid-pterodactyl decision', () => {
  let compared = 0;
  for (const seed of SEEDS) {
    const a = createGame({ rng: createRng(seed), version: 'v2' }), b = createGame({ rng: createRng(seed), version: 'v2' });
    const jump = createSolver(), duck = createDuckSolver();
    for (let f = 0; f < 200000 && a.state !== 'over'; f++) {
      const ka = jump(a), kb = duck(b);
      if (duck.stats.midEncountered > 0) break;
      assert.deepStrictEqual(kb, ka, `seed ${seed} frame ${f}`);
      a.step(ka); b.step(kb); compared++;
    }
    assert.ok(duck.stats.midEncountered > 0, `seed ${seed} meets a mid pterodactyl`);
  }
  assert.ok(compared > 1000);
});

test('the duck solver ducks under mid pterodactyls, survives, never presses DOWN airborne, and jumps right after', () => {
  for (const seed of SEEDS) {
    const g = createGame({ rng: createRng(seed), version: 'v2' }), act = createDuckSolver();
    let duckedPasses = 0, prev = { jump: false, down: false }, f = 0;
    while (g.state !== 'over' && (g.state === 'idle' || g.score() < 20000)) {
      const k = act(g);
      assert.ok(!(k.down && g.dino.jumping), `seed ${seed} frame ${f}: DOWN while airborne`);
      assert.ok(!(k.down && k.jump), 'never both keys');
      const pressed = k.jump && !prev.jump && g.state === 'running';
      const wasDucking = prev.down;
      g.step(k); f++;
      if (pressed && wasDucking) assert.ok(g.dino.jumping, 'jump right after a duck takes off');
      // A mid pterodactyl over the standing box while the dino is grounded: it must be ducking.
      for (const o of g.obstacles) {
        const b = obstacleBox(o);
        if (g.state === 'running' && !g.dino.jumping && overlapsStandX(o) && b.y + b.h > stand.y && b.y + b.h <= 120) {
          assert.ok(g.dino.ducking, `seed ${seed} frame ${f}: grounded under a mid pterodactyl without ducking`);
          duckedPasses++;
        }
      }
      prev = k;
    }
    assert.strictEqual(g.state, 'running', `seed ${seed}: no crash to 20,000`);
    const s = act.stats;
    assert.ok(s.midDucked > 0 && duckedPasses > 0 && s.downFrames > 0);
    assert.strictEqual(s.midEncountered, s.midDucked + s.fallback);
    assert.strictEqual(s.fallback, Object.values(s.fallbackWhy).reduce((a, x) => a + x, 0));
    assert.strictEqual(s.conflicts, 0);
  }
});
