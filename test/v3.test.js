'use strict';
// Rules v3: exactly rules v2 up to 100,000 points, then a second log-scale segment
// toward superhuman reaction budgets (research/game-v3.md). Still always survivable.
const test = require('node:test');
const assert = require('node:assert');
const C = require('../core.js');
const { snapshot, decodeKeys, readTraces, GOLDEN_V2_DIR } = require('./golden-format.js');
const { createSolver, createDuckSolver } = require('./solver.js');

function play(version, seed, capScore, startScore, mk = createSolver) {
  const g = C.createGame({ rng: C.createRng(seed), version, startScore });
  const act = mk();
  const capDistance = capScore / C.CFG.scoreCoef;
  do g.step(act(g)); while (g.state === 'running' && g.distance < capDistance);
  return { g, act };
}

test('v3 is exactly v2 below 100,000 points (v2 golden traces replayed on v3)', () => {
  let compared = 0;
  for (const trace of readTraces(GOLDEN_V2_DIR)) {
    if (trace.startScore >= C.V3.startScore) continue;
    const base = C.createRng(trace.seed);
    let calls = 0;
    const game = C.createGame({ rng: () => { calls++; return base(); }, version: 'v3', startScore: trace.startScore });
    const view = { get state() { return game.state; }, dino: game.dino, get speed() { return game.speed; },
      get distance() { return game.distance; }, get obstacles() { return game.obstacles; }, get rngCalls() { return calls; } };
    for (let f = 0; f < trace.rows.length; f++) {
      game.step(decodeKeys(trace.inputs[f]));
      assert.deepStrictEqual(snapshot(view), trace.rows[f], `${trace.name} frame ${f}`);
      compared++;
    }
  }
  assert.ok(compared >= 17000, `compared ${compared} frames`);
});

test('v3 is exactly v2 below 100,000 points (fresh solver games, both solvers)', () => {
  for (const [start, mk] of [[0, createSolver], [60000, createDuckSolver], [95000, createSolver]]) {
    const a = play('v2', 77, Math.min(start + 4000, C.V3.startScore), start || undefined, mk).g;
    const b = play('v3', 77, Math.min(start + 4000, C.V3.startScore), start || undefined, mk).g;
    assert.strictEqual(a.frameIndex, b.frameIndex);
    assert.strictEqual(a.distance, b.distance);
    assert.deepStrictEqual(a.obstacles.map(o => [o.x, o.y, o.w]), b.obstacles.map(o => [o.x, o.y, o.w]));
  }
});

test('v3 schedule: e rises on a log scale from 100k; only the speed cap departs from v2', () => {
  const e = (s) => C.superLevel('v3', s);
  assert.strictEqual(e(0), 0);
  assert.strictEqual(e(100000), 0);
  assert.strictEqual(e(1000000), 1);
  assert.strictEqual(e(5000000), 1);
  assert.ok(Math.abs(e(316228) - 0.5) < 1e-5);
  assert.strictEqual(C.superLevel('v2', 1000000), 0);
  assert.strictEqual(C.difficulty('v3', 300000), C.difficulty('v2', 300000));
  const { rulesAt } = C._internal;
  for (const s of [150000, 300000, 1000000]) {
    const v2 = rulesAt(C.difficulty('v2', s), {}), v3 = rulesAt(C.difficulty('v3', s), {}, e(s));
    for (const k of Object.keys(v2)) if (k !== 'maxSpeed' && k !== 'superLevel') assert.strictEqual(v3[k], v2[k], `${k} at ${s}`);
    assert.ok(v3.maxSpeed > v2.maxSpeed, `speed cap at ${s}`);
  }
  assert.strictEqual(rulesAt(1, {}, 1).maxSpeed, C.V3.maxSpeed);
  // A step at 100k: v2's cap up to 100k, then V3.stepSpeed rising to V3.maxSpeed.
  assert.strictEqual(rulesAt(C.difficulty('v3', 100000), {}, e(100000)).maxSpeed, rulesAt(C.difficulty('v2', 100000), {}).maxSpeed);
  assert.ok(Math.abs(rulesAt(C.difficulty('v3', 100001), {}, e(100001)).maxSpeed - C.V3.stepSpeed) < 0.01);
});

test('v3 speed step at 100k: spawning pauses, the speed jumps on a clear track, both solvers survive', () => {
  for (const [name, mk] of [['jump', createSolver], ['duck', createDuckSolver]]) {
    for (let seed = 1; seed <= 6; seed++) {
      const g = C.createGame({ rng: C.createRng(3500 + seed), version: 'v3', startScore: 98000 });
      const act = mk();
      let eAt = -1, stepAt = -1, prev = g.speed;
      do {
        g.step(act(g));
        if (eAt < 0 && g.superLevel > 0) eAt = g.frameIndex;
        if (stepAt < 0 && g.speed - prev > 1) {
          stepAt = g.frameIndex;
          assert.ok(eAt >= 0 && prev < 20 && g.speed >= C.V3.stepSpeed, `${name} ${seed}: ${prev} -> ${g.speed}`);
          // Nothing placed at the old speed is on the track at the new one.
          assert.ok(g.obstacles.every(o => o.late), `${name} ${seed}: pre-step obstacle on the track`);
          assert.ok(!g.dino.jumping);
        }
        prev = g.speed;
      } while (g.state === 'running' && g.score() < 110000);
      assert.strictEqual(g.state, 'running', `${name} seed ${seed} crashed at ${g.score()}`);
      assert.ok(stepAt > eAt && stepAt - eAt < 120, `${name} ${seed}: pause ${stepAt - eAt} frames`);
    }
  }
});

test('v3 late anchors: obstacles spawned past 100k are anchored late, never on v1/v2', () => {
  const lateOf = (version, start) => {
    const { g } = play(version, 5, start + 3000, start);
    return g.obstacles.map(o => o.late);
  };
  assert.ok(lateOf('v3', 200000).every(x => x === true));
  assert.ok(lateOf('v2', 200000).every(x => x === false));
  assert.ok(lateOf('v3', 50000).every(x => x === false));
});

// The exact solvers must never crash: every v3 sequence is survivable.
test('v3 survivable: both solvers from 100k to 400k points (3 seeds each)', () => {
  for (const [name, mk] of [['jump', createSolver], ['duck', createDuckSolver]]) {
    for (let seed = 1; seed <= 3; seed++) {
      const { g, act } = play('v3', 3300 + seed, 400000, 100000, mk);
      assert.strictEqual(g.state, 'running', `${name} seed ${seed} crashed at ${g.score()}`);
      if (act.stats) assert.strictEqual(act.stats.conflicts, 0);
    }
  }
});

test('v3 survivable at high levels: both solvers, 6 seeds x 3 start levels x 20k points', () => {
  for (const [name, mk] of [['jump', createSolver], ['duck', createDuckSolver]]) {
    for (const start of [300000, 600000, 1000000]) {
      for (let seed = 1; seed <= 6; seed++) {
        const { g, act } = play('v3', 3400 + seed, start + 20000, start, mk);
        assert.strictEqual(g.state, 'running', `${name} start ${start} seed ${seed} crashed at ${g.score()}`);
        if (act.stats) assert.strictEqual(act.stats.conflicts, 0);
      }
    }
  }
  assert.strictEqual(play('v3', 1, 1020000, 1000000).g.speed, C.V3.maxSpeed);
});

// assess --game: a cross-rules assessment plays the asked rules, not the model's own.
test('assess --game v3 plays a v2 model on rules v3 and records both versions', async () => {
  const path = require('path');
  const { evalHeldOut } = require('../trainer/cli.js');
  const { runEpisode, evalSeeds } = require('../trainer/env.js');
  const file = path.join(__dirname, '..', 'runs', 'exp-bc-duck', 'v2-bc-duck-d2w8-s1', 'best.json');
  const model = require(file);
  assert.strictEqual(model.gameVersion, 'v2');
  const warn = console.warn; const warnings = [];
  console.warn = (m) => warnings.push(m);
  let rec;
  try { rec = await evalHeldOut({ model: file, workers: 1, seeds: 2, seedOffset: 0 }, 101000, false, 'v3'); } finally { console.warn = warn; }
  assert.strictEqual(rec.gameVersion, 'v3');
  assert.strictEqual(rec.modelGameVersion, 'v2');
  assert.ok(warnings.some(w => /cross-rules assessment: a v2 model on rules v3/.test(w)));
  const seeds = evalSeeds(2);
  assert.deepStrictEqual(rec.scores, seeds.map(s => runEpisode({ ...model, gameVersion: 'v3' }, s, 101000).score));
  const own = await evalHeldOut({ model: file, workers: 1, seeds: 2, seedOffset: 0 }, 2000);
  assert.strictEqual(own.gameVersion, undefined);
});

// Assessment records: assess.jsonl for new writes, legacy evals.jsonl still read; merged, legacy first.
test('assess.jsonl and legacy evals.jsonl are both read and merged', () => {
  const fs = require('fs'), os = require('os'), path = require('path');
  const { readAssessLines, ASSESS_FILE, ASSESS_FILES } = require('../trainer/store.js');
  assert.strictEqual(ASSESS_FILE, 'assess.jsonl');
  assert.deepStrictEqual(ASSESS_FILES, ['evals.jsonl', 'assess.jsonl']);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'assess-'));
  try {
    assert.deepStrictEqual(readAssessLines(dir), []);
    fs.writeFileSync(path.join(dir, 'assess.jsonl'), '{"n":3}\n');
    assert.deepStrictEqual(readAssessLines(dir), ['{"n":3}']);
    fs.writeFileSync(path.join(dir, 'evals.jsonl'), '{"n":1}\n{"n":2}\n\n');
    assert.deepStrictEqual(readAssessLines(dir).map(l => JSON.parse(l).n), [1, 2, 3]);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
