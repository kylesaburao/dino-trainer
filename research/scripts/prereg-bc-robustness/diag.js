'use strict';
// Per-failure diagnosis for the robustness scan (research/prereg-bc-robustness.md §4).
// For each failing (arm, seed) of a scan file:
//   1. replays the model with the JS forward pass (policy.js createAgent) and asserts the
//      replay dies at the scan's recorded score (the scan may have used --wasm);
//   2. classifies the death like tools/wall-diag.js (killer type, ptero height class,
//      speed, dino state, last-30-frame actions), adapted to best.json params;
//   3. arm S: plays test/solver.js on the same seed from 0 to the target;
//   4. oracle-disagreement trace: on the model's own trajectory, a shadow solver is
//      queried every frame (only the model's keys are applied) and every frame where
//      the two differ is logged. Validity of the query (see solver.js): the solver's
//      only private state is `hold`, read only while the dino is airborne and set only
//      on a takeoff the solver itself chose. So on ground frames the query is exact;
//      on airborne frames it is exact only if the shadow chose the same takeoff frame;
//      airborne frames of a model-initiated takeoff the shadow did not choose have no
//      defined solver answer and are counted as 'undefined', not compared;
//   5. drift: the solver's own trajectory on the same seed is recorded per frame (dino
//      y/ducking, obstacle-list hash) and compared frame-aligned to the model's.
// Usage: node diag.js --arm C1 --scan <C1.jsonl> --model <best.json> --out <diag-C1.json>
//          [--seeds a,b,c] (default: every failing seed in --scan)
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const { createRng, createGame, standBox, obstacleBox, GROUND, DUCK_H, CFG } = require(path.join(ROOT, 'core.js'));
const { createAgent } = require(path.join(ROOT, 'policy.js'));
const { createSolver } = require(path.join(ROOT, 'test', 'solver.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const ARM = opt('--arm');
const lines = fs.readFileSync(path.resolve(opt('--scan')), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const header = lines.find(l => l.header);
const TARGET = header.target;
const recorded = new Map(lines.filter(l => l.seed !== undefined).map(l => [l.seed, l]));
const m = JSON.parse(fs.readFileSync(path.resolve(opt('--model')), 'utf8'));
const model = { layers: m.layers, params: m.params, actionRepeat: m.actionRepeat, gameVersion: m.gameVersion };
const seeds = opt('--seeds') ? opt('--seeds').split(',').map(Number) : [...recorded.values()].filter(r => !r.pass).map(r => r.seed);

const PTERO_CLASS = { 108: 'low', 82: 'mid', 58: 'high' };
const actName = (k) => (k.jump ? 'jump' : k.down ? 'duck' : 'none');
const overlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
const rle = (xs) => { const o = []; for (const x of xs) { if (o.length && o[o.length - 1][0] === x) o[o.length - 1][1]++; else o.push([x, 1]); } return o.map(([a, n]) => `${a}x${n}`).join(' '); };
const capDistance = TARGET / CFG.scoreCoef;
// Cheap per-frame signature of the obstacle list (type, size, x, y).
function obsHash(game) {
  let h = 2166136261 >>> 0;
  for (const o of game.obstacles) {
    const v = `${o.t.name}${o.size}|${o.x.toFixed(9)}|${o.y}`;
    for (let i = 0; i < v.length; i++) h = Math.imul(h ^ v.charCodeAt(i), 16777619) >>> 0;
  }
  return h;
}

// The solver's own game on this seed, up to maxFrames (or crash / target).
function solverRun(seed, maxFrames) {
  const game = createGame({ rng: createRng(seed), version: model.gameVersion });
  const act = createSolver();
  const keep = maxFrames !== undefined;
  const y = keep ? new Float64Array(maxFrames + 1) : null, duck = keep ? new Uint8Array(maxFrames + 1) : null, oh = keep ? new Uint32Array(maxFrames + 1) : null;
  let f = 0;
  do {
    if (keep && f <= maxFrames) { y[f] = game.dino.y; duck[f] = game.dino.ducking ? 1 : 0; oh[f] = obsHash(game); }
    game.step(act(game));
    f++;
  } while (game.state === 'running' && game.distance < capDistance && (!keep || f <= maxFrames));
  return { score: Math.min(game.score(), TARGET), crashed: game.state === 'over', frames: f, y, duck, oh };
}

function diagnose(seed) {
  const rec = recorded.get(seed);
  const game = createGame({ rng: createRng(seed), version: model.gameVersion });
  const agent = createAgent(model);
  const shadow = createSolver();
  const acts = [];                // per-frame action names (model)
  const ys = [], ducks = [], ohs = [], scores = [];
  const dis = [];                 // disagreements
  let undefinedAir = 0, jumpValid = true, wasDucking = false, lastAir = -Infinity, frame = 0;
  const firstSeen = new Map();    // obstacle -> first frame with x < 600 (observable)
  do {
    for (const o of game.obstacles) if (o.x < 600 && !firstSeen.has(o)) firstSeen.set(o, frame);
    scores.push(game.score()); ys.push(game.dino.y); ducks.push(game.dino.ducking ? 1 : 0); ohs.push(obsHash(game));
    const airborne = game.dino.jumping;
    const k = agent.act(game);
    const sk = shadow(game);      // query only; never applied
    const a = actName(k), sa = actName(sk);
    if (game.state === 'running') {
      if (!airborne) {
        if (a !== sa) {
          // keyheld: the jump key is already down on the ground; solver says release, model
          // holds. Neither presses, so this frame has no effect on the dino.
          const kind = game.keys.jump && a === 'jump' && sa === 'none' ? 'ground-keyheld' : 'ground';
          dis.push({ frame, score: game.score(), kind, model: a, solver: sa });
        }
      } else if (!jumpValid) undefinedAir++;
      else if (a !== sa) dis.push({ frame, score: game.score(), kind: 'air', model: a, solver: sa });
    }
    acts.push(a);
    if (game.dino.jumping) lastAir = frame;
    wasDucking = !game.dino.jumping && (game.dino.ducking || k.down);
    game.step(k);
    // A takeoff this frame: the shadow's airborne answers are defined only if it chose it.
    if (!airborne && game.dino.jumping) jumpValid = sa === 'jump';
    frame++;
  } while (game.state === 'running' && game.distance < capDistance);
  const score = Math.min(game.score(), TARGET);
  const crashed = game.state === 'over';
  if (score !== rec.score) throw new Error(`${ARM} seed ${seed}: JS replay score ${score} != scan score ${rec.score}`);
  const d = game.dino;
  const box = wasDucking && !d.jumping ? { x: d.x + 4, y: GROUND - DUCK_H + 4, w: 50, h: 18 } : standBox(d.y);
  const hitters = crashed ? game.obstacles.filter(o => overlap(box, obstacleBox(o)) || overlap(standBox(d.y), obstacleBox(o))) : [];
  const killer = hitters.sort((p, q) => p.x - q.x)[0] || null;
  const crashFrame = frame - 1;
  const killerName = killer ? (killer.t.name === 'ptero' ? `ptero-${PTERO_CLASS[killer.y] || killer.y}` : `cactus-${killer.t.name}${killer.size > 1 ? 'x' + killer.size : ''}`) : (crashed ? 'unknown' : 'survived');
  // Near-fatal window (fixed before results): from the frame the killing obstacle first became
  // observable (x < 600) to the crash; at least the last 30 frames.
  const killerSeen = killer && firstSeen.has(killer) ? firstSeen.get(killer) : crashFrame;
  const nearFrom = Math.min(killerSeen, crashFrame - 30);
  const effective = dis.filter(x => x.kind !== 'ground-keyheld');
  const early = effective.filter(x => x.frame < nearFrom);

  // Arm S and drift: the solver's own trajectory.
  const S = solverRun(seed);
  const traj = solverRun(seed, crashFrame);
  let firstDinoDiff = -1, dinoDiffFrames = 0, dinoDiffBeforeNear = 0, firstWorldDiff = -1;
  const upto = Math.min(crashFrame, traj.frames - 1);
  for (let f = 0; f <= upto; f++) {
    if (ys[f] !== traj.y[f] || ducks[f] !== traj.duck[f]) {
      if (firstDinoDiff < 0) firstDinoDiff = f;
      dinoDiffFrames++;
      if (f < nearFrom) dinoDiffBeforeNear++;
    }
    if (firstWorldDiff < 0 && ohs[f] !== traj.oh[f]) firstWorldDiff = f;
  }
  const verdict = effective.length === 0 ? 'no-effective-disagreement'
    : early.length > 0 ? 'covariate-shift (first effective disagreement before the near-fatal window)'
      : 'in-distribution action error (all effective disagreements in the near-fatal window)';
  return {
    arm: ARM, seed, score, crashFrame, speed: game.speed, killer: killerName,
    state: !crashed ? '-' : d.jumping ? 'air' : crashFrame - lastAir <= 3 ? 'landing' : wasDucking ? 'ducking' : 'ground',
    last30: rle(acts.slice(-30)),
    solver: { pass: S.score >= TARGET, score: S.score, crashed: S.crashed },
    trace: {
      disagreements: dis.length, effective: effective.length, keyheld: dis.length - effective.length, undefinedAirFrames: undefinedAir,
      firstEffective: effective[0] || null, effectiveBeforeNearWindow: early.length, nearWindowFrom: nearFrom,
      killerFirstObservable: killerSeen, lastEffective: effective.slice(-5),
      drift: { firstDinoDiffFrame: firstDinoDiff, firstDinoDiffScore: firstDinoDiff >= 0 ? scores[firstDinoDiff] : null, dinoDiffFrames, dinoDiffBeforeNear, firstWorldDiffFrame: firstWorldDiff, solverTrajFrames: traj.frames, solverTrajCrashed: traj.crashed },
      verdict,
    },
  };
}

const out = [];
for (const s of seeds) { const r = diagnose(s); out.push(r); console.log(JSON.stringify(r)); }
fs.writeFileSync(path.resolve(opt('--out')), JSON.stringify(out, null, 1) + '\n');
