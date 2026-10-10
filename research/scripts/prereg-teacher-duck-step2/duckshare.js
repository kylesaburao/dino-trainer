'use strict';
// Duck share in play (research/prereg-teacher-duck-step2.md §5.3 step 4).
// Plays full games (0 -> --target; rules v2 by default, see Rules below) on seeds BASE..BASE+N-1 with a model
// (policy.js createAgent, JS forward pass) or with the duck teacher (test/solver.js
// createDuckSolver), single-threaded, and classifies every DUCKABLE obstacle: one whose box
// overlaps the standing box vertically but clears the ducking box (mid pterodactyls; computed
// from the boxes, not from o.y). For each, over the frames (state after the step, i.e. at
// collision time) during which its box overlaps the standing box's x-range [56, 86):
//   ducked  = grounded and ducking on every such frame;
//   jumped  = airborne on every such frame;
//   mixed   = anything else;  crashed = the game ended while it overlapped.
// Obstacles not yet reached when a game ends at the target are 'pending' and not counted.
// Duck share = ducked / (ducked + jumped + mixed + crashed).
// Also per game: DOWN frames (running frames whose action holds DOWN), DOWN while airborne
// (a fast-drop; the teacher never does it).
// Counter control (teacher only; the script exits 1 if it fails): DOWN frames equal
// act.stats.downFrames; DOWN-airborne = 0; mixed = crashed = 0; and
// ducked <= stats.midDucked <= ducked + pending (the teacher counts an obstacle at its decision
// frame, which can precede the end of the game by less than one obstacle's approach).
// Usage: node duckshare.js (--model <best.json> | --teacher duck) [--base 30003000] [--n 10]
//          [--target 1000000] [--game v2|v3] [--out f.json]
// Rules (research/prereg-v3-clones.md §4.0): a model plays its own gameVersion (v2 or v3), the
// teacher plays --game (default v2). --game on a model is a cross-rules override (warned). The
// report gains game/modelGameVersion only when --game is given, so the default is unchanged.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const Core = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));
const { createDuckSolver, duckBox } = require(path.join(ROOT, 'test', 'solver.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const modelFile = opt('--model', null);
const teacher = opt('--teacher', null);
if (!!modelFile === !!teacher) throw new Error('give exactly one of --model <best.json> or --teacher duck');
if (teacher && teacher !== 'duck') throw new Error(`--teacher must be duck, got ${teacher}`);
const BASE = Number(opt('--base', 30003000)), N = Number(opt('--n', 10)), TARGET = Number(opt('--target', 1000000));
const m = modelFile ? JSON.parse(fs.readFileSync(path.resolve(modelFile), 'utf8')) : null;
const PLAYABLE = ['v2', 'v3'];
if (m && !PLAYABLE.includes(m.gameVersion)) throw new Error(`model gameVersion ${m.gameVersion}`);
const GAME_OPT = opt('--game', null);
if (GAME_OPT !== null && !PLAYABLE.includes(GAME_OPT)) throw new Error(`--game must be one of ${PLAYABLE.join(', ')}`);
const GAME = GAME_OPT || (m ? m.gameVersion : 'v2');
if (m && GAME !== m.gameVersion) console.warn(`WARNING: cross-rules duck share: a ${m.gameVersion} model on rules ${GAME}`);
const cap = TARGET / Core.CFG.scoreCoef;
const stand = Core.standBox(Core.STAND_TOP);

function duckable(o, duckY) {
  const b = Core.obstacleBox(o);
  return b.y < stand.y + stand.h && b.y + b.h > stand.y && b.y + b.h <= duckY;
}

function playSeed(seed) {
  const g = Core.createGame({ rng: Core.createRng(seed), version: GAME });
  const actor = teacher ? createDuckSolver() : null;
  const agent = m ? P.createAgent(m) : null;
  const track = new Map();       // obstacle -> { overlap, ducked, air, done, crashed }
  const res = { seed, ducked: 0, jumped: 0, mixed: 0, crashed: 0, pending: 0, downFrames: 0, downAir: 0, frames: 0 };
  const duckY = duckBox(g.dino).y;
  const close = (t) => {
    if (t.crashed) res.crashed++;
    else if (t.ducked === t.overlap) res.ducked++;
    else if (t.air === t.overlap) res.jumped++;
    else res.mixed++;
  };
  do {
    const running = g.state === 'running';
    const airborne = g.dino.jumping;
    const k = actor ? actor(g) : agent.act(g);
    if (running && k.down) { res.downFrames++; if (airborne) res.downAir++; }
    g.step(k);
    res.frames++;
    if (g.state !== 'running') continue;
    for (const o of g.obstacles) {
      if (o.t.name !== 'ptero' || !duckable(o, duckY)) continue;
      const b = Core.obstacleBox(o);
      let t = track.get(o);
      const over = b.x < stand.x + stand.w && b.x + b.w > stand.x;
      if (over) {
        if (!t) { t = { overlap: 0, ducked: 0, air: 0, done: false, crashed: false }; track.set(o, t); }
        t.overlap++;
        if (g.dino.jumping) t.air++;
        else if (g.dino.ducking) t.ducked++;
      } else if (t && !t.done && b.x + b.w <= stand.x) { t.done = true; close(t); }
    }
  } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
  const crashedGame = g.state === 'over';
  for (const [o, t] of track) {
    if (t.done) continue;
    const b = Core.obstacleBox(o);
    const over = b.x < stand.x + stand.w && b.x + b.w > stand.x;
    if (crashedGame && over) { t.crashed = true; close(t); } else if (t.overlap > 0) close(t);
  }
  // Duckable obstacles on screen but never reached: pending (only matters for the control).
  for (const o of g.obstacles) if (o.t.name === 'ptero' && duckable(o, duckY) && !track.has(o) && o.x < Core.W) res.pending++;
  res.score = Math.min(g.score(), TARGET);
  res.pass = res.score >= TARGET;
  if (actor) res.stats = JSON.parse(JSON.stringify(actor.stats));
  return res;
}

const t0 = Date.now();
const games = [];
for (let s = BASE; s < BASE + N; s++) {
  const r = playSeed(s);
  games.push(r);
  console.error(`seed ${s}: score ${r.score} ducked ${r.ducked} jumped ${r.jumped} mixed ${r.mixed} crashed ${r.crashed} downAir ${r.downAir} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
const sum = (k) => games.reduce((a, r) => a + r[k], 0);
const met = sum('ducked') + sum('jumped') + sum('mixed') + sum('crashed');
const report = {
  model: modelFile ? path.resolve(modelFile) : null, teacher, layers: m ? m.layers : null, base: BASE, n: N, target: TARGET,
  passed: games.filter(r => r.pass).length, met, ducked: sum('ducked'), jumped: sum('jumped'), mixed: sum('mixed'), crashed: sum('crashed'),
  duckShare: met ? sum('ducked') / met : null, perGameDuckShare: games.map(r => { const n = r.ducked + r.jumped + r.mixed + r.crashed; return n ? r.ducked / n : null; }),
  downFrames: sum('downFrames'), downAir: sum('downAir'), games, seconds: (Date.now() - t0) / 1000,
};
if (GAME_OPT !== null) Object.assign(report, { game: GAME, modelGameVersion: m ? m.gameVersion : null });
let ok = true;
if (teacher) {
  const fails = [];
  for (const r of games) {
    if (r.downFrames !== r.stats.downFrames) fails.push(`seed ${r.seed}: downFrames ${r.downFrames} != stats ${r.stats.downFrames}`);
    if (r.downAir !== 0) fails.push(`seed ${r.seed}: DOWN while airborne ${r.downAir}`);
    if (r.mixed || r.crashed) fails.push(`seed ${r.seed}: mixed ${r.mixed} crashed ${r.crashed}`);
    if (!(r.ducked <= r.stats.midDucked && r.stats.midDucked <= r.ducked + r.pending)) fails.push(`seed ${r.seed}: ducked ${r.ducked} pending ${r.pending} vs stats.midDucked ${r.stats.midDucked}`);
  }
  report.counterControl = { pass: fails.length === 0, fails };
  ok = fails.length === 0;
}
console.log(JSON.stringify({ ...report, games: undefined }, null, 1));
const out = opt('--out', null);
if (out) fs.writeFileSync(out, JSON.stringify(report, null, 1) + '\n');
if (!ok) { console.error('COUNTER CONTROL FAILED'); process.exit(1); }
