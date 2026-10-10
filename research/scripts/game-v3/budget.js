'use strict';
// Reaction-budget probe for rules v3 (research/game-v3.md). Single thread, seconds.
//
//   node research/scripts/game-v3/budget.js [--versions v2,v3] [--levels 99000,101000,110000,150000,200000,500000,1000000]
//        [--games 12] [--span 2000] [--teacher jump|duck|both] [--set maxSpeed=60,curve=0.5] [--json]
//
// Plays solver-driven games (test/solver.js createSolver / createDuckSolver) from each
// start level (createGame startScore: the speed starts at that level's cap) for --span
// points, and records every obstacle on the first frame it is visible (o.x < W, after
// placeFairly's parking). For each MUST-ACT obstacle (it would hit a grounded, standing
// dino on some whole frame at the current speed) the reaction budget is
//   budget = latest valid action start - first visibility   (frames; x FRAME_MS for ms)
// where the latest valid action is the most lenient one for that obstacle alone:
//   - a jump: max over RELEASE_FRAMES of tau - a_r, over the releases whose arc clears it
//     (window b_r - a_r - delta >= 0; tau, a, b, delta as in core.js requirement());
//   - a duck (mid pterodactyls, whose box clears the ducking box): tau, the frame the
//     standing box would start to overlap it.
// Not must-act: overhead obstacles (high pterodactyls: stay grounded) and "tunnelled"
// ones that a grounded dino would skip between two frames (counted separately).
// Also recorded: the canonical takeoff (requirement().from, the solver's action) relative
// to visibility; negative means the solver acts before the obstacle is on screen.
// --set overrides V3 fields in-process (exploration only; the shipped numbers live in core.js).
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const C = require(path.join(ROOT, 'core.js'));
const { createSolver, createDuckSolver, duckBox } = require(path.join(ROOT, 'test', 'solver.js'));
const { requirement, jumpArc } = C._internal;

const arg = (name, def) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : def; };
const versions = arg('--versions', 'v2,v3').split(',');
const levels = arg('--levels', '99000,101000,110000,150000,200000,500000,1000000').split(',').map(Number);
const games = Number(arg('--games', 12)), span = Number(arg('--span', 2000));
const teacher = arg('--teacher', 'both');
const asJson = process.argv.includes('--json');
for (const kv of (arg('--set', '') || '').split(',').filter(Boolean)) {
  const [k, v] = kv.split('=');
  if (!(k in C.V3)) throw new Error(`unknown V3 field ${k}`);
  C.V3[k] = Number(v);
}

const stand = C.standBox(C.STAND_TOP);
const FRONT = stand.x + stand.w;
const BUDGET = 150 / C.FRAME_MS;   // 9 frames

// Per-obstacle measurement at its first visible frame.
function measure(o, s) {
  const b = C.obstacleBox(o), v = s + o.speedOffset;
  if (b.y + b.h <= stand.y) return { kind: 'overhead' };
  let hit = false;   // would a grounded standing dino overlap it on some whole frame?
  for (let k = 1; k * v < b.x + b.w - stand.x + v; k++) {
    const x = b.x - k * v;
    if (x < FRONT && x + b.w > stand.x) { hit = true; break; }
  }
  if (!hit) return { kind: 'tunnelled' };
  const tau = (b.x - FRONT) / v, delta = (b.w + stand.w) / v, H = stand.y + stand.h - b.y;
  let latest = -Infinity;
  for (const rel of C.RELEASE_FRAMES) {
    const h = jumpArc(s, rel);
    const a = h.findIndex(x => x > H) + 1, bb = h.findLastIndex(x => x > H) + 1;
    if (a > 0 && bb - a - delta >= 0) latest = Math.max(latest, tau - a);
  }
  const duckable = o.t.name === 'ptero' && b.y + b.h <= duckBox({ x: C.DINO_X }).y;
  if (duckable) latest = Math.max(latest, tau);
  const r = requirement(o, s);
  return { kind: duckable ? 'mid' : o.t.name === 'ptero' ? 'low' : 'cactus', budget: latest, canonical: r && !r.grounded ? r.from : null, v };
}

function play(version, level, seed, mk) {
  const g = C.createGame({ rng: C.createRng(seed), version, startScore: level });
  const act = mk();
  const seen = new WeakSet(), out = [];
  const cap = (level + span) / C.CFG.scoreCoef;
  do {
    g.step(act(g));
    for (const o of g.obstacles) {
      if (seen.has(o) || o.x >= C.W) continue;
      seen.add(o);
      out.push(measure(o, g.speed));
    }
  } while (g.state === 'running' && g.distance < cap);
  return { rows: out, crashed: g.state !== 'running', score: g.score(), speed: g.speed };
}

const q = (xs, p) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN; };
const teachers = teacher === 'both' ? [['jump', createSolver], ['duck', createDuckSolver]] : [[teacher, teacher === 'duck' ? createDuckSolver : createSolver]];
const results = [];
const t0 = Date.now();
for (const version of versions) {
  for (const level of levels) {
    for (const [tn, mk] of teachers) {
      const rows = [];
      let crashes = 0, speed = 0;
      for (let i = 0; i < games; i++) {
        const r = play(version, level, 9100000 + i, mk);
        rows.push(...r.rows); crashes += r.crashed ? 1 : 0; speed = r.speed;
      }
      const must = rows.filter(r => r.budget !== undefined);
      const b = must.map(r => r.budget);
      const canon = must.filter(r => r.canonical !== null).map(r => r.canonical);
      const res = {
        version, level, teacher: tn, games, crashes, speed: +speed.toFixed(2), mustAct: must.length,
        overhead: rows.filter(r => r.kind === 'overhead').length, tunnelled: rows.filter(r => r.kind === 'tunnelled').length,
        medianFrames: +q(b, 0.5).toFixed(2), p10Frames: +q(b, 0.1).toFixed(2), p90Frames: +q(b, 0.9).toFixed(2),
        medianMs: Math.round(q(b, 0.5) * C.FRAME_MS), p90Ms: Math.round(q(b, 0.9) * C.FRAME_MS),
        shareUnder150: +(b.filter(x => x < BUDGET).length / Math.max(1, b.length)).toFixed(3),
        canonicalBeforeVisible: +(canon.filter(x => x < 0).length / Math.max(1, canon.length)).toFixed(3),
        canonicalMedianFrames: +q(canon, 0.5).toFixed(2),
      };
      results.push(res);
      if (!asJson) {
        console.log(`${version} ${String(level).padStart(7)} ${tn.padEnd(4)} speed ${res.speed.toFixed(1).padStart(5)}  must-act ${String(res.mustAct).padStart(4)}` +
          ` (tunnelled ${res.tunnelled}, overhead ${res.overhead})  budget median ${res.medianFrames.toFixed(1)} f / ${res.medianMs} ms,` +
          ` p90 ${res.p90Frames.toFixed(1)} f / ${res.p90Ms} ms, share<150ms ${(100 * res.shareUnder150).toFixed(1)}%` +
          `  canonical median ${res.canonicalMedianFrames.toFixed(1)} f (before visible ${(100 * res.canonicalBeforeVisible).toFixed(1)}%)  crashes ${crashes}/${games}`);
      }
    }
  }
}
if (asJson) console.log(JSON.stringify({ V3: C.V3, games, span, results }, null, 1));
else console.log(`V3 ${JSON.stringify(C.V3)}; ${((Date.now() - t0) / 1000).toFixed(1)} s`);
