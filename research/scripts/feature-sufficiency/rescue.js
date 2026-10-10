'use strict';
// For each crash (seed, level) of the shipped model: replay its exact actions up to
// frame f, then try rescue macros from f: (a) hand over to the exact solver,
// (b) hold DOWN for n frames (fast-drop if airborne, duck if grounded) then solver,
// (c) NONE for n frames then solver. Report the latest f (frames before the crash)
// from which some macro survives the crash obstacle + 150 frames, and whether the
// crash obstacle was visible (x < 600) at that f and at the model's last takeoff.
// usage: node rescue.js <crashlist.json> <out.json>
const R = '/Users/kyle/Desktop/dino-game/';
const Core = require(R + 'core.js');
const P = require(R + 'policy.js');
const model = require(R + 'model.js');
const { createSolver } = require(R + 'test/solver.js');
const fs = require('fs');
const { createGame, createRng, CFG, W } = Core;
const NONE = P.ACTIONS[0], DOWN = P.ACTIONS[2];

const list = JSON.parse(fs.readFileSync(process.argv[2]));
const results = [];

function modelRun(seed, level, cap) {
  const g = createGame({ rng: createRng(seed), version: 'v2', startScore: level || undefined });
  const ag = P.createAgent(model);
  const acts = [], vis = [], jumping = [];
  do {
    const a = ag.act(g);
    acts.push(a);
    g.step(a);
    vis.push(g.obstacles.map(o => o.x < W));
    jumping.push(g.dino.jumping);
  } while (g.state === 'running' && g.distance < cap);
  const d = g.dino, db = Core.standBox(d.y);
  const hit = g.obstacles.find(o => { const q = Core.obstacleBox(o); return db.x < q.x + q.w && db.x + db.w > q.x && db.y < q.y + q.h && db.y + db.h > q.y; }) || g.obstacles[0];
  const visibleFor = hit ? (W - hit.x) / (g.speed + hit.speedOffset) : null;
  return { acts, state: g.state, n: acts.length, jumping, visibleFor, hitType: hit && hit.t.name };
}

function tryRescue(seed, level, acts, f, macro, endFrame) {
  const g = createGame({ rng: createRng(seed), version: 'v2', startScore: level || undefined });
  for (let i = 0; i < f; i++) g.step(acts[i]);
  const sol = createSolver();
  let i = f;
  for (; i < endFrame && g.state === 'running'; i++) {
    const k = i - f;
    let a;
    if (macro.kind === 'down' && k < macro.n) a = DOWN;
    else if (macro.kind === 'none' && k < macro.n) a = NONE;
    else a = sol(g);
    g.step(a);
  }
  return g.state === 'running';
}

const MACROS = [{ kind: 'solver' }];
for (const n of [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]) MACROS.push({ kind: 'down', n });
for (const n of [1, 2, 4, 8]) MACROS.push({ kind: 'none', n });

for (const c of list) {
  const cap = (c.level + (c.level ? 10000 : 2000)) / CFG.scoreCoef;
  const run = modelRun(c.seed, c.level, cap);
  const crashF = run.n - 1;                 // index of the step that crashed
  const end = crashF + 150;
  let latest = null, latestMacro = null;
  const ok = [];
  for (let back = 1; back <= 45; back++) {
    const f = crashF - back + 1;            // replay model actions [0, f), rescue from f
    if (f < 0) break;
    const m = MACROS.find(m => tryRescue(c.seed, c.level, run.acts, f, m, end));
    ok.push(m ? m.kind + (m.n ? m.n : '') : '-');
    if (m && latest === null) { latest = back; latestMacro = m; }
  }
  // last takeoff frame before the crash
  let tk = null;
  for (let i = crashF; i > 0; i--) if (run.jumping[i] && !run.jumping[i - 1]) { tk = crashF - i; break; }
  results.push({ ...c, crashF, visibleFramesBefore: run.visibleFor && Math.round(run.visibleFor * 10) / 10, hitType: run.hitType, latestRescueFramesBefore: latest, macro: latestMacro, takeoffFramesBefore: tk, rescueProfile: ok.join(' ') });
  console.error(c.seed, c.level, c.type, 'latest rescue', latest, latestMacro && JSON.stringify(latestMacro), 'takeoff', tk, 'visible', run.visibleFor && run.visibleFor.toFixed(1));
}
fs.writeFileSync(process.argv[3], JSON.stringify(results, null, 1));
