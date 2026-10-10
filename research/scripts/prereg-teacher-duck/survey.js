'use strict';
// teacher-duck step-1 survey (research/prereg-teacher-duck.md §4): play full rules-v2 games
// from score 0 to --cap points with a solver, on worker threads, one JSONL line per seed.
// Usage: node research/scripts/prereg-teacher-duck/survey.js --solver jump|duck
//          [--base 40000000] [--n 5000] [--cap 1000000] [--workers 16] [--out file.jsonl]
//   jump -> test/solver.js createSolver; duck -> test/solver.js createDuckSolver
// Per seed: crashed, score, frames, downFrames (DOWN held), obstacles passed / ducked,
// mid pteros encountered (from the game: ptero boxes that clear the ducking box but hit the
// standing box), the
// solver's own counters if it exposes them (hook: solver.stats), and for a crash the
// obstacles on screen and the last 30 frames.
// Reads only standard game state (dino, keys, obstacles, speed, score) plus the solver.
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..', '..');

function arg(name, def) { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : def; }

if (isMainThread) {
  const solver = arg('--solver', null);
  if (!['jump', 'duck'].includes(solver)) { console.error('--solver jump|duck is required'); process.exit(2); }
  const base = Number(arg('--base', 40000000)), n = Number(arg('--n', 5000));
  const cap = Number(arg('--cap', 1000000)), workers = Number(arg('--workers', 16));
  const out = arg('--out', `runs/exp-teacher-duck/survey-${solver}-${base}-${n}.jsonl`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const fd = fs.openSync(out, 'w');
  const t0 = Date.now();
  let next = 0, done = 0, crashes = 0, frames = 0;
  const log = (m) => console.log(`[${new Date().toISOString()}] ${m}`);
  log(`survey solver=${solver} seeds ${base}..${base + n - 1} cap ${cap} workers ${workers} -> ${out}`);
  for (let w = 0; w < workers; w++) {
    const wk = new Worker(__filename, { workerData: { solver, cap } });
    const feed = () => { if (next < n) wk.postMessage(base + next++); else wk.postMessage(null); };
    wk.on('message', (r) => {
      fs.writeSync(fd, JSON.stringify(r) + '\n');
      done++; frames += r.frames; if (r.crashed) crashes++;
      if (done % 250 === 0 || done === n) log(`${done}/${n} seeds, ${crashes} crashes, ${(frames / 1e9).toFixed(2)}B frames, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
      if (done === n) { fs.closeSync(fd); log('done'); }
      feed();
    });
    wk.on('error', (e) => { console.error(e); process.exit(1); });
    feed();
  }
} else {
  const Core = require(path.join(ROOT, 'core.js'));
  const Solver = require(path.join(ROOT, 'test', 'solver.js'));
  const { createRng, createGame, CFG, obstacleBox, standBox, STAND_TOP, GROUND, DUCK_H } = Core;
  const make = workerData.solver === 'jump' ? Solver.createSolver : Solver.createDuckSolver;
  if (typeof make !== 'function') throw new Error(`test/solver.js has no ${workerData.solver === 'jump' ? 'createSolver' : 'createDuckSolver'}`);
  const DUCK_TOP = GROUND - DUCK_H + 4;          // top of the ducking box (core.js dinoBox)
  const capDistance = workerData.cap / CFG.scoreCoef;
  const boxOf = (o) => obstacleBox(o);
  const STAND_BOTTOM_TOP = standBox(STAND_TOP).y;   // top of the standing box (y = 100)
  // Mid ptero: its box clears the ducking box but not the standing box, so a grounded dino
  // must duck or jump (high pteros clear both; low ones clear neither).
  const isMid = (o) => { if (o.t.name !== 'ptero') return false; const b = boxOf(o); return b.y + b.h <= DUCK_TOP && b.y + b.h > STAND_BOTTOM_TOP; };
  const describe = (o) => ({ type: o.t.name, size: o.size, x: +o.x.toFixed(2), y: o.y, speedOffset: o.speedOffset, box: boxOf(o) });

  function play(seed) {
    const g = createGame({ rng: createRng(seed), version: 'v2' });
    const sol = make();
    const F = 8, ring = new Float64Array(30 * F);   // y, vy, jumping, ducking, keyJump, keyDown, act (0 none, 1 jump, 2 down), speed
    let frames = 0, downFrames = 0, ri = 0;
    const seen = new Set(), mid = new Set(), passed = new Set();
    const overlap = new Map();                   // obstacle -> { ducked: all overlap frames grounded+ducking }
    do {
      const act = sol(g);
      const d = g.dino;
      if (g.state === 'running') {
        frames++;
        if (act.down) downFrames++;
        const k = (ri++ % 30) * F;
        ring[k] = d.y; ring[k + 1] = d.vy; ring[k + 2] = d.jumping ? 1 : 0; ring[k + 3] = d.ducking ? 1 : 0;
        ring[k + 4] = g.keys.jump ? 1 : 0; ring[k + 5] = g.keys.down ? 1 : 0; ring[k + 6] = act.jump ? 1 : act.down ? 2 : 0; ring[k + 7] = g.speed;
      }
      g.step(act);
      if (g.state !== 'running') break;
      const db = { x: d.x + 4, w: 50 };            // x-extent of the ducking box, which contains the standing box's
      for (const o of g.obstacles) {
        if (o.x >= 600) continue;
        if (!seen.has(o)) { seen.add(o); if (isMid(o)) mid.add(o); }
        const b = boxOf(o);
        if (b.x < db.x + db.w && b.x + b.w > db.x) {
          const st = overlap.get(o) || { ducked: true };
          if (d.jumping || !d.ducking) st.ducked = false;
          overlap.set(o, st);
        } else if (b.x + b.w <= db.x && !passed.has(o)) passed.add(o);
      }
    } while (g.distance < capDistance);
    const crashed = g.state === 'over';
    let ducked = 0, midDucked = 0;
    for (const o of passed) { const st = overlap.get(o); if (st && st.ducked) { ducked++; if (mid.has(o)) midDucked++; } }
    const r = {
      seed, crashed, score: g.score(), frames, downFrames,
      obstaclesSeen: seen.size, obstaclesPassed: passed.size, obstaclesDucked: ducked,
      midEncountered: mid.size, midPassed: [...passed].filter(o => mid.has(o)).length, midDucked,
      solverStats: typeof sol.stats === 'function' ? sol.stats() : (sol.stats ?? null),
    };
    if (crashed) {
      r.onScreen = g.obstacles.filter(o => o.x < 600).map(describe);
      r.last30 = [];
      for (let j = Math.max(0, ri - 30); j < ri; j++) {
        const k = (j % 30) * F;
        r.last30.push({ y: +ring[k].toFixed(2), vy: +ring[k + 1].toFixed(3), jumping: !!ring[k + 2], ducking: !!ring[k + 3],
          keyJump: !!ring[k + 4], keyDown: !!ring[k + 5], act: ['none', 'jump', 'down'][ring[k + 6]], speed: +ring[k + 7].toFixed(3) });
      }
      r.dino = { y: g.dino.y, vy: g.dino.vy, jumping: g.dino.jumping };
    }
    return r;
  }
  parentPort.on('message', (seed) => { if (seed === null) process.exit(0); parentPort.postMessage(play(seed)); });
}
