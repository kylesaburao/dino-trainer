'use strict';
// teacher-duck step 1 (research/prereg-teacher-duck.md §5): classify survey crashes.
// Replays each crashed seed with the solver, checks that the replay reproduces the survey
// record (score, running frames, last 30 frames), finds the obstacle(s) the dino hit, and
// classifies the crash:
//   failure1  the ducking box met an obstacle (DOWN held into the next obstacle; subtype
//             'duck-box-wider' if the standing box would have cleared it), or a takeoff was
//             refused because the dino was still ducking ('jump-refused-while-ducking')
//   failure2  DOWN was pressed while airborne during the fatal jump (fast-drop: an altered arc)
//   other     'duck-window' (grounded and standing under a mid pterodactyl: DOWN released too
//             early or pressed too late), 'jump-plan' (airborne, arc unaltered: the jump plan
//             itself), 'missed-takeoff' (grounded, standing, hit by an obstacle that must be jumped),
//             'unknown'
// It also replays createSolver (teacher-jump) on the seed to the crash frame + 100 and
// reports whether it survives there (if not, the game or harness is suspect, as for the control).
//
// Usage:
//   node research/scripts/prereg-teacher-duck/crash-classify.js <survey.jsonl> [--solver duck|jump]
//        [--solver-module path/to/solver.js] [--cap 1000000] [--json]
//     reads survey.js output (one JSON line per seed; only crashed lines are replayed)
//   node research/scripts/prereg-teacher-duck/crash-classify.js --find N [--base 40100000]
//        [--cap 20000] [--solver-module ...] [--json]
//     self-test: plays seeds base..base+N-1 to --cap (single thread), records crashes in
//     survey.js's format, then classifies them. Keep --base off the survey's seeds
//     (40,000,000-40,004,999).
// --solver-module loads createDuckSolver / createSolver from another file (a mutated solver for
// testing the classifier); the game is always this repo's core.js.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..', '..');
const { createGame, createRng, CFG, obstacleBox, standBox, STAND_TOP, GROUND, DUCK_H } = require(path.join(ROOT, 'core.js'));
const { createSolver: controlSolver } = require(path.join(ROOT, 'test', 'solver.js'));

const argv = process.argv.slice(2);
const arg = (name, def) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : def; };
const solverKind = arg('--solver', 'duck');
const solverModule = require(path.resolve(arg('--solver-module', path.join(ROOT, 'test', 'solver.js'))));
const make = solverKind === 'jump' ? solverModule.createSolver : solverModule.createDuckSolver;
if (typeof make !== 'function') throw new Error(`no ${solverKind} solver in the solver module`);

const DUCK_TOP = GROUND - DUCK_H + 4;
const STAND = standBox(STAND_TOP);
const duckBox = (dino) => ({ x: dino.x + 4, y: DUCK_TOP, w: 50, h: 18 });   // core.js dinoBox, ducking
const hits = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
const isMid = (o) => { if (o.t.name !== 'ptero') return false; const b = obstacleBox(o); return b.y + b.h <= DUCK_TOP && b.y + b.h > STAND.y; };
const kind = (o) => (o.t.name === 'ptero' ? (isMid(o) ? 'ptero-mid' : obstacleBox(o).y + obstacleBox(o).h <= STAND.y ? 'ptero-high' : 'ptero-low') : o.t.name);

// Play a seed with the solver, as survey.js does (same frame counting and last-30 encoding),
// with the extra trace the classifier needs. Stops at a crash, at capScore, or at maxFrames.
function play(seed, makeSolver, capScore, maxFrames = Infinity) {
  const g = createGame({ rng: createRng(seed), version: 'v2' });
  const sol = makeSolver();
  const capDistance = capScore / CFG.scoreCoef;
  const ring = [];
  let frames = 0, downAirborne = 0, downAirborneThisJump = false, speedDropThisJump = false;
  let lastRefused = -Infinity, lastDown = -Infinity;
  do {
    const act = sol(g);
    const d = g.dino;
    const pre = { jumping: d.jumping, ducking: d.ducking, keyJump: g.keys.jump };
    if (g.state === 'running') {
      frames++;
      ring.push({ y: +d.y.toFixed(2), vy: +d.vy.toFixed(3), jumping: d.jumping, ducking: d.ducking, keyJump: g.keys.jump,
        keyDown: g.keys.down, act: act.jump ? 'jump' : act.down ? 'down' : 'none', speed: +g.speed.toFixed(3) });
      if (ring.length > 30) ring.shift();
      if (act.down) lastDown = frames;
      if (act.down && pre.jumping) { downAirborne++; downAirborneThisJump = true; }
    }
    const wasRunning = g.state === 'running';
    g.step(act);
    if (!wasRunning) continue;
    if (!pre.jumping && d.jumping) { downAirborneThisJump = false; speedDropThisJump = false; }   // a new takeoff
    if (d.speedDrop) speedDropThisJump = true;
    // A jump press (key transition) on a grounded frame that did not take off was refused.
    if (act.jump && !pre.keyJump && !pre.jumping && !d.jumping) lastRefused = frames;
  } while (g.state === 'running' && g.distance < capDistance && frames < maxFrames);
  return { g, sol, frames, ring, downAirborne, downAirborneThisJump, speedDropThisJump, lastRefused, lastDown };
}

function classify(rec, cap = 1000000) {
  const p = play(rec.seed, make, cap, rec.frames != null ? rec.frames + 1 : Infinity);
  const { g } = p;
  const out = { seed: rec.seed, score: g.score(), frames: p.frames, crashed: g.state === 'over' };
  out.replayMatches = out.crashed && (rec.score == null || rec.score === out.score) && (rec.frames == null || rec.frames === out.frames) &&
    (rec.last30 == null || JSON.stringify(rec.last30) === JSON.stringify(p.ring));
  if (!out.crashed) { out.class = 'not-reproduced'; return out; }
  // The dino at the collision: gameOver() clears ducking (and resets y) afterwards, but the key
  // state and `jumping` are untouched; grounded with DOWN held means it was ducking.
  const d = g.dino;
  const ducking = !d.jumping && g.keys.down;
  const box = ducking ? duckBox(d) : standBox(d.jumping ? d.y : STAND_TOP);
  const killers = g.obstacles.filter(o => hits(box, obstacleBox(o)));
  out.dino = { ducking, jumping: d.jumping, y: +d.y.toFixed(2), keyDown: g.keys.down, keyJump: g.keys.jump };
  out.killers = killers.map(o => ({ kind: kind(o), size: o.size, x: +o.x.toFixed(2), y: o.y, speedOffset: o.speedOffset }));
  out.downAirborneInGame = p.downAirborne;
  out.solverStats = p.sol.stats ?? null;
  const recent = (f) => p.frames - f <= 30;
  if (d.jumping && (p.downAirborneThisJump || p.speedDropThisJump)) {
    out.class = 'failure2'; out.subtype = 'down-while-airborne';
  } else if (!d.jumping && recent(p.lastRefused)) {
    out.class = 'failure1'; out.subtype = 'jump-refused-while-ducking';
  } else if (ducking) {
    out.class = 'failure1';
    const standHit = killers.some(o => hits(standBox(STAND_TOP), obstacleBox(o)));
    out.subtype = standHit ? 'held-into-obstacle' : 'duck-box-wider';
  } else if (!d.jumping && killers.some(isMid)) {
    out.class = 'other'; out.subtype = 'duck-window';
    out.framesSinceDown = p.frames - p.lastDown;
  } else if (d.jumping) {
    out.class = 'other'; out.subtype = 'jump-plan';
  } else if (killers.length) {
    out.class = 'other'; out.subtype = 'missed-takeoff';
  } else {
    out.class = 'other'; out.subtype = 'unknown';
  }
  // Control: teacher-jump on the same seed past the crash frame.
  const c = play(rec.seed, controlSolver, Infinity, p.frames + 100);
  out.controlSurvives = c.g.state === 'running';
  return out;
}

function findCrashes(n, base, cap) {
  const recs = [];
  for (let s = base; s < base + n; s++) {
    const p = play(s, make, cap);
    if (p.g.state === 'over') recs.push({ seed: s, crashed: true, score: p.g.score(), frames: p.frames, last30: p.ring });
  }
  return recs;
}

function main() {
  const json = argv.includes('--json');
  let recs;
  if (argv.includes('--find')) {
    const n = Number(arg('--find')), base = Number(arg('--base', 40100000)), cap = Number(arg('--cap', 20000));
    if (base < 40005000 && base + n > 40000000) throw new Error('--base overlaps the survey seeds 40,000,000-40,004,999');
    recs = findCrashes(n, base, cap);
    console.log(`played seeds ${base}..${base + n - 1} to ${cap}: ${recs.length} crashes`);
  } else {
    const VALUED = ['--solver', '--solver-module', '--cap', '--base', '--find'];
    const file = argv.find((a, i) => !a.startsWith('--') && !VALUED.includes(argv[i - 1]));
    if (!file) { console.error('usage: crash-classify.js <survey.jsonl> | --find N [--base B] [--cap C]'); process.exit(2); }
    recs = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => r.crashed);
    console.log(`${file}: ${recs.length} crashed records`);
  }
  const counts = {};
  for (const rec of recs) {
    const r = classify(rec, Number(arg('--cap', argv.includes('--find') ? 20000 : 1000000)));
    const key = `${r.class}${r.subtype ? ':' + r.subtype : ''}`;
    counts[key] = (counts[key] || 0) + 1;
    if (json) console.log(JSON.stringify(r));
    else console.log(`seed ${r.seed} score ${r.score}: ${key}; killers ${r.killers ? r.killers.map(k => k.kind + (k.size > 1 ? 'x' + k.size : '')).join(',') : '-'}; ` +
      `replay ${r.replayMatches ? 'matches' : 'DIFFERS'}; control ${r.controlSurvives == null ? '-' : r.controlSurvives ? 'survives' : 'ALSO CRASHES'}`);
  }
  console.log('summary', JSON.stringify(counts));
}

if (require.main === module) main();
module.exports = { classify, play, findCrashes };
