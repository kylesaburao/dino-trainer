'use strict';
// Opening scan and release lag (research/prereg-teacher-duck-stuck.md §4.4).
// Plays rules-v2 games from 0 to --target (default 2000 points) on seeds BASE..BASE+N-1
// (or, with --val, on trainer/env.js validationSeeds(25), default target 1,000,000) with a
// model (policy.js createAgent, JS forward pass) or the duck teacher (test/solver.js
// createDuckSolver), single-threaded.
// A DOWN run = consecutive running frames on which the action holds DOWN while the dino is
// not jumping (the pre-step state, as duckrun.js). A run is "mid" if, in the state after one of
// its steps, a duckable obstacle (box overlaps the standing box vertically, clears the ducking
// box; as duckshare.js) overlaps the standing box's x-range [56, 86). Release lag = the DOWN
// frames of the run after its last such overlap frame. A run that ends with the game over is
// fatal (lag right-censored). Runs that are not mid are "no-mid". A run still open when the
// game reaches the target is "truncated" and not counted (found by the teacher control: 2 such
// one-frame runs started at 1,999 points).
// Failing games record score, killer (as duckrun.js), and the final DOWN run (the run active at
// the crash: length, mid, lag). Stuck criterion (prereg §4.6): final run mid and length >= 20.
// Usage: node opening.js (--model <best.json> | --teacher duck) [--base 30003000] [--n 1000]
//          [--target 2000] [--val] [--out f.json]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const Core = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));
const { createDuckSolver, duckBox } = require(path.join(ROOT, 'test', 'solver.js'));
const { validationSeeds } = require(path.join(ROOT, 'trainer', 'env.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const modelFile = opt('--model', null);
const teacher = opt('--teacher', null);
if (!!modelFile === !!teacher) throw new Error('give exactly one of --model <best.json> or --teacher duck');
if (teacher && teacher !== 'duck') throw new Error(`--teacher must be duck, got ${teacher}`);
const VAL = args.includes('--val');
const BASE = Number(opt('--base', 30003000)), N = Number(opt('--n', 1000));
const TARGET = Number(opt('--target', VAL ? 1000000 : 2000));
const seeds = VAL ? validationSeeds(25) : Array.from({ length: N }, (_, i) => BASE + i);
const m = modelFile ? JSON.parse(fs.readFileSync(path.resolve(modelFile), 'utf8')) : null;
if (m && m.gameVersion !== 'v2') throw new Error(`model gameVersion ${m.gameVersion}`);
const cap = TARGET / Core.CFG.scoreCoef;
const stand = Core.standBox(Core.STAND_TOP);
const STUCK_LEN = 20;
const PC = { 108: 'low', 82: 'mid', 58: 'high' };
const name = o => o.t.name === 'ptero' ? `ptero-${PC[o.y]}` : `cactus-${o.t.name}${o.size > 1 ? 'x' + o.size : ''}`;

function playSeed(seed, lags, agg) {
  const g = Core.createGame({ rng: Core.createRng(seed), version: 'v2' });
  const actor = teacher ? createDuckSolver() : null;
  const agent = m ? P.createAgent(m) : null;
  const duckY = duckBox(g.dino).y;
  let run = null;
  const closeRun = (fatal) => {
    if (!run) return;
    if (run.mid) {
      const lag = run.len - run.lastOverlap;
      if (fatal) agg.fatalMid++; else { lags.push(lag); agg.midRuns++; }
    } else if (fatal) agg.fatalNoMid++; else agg.noMidRuns++;
  };
  do {
    const running = g.state === 'running';
    const k = actor ? actor(g) : agent.act(g);
    const downRun = running && k.down && !g.dino.jumping;
    if (downRun) { if (!run) run = { len: 0, mid: false, lastOverlap: 0, startScore: g.score() }; run.len++; }
    else if (run) { closeRun(false); run = null; }
    g.step(k);
    if (run && g.state === 'running' || run && g.state === 'over') {
      for (const o of g.obstacles) {
        if (o.t.name !== 'ptero') continue;
        const b = Core.obstacleBox(o);
        if (!(b.y < stand.y + stand.h && b.y + b.h > stand.y && b.y + b.h <= duckY)) continue;
        if (b.x < stand.x + stand.w && b.x + b.w > stand.x) { run.mid = true; run.lastOverlap = run.len; }
      }
    }
  } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
  const crashed = g.state === 'over';
  const res = { seed, score: Math.min(g.score(), TARGET), pass: !crashed };
  if (crashed) {
    const hit = g.obstacles.filter(o => { const b = Core.obstacleBox(o); return b.x < 104 && b.x + b.w > 54; }).sort((p, q) => p.x - q.x)[0];
    res.killer = hit ? name(hit) : null;
    res.finalDuck = run ? { len: run.len, mid: run.mid, lag: run.len - run.lastOverlap, startScore: run.startScore } : null;
    res.stuck = !!(run && run.mid && run.len >= STUCK_LEN);
  }
  if (crashed) closeRun(true); else if (run) agg.truncated++;   // a run still open at the target is not counted
  return res;
}

const t0 = Date.now();
const lags = [], agg = { midRuns: 0, noMidRuns: 0, fatalMid: 0, fatalNoMid: 0, truncated: 0 };
const fails = [];
let passed = 0;
for (const s of seeds) {
  const r = playSeed(s, lags, agg);
  if (r.pass) passed++; else fails.push(r);
}
lags.sort((a, b) => a - b);
const q = (p) => lags.length ? lags[Math.min(lags.length - 1, Math.floor(p * lags.length))] : null;
const report = {
  model: modelFile ? path.relative(ROOT, path.resolve(modelFile)) : null, teacher, layers: m ? m.layers : null,
  seeds: VAL ? 'validationSeeds(25)' : { base: BASE, n: N }, target: TARGET,
  games: seeds.length, passed, failed: fails.length, stuck: fails.filter(f => f.stuck).length,
  lag: { n: lags.length, p50: q(0.5), p99: q(0.99), p999: q(0.999), max: lags.length ? lags[lags.length - 1] : null,
    ge5: lags.filter(x => x >= 5).length, ge10: lags.filter(x => x >= 10).length, ge20: lags.filter(x => x >= 20).length },
  runs: agg, fails, seconds: (Date.now() - t0) / 1000,
};
const out = opt('--out', null);
if (out) fs.writeFileSync(out, JSON.stringify(report) + '\n');
console.log(JSON.stringify({ ...report, fails: report.fails.length > 8 ? `${report.fails.length} (see --out)` : report.fails }));
