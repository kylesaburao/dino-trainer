'use strict';
// Probe B driver.
//   node cloneB.js gen  <seedFrom> <seedTo> <out.bin>           solver demonstrations
//   node cloneB.js roll <model.json> <seedFrom> <seedTo> <out.json> [collect.bin]
//        roll the clone (argmax MLP, same idle->START loop as policy.createAgent),
//        optionally recording visited states labelled by the solver (DAgger).
// Rows: Float32 [ROW features..., label, weight].
const R = '/Users/kyle/Desktop/dino-game/';
const Core = require(R + 'core.js');
const P = require(R + 'policy.js');
const { createSolver } = require(R + 'test/solver.js');
const fs = require('fs');
const { ROW, observeRow, VARIANTS, expert, pressTarget } = require('./featx.js');
const { createGame, createRng, CFG, W } = Core;

const LEVELS = process.env.ROLL_LEVELS ? process.env.ROLL_LEVELS.split(',').map(Number) : [0, 2000, 50000, 300000, 900000];
// gen: 3k from 0, 5k windows elsewhere. roll: ROLL_SPAN0 / ROLL_SPAN override (FULL=1: to 1,000,000).
const SPAN = (L) => (process.env.FULL ? 1000000 - L : L === 0 ? +(process.env.ROLL_SPAN0 || 3000) : +(process.env.ROLL_SPAN || 5000));
const ACT = P.ACTIONS;

function sink(path) {
  if (!path) return null;
  const fd = fs.openSync(path, 'w');
  let buf = new Float32Array(ROW + 2 * 1), n = 0;
  const chunk = new Float32Array((ROW + 2) * 4096); let c = 0;
  return {
    push(x, label, w) {
      chunk.set(x.subarray(0, ROW), c * (ROW + 2)); chunk[c * (ROW + 2) + ROW] = label; chunk[c * (ROW + 2) + ROW + 1] = w;
      if (++c === 4096) { fs.writeSync(fd, Buffer.from(chunk.buffer)); c = 0; } n++;
    },
    close() { if (c) fs.writeSync(fd, Buffer.from(chunk.buffer, 0, c * (ROW + 2) * 4)); fs.closeSync(fd); return n; },
  };
}

// Keep every "interesting" frame; keep 1 in 10 boring frames (grounded, label none,
// nearest obstacle more than 40 frames away or absent) with weight 10.
let boringCounter = 0;
function keep(x, label) {
  const boring = !x[2] && label === 0 && x[44] > 40 / 30;
  if (!boring) return 1;
  return (boringCounter++ % 10 === 0) ? 10 : 0;
}

function gen(a, b, out) {
  const s = sink(out);
  const x = new Float32Array(ROW);
  let agree = 0, total = 0, presses = 0, invisible = 0, crashes = 0;
  for (let seed = a; seed < b; seed++) for (const L of LEVELS) {
    const g = createGame({ rng: createRng(seed), version: 'v2', startScore: L || undefined });
    const sol = createSolver();
    const cap = (L + SPAN(L)) / CFG.scoreCoef;
    let k = 0;
    do {
      const act = sol(g);
      if (g.state === 'running') {
        const label = act.jump ? 1 : 0;
        if (expert(g, k) === label) agree++; total++;
        if (label === 1 && !g.dino.jumping && !g.keys.jump) {
          presses++;
          const t = pressTarget(g); if (t && !(t.x < W)) invisible++;
        }
        observeRow(g, x);
        const w = keep(x, label);
        if (w) s.push(x, label, w);
      }
      const wasJ = g.dino.jumping;
      g.step(act);
      k = g.dino.jumping ? (wasJ ? k + 1 : 1) : 0;
    } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
    if (g.state === 'over') crashes++;
  }
  const n = s.close();
  console.log(JSON.stringify({ mode: 'gen', a, b, rows: n, expertAgreement: agree / total, presses, pressTargetInvisible: invisible, solverCrashes: crashes }));
}

function loadNet(path) {
  const m = JSON.parse(fs.readFileSync(path));
  const cols = VARIANTS[m.variant];
  if (cols.length !== m.layers[0]) throw new Error('variant/layers mismatch');
  return { decide: P.createNet(m.layers, Float64Array.from(m.params)), cols, m };
}

function roll(modelPath, a, b, out, collect) {
  const { decide, cols } = loadNet(modelPath);
  const s = sink(collect);
  const x = new Float32Array(ROW), inp = new Float64Array(cols.length);
  const res = {};
  for (const L of LEVELS) {
    const scores = [];
    for (let seed = a; seed < b; seed++) {
      const g = createGame({ rng: createRng(seed), version: 'v2', startScore: L || undefined });
      const cap = (L + SPAN(L)) / CFG.scoreCoef;
      let k = 0;
      do {
        let act;
        if (g.state === 'idle') act = ACT[1];
        else {
          observeRow(g, x);
          for (let i = 0; i < cols.length; i++) inp[i] = x[cols[i]];
          act = ACT[decide(inp)];
          if (s) { const lab = expert(g, k); const w = keep(x, lab); if (w) s.push(x, lab, w); }
        }
        const wasJ = g.dino.jumping;
        g.step(act);
        k = g.dino.jumping ? (wasJ ? k + 1 : 1) : 0;
      } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
      scores.push(Math.min(g.score(), L + SPAN(L)) - L);
    }
    scores.sort((p, q) => p - q);
    res[L] = { n: scores.length, survive: scores.filter(v => v >= SPAN(L)).length / scores.length,
      crashScores: scores.filter(v => v < SPAN(L)).map(v => v + L).slice(0, 40), median: scores[scores.length >> 1], p10: scores[Math.floor(scores.length * 0.1)], mean: scores.reduce((p, q) => p + q, 0) / scores.length };
  }
  if (s) s.close();
  fs.writeFileSync(out, JSON.stringify(res));
  console.log(modelPath, JSON.stringify(res));
}

const [mode, ...args] = process.argv.slice(2);
if (mode === 'gen') gen(+args[0], +args[1], args[2]);
else if (mode === 'roll') roll(args[0], +args[1], +args[2], args[3], args[4]);
