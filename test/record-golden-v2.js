'use strict';
// Records the rules-v2 golden traces (test/golden-v2/) from core.js and test/solver.js
// AS OF commit 10cd0e0 (the last commit before rules v3 was added), loaded with
// `git show`, so re-running it later still records the pre-v3 behaviour.
// Run once: `node test/record-golden-v2.js`. Re-run only if the trace format or the
// scripts change; never to "fix" a parity failure (test/v2-parity.test.js).
//
// Rules v1 has its own goldens (test/golden/, recorded from the original game.js by
// test/record-golden.js). Rules v2 had none before rules v3; these cover v2 from 0
// points (the 1k onset), at 30k (cactus groups of 4 from 31.6k), through 150k, and
// across 1M (difficulty capped at 1), with the jump solver, the duck solver and
// random keys (crashes, restarts, fast-drops). Each trace stores its version and
// startScore; the replay builds the same game and feeds the stored inputs.
const vm = require('vm');
const path = require('path');
const { execFileSync } = require('child_process');
const { snapshot, writeTrace, GOLDEN_V2_DIR } = require('./golden-format.js');

const BASELINE = '10cd0e0';
const REPO = path.join(__dirname, '..');
const show = (file) => execFileSync('git', ['show', `${BASELINE}:${file}`], { cwd: REPO, encoding: 'utf8' });

// CommonJS-load a source string with a custom require.
function load(src, filename, req) {
  const module = { exports: {} };
  const fn = vm.compileFunction(src, ['require', 'module', 'exports', '__filename', '__dirname'], { filename });
  fn(req, module, module.exports, filename, path.dirname(filename));
  return module.exports;
}
const Core = load(show('core.js'), `core.js@${BASELINE}`, require);
const Solver = load(show('test/solver.js'), `test/solver.js@${BASELINE}`,
  (m) => (m === '../core.js' ? Core : require(m)));

function record(name, seed, startScore, frames, script) {
  const base = Core.createRng(seed);
  let calls = 0;
  const game = Core.createGame({ rng: () => { calls++; return base(); }, version: 'v2', startScore });
  const view = { get state() { return game.state; }, dino: game.dino, get speed() { return game.speed; },
    get distance() { return game.distance; }, get obstacles() { return game.obstacles; }, get rngCalls() { return calls; } };
  const scriptRng = Core.createRng(seed ^ 0x5eed);
  const mem = {};
  let inputs = '';
  const rows = [];
  for (let f = 0; f < frames; f++) {
    const k = script(game, f, { rng: scriptRng, mem });
    const code = (k.jump ? 1 : 0) | (k.down ? 2 : 0);
    inputs += String(code);
    game.step({ jump: !!(code & 1), down: !!(code & 2) });
    rows.push(snapshot(view));
  }
  const last = rows[rows.length - 1];
  const crashes = rows.filter((r, i) => r[0] === 2 && (i === 0 || rows[i - 1][0] !== 2)).length;
  console.log(`${name}: ${frames} frames, start ${startScore}, final score ${Math.floor(last[7] * Core.CFG.scoreCoef)}, ` +
    `${crashes} crashes, final speed ${last[6].toFixed(3)}, rng calls ${last[8]}`);
  writeTrace({ name, baseline: BASELINE, version: 'v2', startScore, seed, inputs, rows }, GOLDEN_V2_DIR);
}

// A solver from the pinned commit (one per trace: the duck solver keeps state).
const solver = (create) => { let act = null; return (g) => (act || (act = create()))(g); };

// Random key states with random hold lengths; after a crash, restart once the lockout ends.
function randomKeys(maxRun, weights) {
  return (g, f, { rng, mem }) => {
    if (g.state === 'over') { mem.over = (mem.over || 0) + 1; return { jump: mem.over > 50 && mem.over % 2 === 0 }; }
    mem.over = 0;
    if (!(mem.left > 0)) {
      const x = rng() * (weights[0] + weights[1] + weights[2] + weights[3]);
      let c = 0, acc = weights[0];
      while (x >= acc) acc += weights[++c];
      mem.code = c;
      mem.left = 1 + Math.floor(rng() * maxRun);
    }
    mem.left--;
    return { jump: !!(mem.code & 1), down: !!(mem.code & 2) };
  };
}

record('v2-01-jump-solver-from-0', 2101, 0, 7000, solver(Solver.createSolver));
record('v2-02-duck-solver-30k', 2202, 30000, 6000, solver(Solver.createDuckSolver));
record('v2-03-jump-solver-148k', 2303, 148000, 6000, solver(Solver.createSolver));
record('v2-04-duck-solver-148k', 2404, 148500, 5000, solver(Solver.createDuckSolver));
record('v2-05-jump-solver-998k', 2505, 998000, 5000, solver(Solver.createSolver));
record('v2-06-random-5k', 2606, 5000, 4000, randomKeys(25, [5, 3, 2, 1]));
record('v2-07-random-200k', 2707, 200000, 3000, randomKeys(12, [6, 3, 2, 0.5]));
