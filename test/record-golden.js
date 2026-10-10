'use strict';
// Records golden traces from the BASELINE game.js at commit 2da34de (not the
// refactored core). Run once: `node test/record-golden.js`. Re-run only if the
// trace format or the scripts change; never to "fix" a parity failure.
//
// The baseline script runs in a Node vm with the DOM stubbed out. The clock is
// stubbed (now = frameIndex * FRAME_MS for both update() and performance.now())
// and Math.random is replaced, after load, by the same seeded PRNG the core uses.
// Each frame applies key transitions (releases before presses, jump before down),
// then calls update(FRAME_MS, now), then snapshots the state.
const vm = require('vm');
const { execFileSync } = require('child_process');
const path = require('path');
const { createRng } = require('../core.js');
const { snapshot, writeTrace } = require('./golden-format.js');

const BASELINE = '2da34de';
const REPO = path.join(__dirname, '..');
const baselineSrc = execFileSync('git', ['show', `${BASELINE}:game.js`], { cwd: REPO, encoding: 'utf8' });

function loadBaseline() {
  const noop = () => {};
  const ctx2d = new Proxy({}, { get: (t, k) => (k in t ? t[k] : noop), set: (t, k, v) => { t[k] = v; return true; } });
  const canvas = () => ({ width: 0, height: 0, getContext: () => ctx2d });
  const clock = { now: 0 };
  const sandbox = {
    document: {
      createElement: canvas,
      getElementById: canvas,
      addEventListener: noop,
      body: { classList: { toggle: noop } },
    },
    localStorage: { getItem: () => null, setItem: noop },
    performance: { now: () => clock.now },
    requestAnimationFrame: noop,
    devicePixelRatio: 1,
  };
  sandbox.window = sandbox;
  const ctx = vm.createContext(sandbox);
  vm.runInContext(baselineSrc, ctx, { filename: 'game.js@' + BASELINE });
  // A later script in the same context sees the baseline's top-level let/const.
  vm.runInContext(`
    globalThis.__view = () => ({ state, dino, speed, distance, obstacles, rngCalls: __rngCalls() });
    globalThis.__requirement = (o) => requirement(o, speed);
    globalThis.__frame = (keys, held, now) => {
      if (held.jump && !keys.jump) { held.jump = false; releaseJump(); }
      if (held.down && !keys.down) { held.down = false; releaseDown(); }
      if (!held.jump && keys.jump) { held.jump = true; pressJump(); }
      if (!held.down && keys.down) { held.down = true; pressDown(); }
      update(FRAME_MS, now);
    };
  `, ctx);
  return { ctx, clock };
}

function record(name, seed, frames, script) {
  const { ctx, clock } = loadBaseline();
  const base = createRng(seed);
  let calls = 0;
  ctx.__rngCalls = () => calls;
  ctx.__rng = () => { calls++; return base(); };
  vm.runInContext("Math.random = __rng;", ctx);   // installed after load (ground texture)

  const held = { jump: false, down: false };
  const FRAME_MS = 1000 / 60;
  const scriptRng = createRng(seed ^ 0x5eed);
  const mem = {};
  let inputs = '';
  const rows = [];
  for (let f = 0; f < frames; f++) {
    const view = ctx.__view();
    const code = script(f, view, { rng: scriptRng, mem, requirement: ctx.__requirement });
    inputs += String(code);
    const now = f * FRAME_MS;
    clock.now = now;
    ctx.__frame({ jump: !!(code & 1), down: !!(code & 2) }, held, now);
    rows.push(snapshot(ctx.__view()));
  }
  const last = rows[rows.length - 1];
  const crashes = rows.filter((r, i) => r[0] === 2 && (i === 0 || rows[i - 1][0] !== 2)).length;
  console.log(`${name}: ${frames} frames, ${crashes} crashes, final speed ${last[6].toFixed(3)}, rng calls ${last[8]}`);
  writeTrace({ name, baseline: BASELINE, seed, inputs, rows });
}

// --- input scripts ---------------------------------------------------------
// Restart helper: after a crash, wait `offset` frames then tap jump once.
function restartAfter(offset) {
  return (f, v, m) => {
    if (v.state !== 'over') { m.overFrame = -1; return null; }
    if (m.overFrame < 0) m.overFrame = f;
    const k = f - m.overFrame;
    return k === offset ? 1 : 0;
  };
}

// Lookahead "cheat" controller using the baseline's own fairness solver. It only
// generates a fixed input script that survives long enough to cover pterodactyls,
// max speed and night mode; the stored inputs (not this controller) are replayed.
function solver(extras) {
  const restart = restartAfter(60);
  return (f, v, { mem, requirement, rng }) => {
    const r0 = restart(f, v, mem);
    if (r0 !== null) { mem.hold = 0; return r0; }
    if (v.state === 'idle') return 1;
    const d = v.dino;
    if (d.jumping) {
      if (extras && mem.fastDrop && d.vy > 0) return 2;            // fast-drop on the way down
      if (mem.hold > 0) { mem.hold--; return 1; }
      return 0;
    }
    mem.fastDrop = false;
    const ahead = [...v.obstacles].sort((a, b) => a.x - b.x);
    for (const o of ahead) {
      const r = requirement(o);
      if (!r || r.to < 0) continue;
      if (r.grounded) { if (r.from < 40) return extras && r.from < 25 ? 2 : 0; continue; }
      if (r.from <= 0.5) {
        mem.hold = r.release === Infinity ? 999 : Math.max(0, r.release - 2);
        mem.fastDrop = extras && rng() < 0.3;
        return 1;
      }
      // Free time before the next jump: sometimes duck briefly.
      if (extras && r.from > 30) {
        if (mem.duck > 0) { mem.duck--; return 2; }
        if (rng() < 0.02) { mem.duck = 5 + Math.floor(rng() * 15); return 2; }
      }
      return 0;
    }
    return 0;
  };
}

// Random key states with random hold lengths; restarts happen naturally.
function randomKeys(maxRun, weights) {
  return (f, v, { rng, mem }) => {
    if (!(mem.left > 0)) {
      const x = rng() * (weights[0] + weights[1] + weights[2] + weights[3]);
      let c = 0, acc = weights[0];
      while (x >= acc) acc += weights[++c];
      mem.code = c;
      mem.left = 1 + Math.floor(rng() * maxRun);
    }
    mem.left--;
    return mem.code;
  };
}

record('01-idle-then-tap', 101, 900, (f) => (f >= 20 && f < 26 ? 1 : 0));
record('02-held-jumps', 202, 1500, (f) => (f % 90 < 40 ? 1 : 0));
record('03-short-taps', 303, 1500, (f) => (f % 37 < 6 ? 1 : 0));
record('04-ducks', 404, 1500, (f) => (f === 0 ? 1 : f % 80 < 50 ? 2 : 0));
record('05-fast-drops', 505, 1500, (f) => { const k = f % 70; return k < 10 ? 1 : k < 25 ? (k >= 6 ? 2 : 0) : 0; });
record('06-jump-and-down', 606, 1500, (f) => { const k = f % 60; return k < 12 ? 1 : k < 20 ? 3 : k < 28 ? 2 : 0; });
for (const offset of [44, 45, 46]) {
  const restart = restartAfter(offset);
  record(`07-restart-lockout-${offset}`, 700 + offset, 2500, (f, v, { mem }) => {
    const r = restart(f, v, mem);
    return r !== null ? r : (f % 200 < 3 ? 1 : 0);
  });
}
record('08-random', 808, 4000, randomKeys(30, [5, 3, 2, 1]));
record('09-random-taps', 909, 4000, randomKeys(8, [6, 3, 1, 0.5]));
record('10-solver-long', 1010, 9000, solver(false));
record('11-solver-ducks-drops', 1111, 6000, solver(true));
