'use strict';
// Premise probe for research/prereg-teacher-duck-step2.md §3 (seconds-scale, no files written
// unless --out): plays test/solver.js createDuckSolver over the bc-gen episode set (default
// seeds 1-40, levels 0,2000,50000,300000,900000, span0 3000, span 5000, keep-boring 10, rules
// v2) with bc-gen's row rule (trainer/bc.js genEpisode + isBoring, re-implemented here because
// bc.js is hard-wired to createSolver on this commit), labelling DOWN as class 2. Reports row
// and weighted class counts, the solver's stats (mid pterodactyls ducked / fallback by reason,
// DOWN frames, conflicts) and crashes. It does NOT produce training data: the demos for step 2
// come from `bc-gen --teacher duck` once that is merged.
// --write-demos <dir> also writes the rows in bc-gen's layout (demos.bin + a minimal demos.json,
// teacherName 'teacher-duck-premise') so conflicts.js can read them. Scratch use only.
// Usage: node duckdemo-premise.js [--seeds 1-40] [--out f.json] [--write-demos <dir>]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const Core = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));
const { createDuckSolver } = require(path.join(ROOT, 'test', 'solver.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const [s0, s1] = opt('--seeds', '1-40').split('-').map(Number);
const LEVELS = [0, 2000, 50000, 300000, 900000], SPAN0 = 3000, SPAN = 5000, KEEP = 10;
const OBS_DIM = P.OBS_DIM;

// trainer/bc.js isBoring (same Float32 arithmetic).
function isBoring(x, label) {
  if (x[2] || label !== 0) return false;
  const tau0 = Math.fround(x[5] ? Math.min(80, (x[6] * Core.W) / (x[17] * 13)) / 30 : 80 / 30);
  return tau0 > 40 / 30;
}

const rows = [0, 0, 0], weighted = [0, 0, 0], boringRows = [0, 0, 0];
const stats = { midEncountered: 0, midDucked: 0, fallback: 0, fallbackWhy: { takeoff: 0, obstacle: 0, newcomer: 0 }, downFrames: 0, conflicts: 0 };
const byLevel = {};
let crashes = 0, episodes = 0;
const crashList = [];
const writeDir = opt('--write-demos', null);
const ROW = OBS_DIM + 2;
let buf = new Float32Array(ROW * (1 << 20)), nrow = 0;
function store(x, label, w) {
  if (!writeDir) return;
  if ((nrow + 1) * ROW > buf.length) { const b = new Float32Array(buf.length * 2); b.set(buf); buf = b; }
  buf.set(x, nrow * ROW); buf[nrow * ROW + OBS_DIM] = label; buf[nrow * ROW + OBS_DIM + 1] = w; nrow++;
}
const t0 = Date.now();
for (let seed = s0; seed <= s1; seed++) {
  for (const level of LEVELS) {
    const span = level ? SPAN : SPAN0;
    const g = Core.createGame({ rng: Core.createRng(seed), version: 'v2', startScore: level || undefined });
    const sol = createDuckSolver();
    const cap = (level + span) / Core.CFG.scoreCoef;
    const x = new Float32Array(OBS_DIM);
    let boring = 0;
    const L = byLevel[level] || (byLevel[level] = { rows: [0, 0, 0], midEncountered: 0, midDucked: 0, fallback: 0 });
    do {
      const act = sol(g);
      if (g.state === 'running') {
        const label = act.down ? 2 : (act.jump ? 1 : 0);
        P.observe(g, x);
        if (!isBoring(x, label)) { rows[label]++; weighted[label] += 1; L.rows[label]++; store(x, label, 1); }
        else if (boring++ % KEEP === 0) { rows[label]++; weighted[label] += KEEP; boringRows[label]++; L.rows[label]++; store(x, label, KEEP); }
      }
      g.step(act);
    } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
    episodes++;
    if (g.state === 'over') { crashes++; crashList.push({ seed, level, score: g.score ? g.score() : null }); }
    const st = sol.stats;
    for (const k of ['midEncountered', 'midDucked', 'fallback', 'downFrames', 'conflicts']) stats[k] += st[k];
    for (const k of Object.keys(stats.fallbackWhy)) stats.fallbackWhy[k] += st.fallbackWhy[k];
    L.midEncountered += st.midEncountered; L.midDucked += st.midDucked; L.fallback += st.fallback;
  }
}
const totalRows = rows[0] + rows[1] + rows[2], totalW = weighted[0] + weighted[1] + weighted[2];
const report = {
  seeds: [s0, s1], levels: LEVELS, episodes, crashes, crashList,
  rows: { none: rows[0], jump: rows[1], duck: rows[2], total: totalRows },
  boringRows: { none: boringRows[0], jump: boringRows[1], duck: boringRows[2] },
  weighted: { none: weighted[0], jump: weighted[1], duck: weighted[2], total: totalW },
  share: { jump: weighted[1] / totalW, duck: weighted[2] / totalW, duckOverJump: weighted[2] / weighted[1] },
  // trainer/bc.js classWeights: sqrt(total / (2 count_c)) over weighted counts.
  classWeight: weighted.map(c => (c > 0 ? Math.sqrt(totalW / (2 * c)) : 0)),
  solver: stats, fallbackShare: stats.midEncountered ? stats.fallback / stats.midEncountered : null, byLevel,
  seconds: (Date.now() - t0) / 1000,
};
console.log(JSON.stringify(report, null, 1));
if (writeDir) {
  fs.mkdirSync(writeDir, { recursive: true });
  const bytes = Buffer.from(buf.buffer, 0, nrow * ROW * 4);
  fs.writeFileSync(path.join(writeDir, 'demos.bin'), bytes);
  const sha256 = require('crypto').createHash('sha256').update(bytes).digest('hex');
  fs.writeFileSync(path.join(writeDir, 'demos.json'), JSON.stringify({ format: 'dino-bc-demos-v1', file: 'demos.bin', obsDim: OBS_DIM, rowFloats: ROW,
    teacher: 'test/solver.js createDuckSolver', teacherName: 'teacher-duck-premise', rows: nrow, sha256, note: 'premise probe output, not bc-gen' }, null, 1) + '\n');
  console.error(`wrote ${nrow} rows to ${writeDir} (sha ${sha256.slice(0, 16)})`);
}
const out = opt('--out', null);
if (out) fs.writeFileSync(out, JSON.stringify(report, null, 1) + '\n');
