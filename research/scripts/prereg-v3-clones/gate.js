'use strict';
// §4.1 demos gate (research/prereg-v3-clones.md). Reads a bc-gen demos dir (demos.json +
// demos.bin; rows are written per seed, per level, in the manifest's episode order) and checks:
//   gameVersion v3, teacherName duck, coreHash prefix 65f3546c, 0 solver crashes, and that every
//   episode at level 98000 contains a row with x4 > 4.9 (the step to speed >= 41 was crossed).
// Records rows per level, and label counts (none / jump / duck; rows and weighted) per regime
// x4 <= 2.5 and x4 > 2.5. Prints a JSON summary; exit 1 if any check fails.
// Usage: node gate.js <demos dir> [--out f.json]
const fs = require('fs');
const path = require('path');
const dir = path.resolve(process.argv[2]);
const outI = process.argv.indexOf('--out');
const out = outI >= 0 ? process.argv[outI + 1] : null;
const man = JSON.parse(fs.readFileSync(path.join(dir, 'demos.json'), 'utf8'));
const OBS = man.obsDim, ROW = man.rowFloats;
const buf = fs.readFileSync(path.join(dir, man.file));
const D = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const nRows = D.length / ROW;
const checks = {};
checks.gameVersion = man.gameVersion === 'v3';
checks.teacherName = man.teacherName === 'duck';
checks.coreHash = String(man.coreHash).startsWith('65f3546c');
checks.solverCrashes0 = man.solverCrashes === 0 && man.episodes.every(e => !e.crashed);
checks.rowsMatch = nRows === man.rows && man.episodes.reduce((a, e) => a + e.rows, 0) === man.rows;
const perLevel = {};
const regime = { le: { rows: [0, 0, 0], weighted: [0, 0, 0] }, gt: { rows: [0, 0, 0], weighted: [0, 0, 0] } };
const step = { episodes: 0, withStep: 0, missing: [] };
let k = 0;
for (const e of man.episodes) {
  const L = perLevel[e.level] || (perLevel[e.level] = { episodes: 0, rows: 0, x4min: Infinity, x4max: -Infinity });
  L.episodes++; L.rows += e.rows;
  let maxX4 = -Infinity;
  for (let r = 0; r < e.rows; r++, k++) {
    const x4 = D[k * ROW + 4], label = D[k * ROW + OBS], w = D[k * ROW + OBS + 1];
    if (x4 > maxX4) maxX4 = x4;
    if (x4 < L.x4min) L.x4min = x4;
    if (x4 > L.x4max) L.x4max = x4;
    const g = x4 <= 2.5 ? regime.le : regime.gt;
    g.rows[label]++; g.weighted[label] += w;
  }
  if (e.level === 98000) {
    step.episodes++;
    if (maxX4 > 4.9) step.withStep++; else step.missing.push(e.seed);
  }
}
checks.stepInEvery98k = step.episodes > 0 && step.withStep === step.episodes;
const named = a => ({ none: a[0], jump: a[1], duck: a[2] });
const res = {
  dir: path.relative(process.cwd(), dir), rows: man.rows, sha256: man.sha256, coreHash: man.coreHash, gameVersion: man.gameVersion,
  teacherName: man.teacherName, solverCrashes: man.solverCrashes, episodes: man.episodes.length, levels: man.levels,
  perLevel, step, regime: { 'x4<=2.5': { rows: named(regime.le.rows), weighted: named(regime.le.weighted) },
    'x4>2.5': { rows: named(regime.gt.rows), weighted: named(regime.gt.weighted) } },
  checks, pass: Object.values(checks).every(Boolean),
};
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1) + '\n');
console.log(JSON.stringify(res, null, 1));
process.exit(res.pass ? 0 : 1);
