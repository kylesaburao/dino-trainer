'use strict';
// Release-frame descriptive stats (prereg step2 §5.1 step 6): rows with ducking flag (f[3]=1)
// and teacher label none (release). Reports slot-1 present share and slot-1 x feature quantiles,
// plus slot-0 x feature quantiles. Slot k base = 5 + 13k (present b, x b+1). Usage: node release.js <demos dir>
const fs = require('fs'), path = require('path');
const dir = process.argv[2];
const M = JSON.parse(fs.readFileSync(path.join(dir, 'demos.json'), 'utf8'));
const buf = fs.readFileSync(path.join(dir, M.file));
const F = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const R = M.rowFloats, D = M.obsDim;
const x1 = [], x0 = []; let rel = 0, pres1 = 0, duckRows = 0;
for (let i = 0; i < M.rows; i++) {
  const o = i * R;
  if (F[o + 3] !== 1) continue;
  const lab = F[o + D];
  if (lab === 2) { duckRows++; continue; }
  if (lab !== 0) continue;
  rel++;
  x0.push(F[o + 6]);
  if (F[o + 18] === 1) { pres1++; x1.push(F[o + 19]); }
}
const q = (a, p) => { const s = a.slice().sort((u, v) => u - v); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null; };
const qs = a => ({ n: a.length, min: q(a, 0), p01: q(a, 0.01), p10: q(a, 0.1), p50: q(a, 0.5), p90: q(a, 0.9), max: q(a, 1) });
console.log(JSON.stringify({ demos: dir, ducking_label2: duckRows, releaseRows: rel, slot1Present: pres1,
  slot1x: qs(x1), slot1xBelow0_03: x1.filter(v => v < 0.03).length, slot1xBelow0_10: x1.filter(v => v < 0.10).length, slot0x: qs(x0) }));
