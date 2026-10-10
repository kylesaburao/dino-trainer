'use strict';
// Spot-check of agreement.py (numpy, float64) against policy.js createNet argmax on the
// same held-out rows: for each run dir, the final snapshot's predictions on every
// STRIDE-th held-out row (stored in agreement.json) are recomputed with policy.js.
// Usage: node spot-check.js <heldout demos.bin> <run dir> [...]   -> one JSON line per run
const fs = require('fs');
const path = require('path');
const P = require('/Users/kyle/Desktop/dino-game/policy.js');
const OBS = 44, ROW = 46;
const buf = fs.readFileSync(path.resolve(process.argv[2]));
const f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
for (const run of process.argv.slice(3)) {
  const a = JSON.parse(fs.readFileSync(path.join(run, 'agreement.json'), 'utf8'));
  const ep = String(a.spot.epoch).padStart(2, '0');
  const s = JSON.parse(fs.readFileSync(path.join(run, 'snapshots', `epoch-${ep}.json`), 'utf8'));
  const net = P.createNet(s.layers, s.params);
  let diff = 0;
  const x = new Float64Array(OBS);
  a.spot.idx.forEach((i, k) => {
    for (let j = 0; j < OBS; j++) x[j] = f[i * ROW + j];
    if (net(x) !== a.spot.pred[k]) diff++;
  });
  console.log(JSON.stringify({ run: a.run, epoch: a.spot.epoch, rows: a.spot.idx.length, mismatches: diff }));
}
