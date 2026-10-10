'use strict';
// Row and class counts of a dino-bc-demos-v1 demos.bin (research/prereg-bc-pipeline-e2e.md §3.1),
// split by weight (weight-1 rows vs down-weighted boring rows), plus the sha256 of the file.
// Usage: node democounts.js <demos.bin> [<demos.bin> ...]
const fs = require('fs');
const crypto = require('crypto');
const ROW = 46, OBS = 44;
for (const file of process.argv.slice(2)) {
  const buf = fs.readFileSync(file);
  const f = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const n = f.length / ROW;
  const byW = {};   // weight -> [none, jump, duck] row counts
  const weighted = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const lab = f[i * ROW + OBS], w = f[i * ROW + OBS + 1];
    (byW[w] ||= [0, 0, 0])[lab]++;
    weighted[lab] += w;
  }
  const sha = crypto.createHash('sha256').update(buf).digest('hex');
  console.log(JSON.stringify({ file, rows: n, sha256: sha, rowsByWeight: byW, weighted }));
}
