'use strict';
// Usage: node tools/wall-table.js <run dir>...
// Second-wall table: per run, sustained escape (first of 10 consecutive gens with fitMean > 200),
// first gen with fitMean > 2000, gens and wall between them, thetaNorm and tiedPairs/64 over the stretch.
const fs = require('fs'), path = require('path');
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
console.log('| Seed | Escape (sustained) | First > 2000 | Gens 200→2000 | Wall 200→2000 s | ‖θ‖ at escape → end of stretch | tiedPairs/64 mean over stretch | Last gen | fitMean end | Peak fitMean (gen) |');
console.log('|---|---|---|---|---|---|---|---|---|---|');
for (const dir of process.argv.slice(2)) {
  const rows = fs.readFileSync(path.join(dir, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const ei = rows.findIndex((r, i) => i + 10 <= rows.length && rows.slice(i, i + 10).every(x => x.fitMean > 200));
  const ki = rows.findIndex(r => r.fitMean > 2000);
  const last = rows[rows.length - 1];
  const end = ki >= 0 ? ki : rows.length - 1;
  const st = ei >= 0 ? rows.slice(ei, end + 1) : [];
  const peak = rows.reduce((b, r) => (r.fitMean > b.fitMean ? r : b));
  const wall = ei >= 0 ? (rows[end].cumulativeMs - rows[ei].cumulativeMs) / 1000 : null;
  console.log(`| ${path.basename(dir).split('-s').pop()} | ${ei >= 0 ? rows[ei].generation : 'none'} | ${ki >= 0 ? rows[ki].generation : '>' + last.generation} | ${ei >= 0 ? (ki >= 0 ? '' : '>') + (rows[end].generation - rows[ei].generation) : '-'} | ${wall == null ? '-' : (ki >= 0 ? '' : '>') + wall.toFixed(0)} | ${ei >= 0 ? rows[ei].thetaNorm.toFixed(1) + ' → ' + rows[end].thetaNorm.toFixed(1) : '-'} | ${st.length ? mean(st.map(r => r.tiedPairs / 64)).toFixed(2) : '-'} | ${last.generation} | ${last.fitMean.toFixed(0)} | ${peak.fitMean.toFixed(0)} (${peak.generation}) |`);
}
