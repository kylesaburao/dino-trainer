'use strict';
// Classify survey runs for research/prereg-es-curriculum.md §10.1: stuck or not, and type T or M.
// Usage: node tools/classify-stuck.js <run dir>... [--json]
//   stuck: escaped (10 consecutive gens with fitMean > 200), mean fitMean of the last 50 gens
//          < 1100, fitMean never > 2000, last validation median < 2000
//   type T: >= 60% of tools/wall-diag.js deaths (50 seeds from 900000, final checkpoint) are
//          ptero-low with the dino on the ground, AND mean tiedPairs/64 over the last 100 gens
//          >= 0.5; one of the two alone is "T-partial" (counted as M); otherwise M
//   also: the share of those from-0 deaths below 800 points (--json adds every death score)
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const args = process.argv.slice(2);
const dirs = args.filter(a => !a.startsWith('--'));
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

const out = dirs.map((dir) => {
  const rows = fs.readFileSync(path.join(dir, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const last = rows[rows.length - 1];
  const ei = rows.findIndex((r, i) => i + 10 <= rows.length && rows.slice(i, i + 10).every(x => x.fitMean > 200));
  const last50 = mean(rows.slice(-50).map(r => r.fitMean));
  const maxFit = Math.max(...rows.map(r => r.fitMean));
  const vals = rows.filter(r => r.valMedian != null);
  const lastVal = vals.length ? vals[vals.length - 1].valMedian : null;
  const tied100 = mean(rows.slice(-100).map(r => r.tiedPairs)) / 64;
  const stuck = ei >= 0 && last50 < 1100 && maxFit <= 2000 && (lastVal == null || lastVal < 2000);
  const r = {
    run: path.basename(dir), lastGen: last.generation, escape: ei >= 0 ? rows[ei].generation : null,
    last50: Math.round(last50), maxFit: Math.round(maxFit), lastVal, tied100: +tied100.toFixed(2), stuck,
  };
  if (stuck) {
    const eps = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'wall-diag.js'), path.join(dir, 'checkpoint.json'), '--json'], { maxBuffer: 1 << 26 }));
    const lowGround = eps.filter(e => e.killer === 'ptero-low' && e.state === 'ground').length / eps.length;
    const kill = {};
    for (const e of eps) kill[e.killer] = (kill[e.killer] || 0) + 1;
    const c1 = lowGround >= 0.6, c2 = tied100 >= 0.5;
    Object.assign(r, { lowGround: +lowGround.toFixed(2), killers: kill, type: c1 && c2 ? 'T' : c1 || c2 ? 'T-partial' : 'M',
      medianScore: eps.map(e => e.score).sort((a, b) => a - b)[eps.length >> 1],
      below800: +(eps.filter(e => e.score < 800).length / eps.length).toFixed(2), deathScores: eps.map(e => e.score) });
  }
  return r;
});

if (args.includes('--json')) { console.log(JSON.stringify(out, null, 1)); process.exit(0); }
console.log('| Run | Last gen | Escape | fitMean last 50 | Max fitMean | Last val | tied/64 last 100 | Stuck | Type | low-ptero ground deaths | Median diag score | Deaths < 800 | Top killers |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|');
for (const r of out) {
  const top = r.killers ? Object.entries(r.killers).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, n]) => `${k} ${n}`).join(', ') : '';
  console.log(`| ${r.run} | ${r.lastGen} | ${r.escape ?? '-'} | ${r.last50} | ${r.maxFit} | ${r.lastVal ?? '-'} | ${r.tied100} | ${r.stuck ? 'yes' : 'no'} | ${r.type ?? '-'} | ${r.lowGround ?? '-'} | ${r.medianScore ?? '-'} | ${r.below800 ?? '-'} | ${top} |`);
}
