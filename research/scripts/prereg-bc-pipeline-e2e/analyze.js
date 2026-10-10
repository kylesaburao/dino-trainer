'use strict';
// Statistics for research/prereg-bc-pipeline-e2e.md: per-clone scan failure counts with
// two-sided and one-sided 95% Clopper-Pearson bounds (cp/oneSidedUpper copied from
// ../prereg-bc-robustness/analyze.js), seed-range and frame checks, and the Python
// backend's per-epoch loss against the prototype log for m_base_s1.
// Usage: node analyze.js <e2e run dir> <prototype log_base_s1.txt>
//   (reads <dir>/py-s1.jsonl, <dir>/node-s1.jsonl, <dir>/{py-s1,node-s1}/{best.json,bc-metrics.jsonl})
const fs = require('fs');
const path = require('path');
const dir = path.resolve(process.argv[2]);
const protoLog = path.resolve(process.argv[3]);

function lgamma(x) { // Lanczos
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const logC = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
const pmf = (n, k, p) => (p === 0 ? (k === 0 ? 1 : 0) : p === 1 ? (k === n ? 1 : 0) : Math.exp(logC(n, k) + k * Math.log(p) + (n - k) * Math.log1p(-p)));
const cdf = (n, k, p) => { let s = 0; for (let i = 0; i <= k; i++) s += pmf(n, i, p); return Math.min(1, s); };
const bisect = (f, lo = 0, hi = 1) => { for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (f(m)) hi = m; else lo = m; } return (lo + hi) / 2; };
function cp(k, n) {
  const lower = k === 0 ? 0 : bisect(p => 1 - cdf(n, k - 1, p) >= 0.025);
  const upper = k === n ? 1 : bisect(p => cdf(n, k, p) <= 0.025);
  return { lower, upper };
}
const oneSidedUpper = (k, n) => bisect(p => cdf(n, k, p) <= 0.05);
const fmt = (x) => (x * 100).toFixed(3) + '%';
const jsonl = (f) => fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));

for (const arm of ['py-s1', 'node-s1']) {
  const ls = jsonl(path.join(dir, `${arm}.jsonl`));
  const header = ls.find(l => l.header), footer = ls.find(l => l.footer), rows = ls.filter(l => l.seed !== undefined);
  const seeds = rows.map(r => r.seed).sort((a, b) => a - b);
  const contiguous = seeds.every((s, i) => s === header.base + i);
  const fails = rows.filter(r => !r.pass);
  const frames = [...new Set(rows.filter(r => r.pass).map(r => r.frames))];
  const ci = cp(fails.length, rows.length);
  const best = JSON.parse(fs.readFileSync(path.join(dir, arm, 'best.json'), 'utf8'));
  console.log(`${arm}: model ${header.model} layers ${header.layers.join('-')} ${header.gameVersion} target ${header.target} wasm ${header.wasm} core ${header.coreHash.slice(0, 12)}`);
  console.log(`  seeds ${seeds[0]}..${seeds[seeds.length - 1]} (${rows.length}, contiguous ${contiguous}); ${fails.length}/${rows.length} failures; two-sided 95% CP [${fmt(ci.lower)}, ${fmt(ci.upper)}]; one-sided 95% upper ${fmt(oneSidedUpper(fails.length, rows.length))}; passing-game frames ${frames.join(',')}; ${footer ? footer.seconds.toFixed(1) + ' s' : 'unfinished'}`);
  for (const r of fails) console.log(`  FAIL seed ${r.seed} score ${r.score}`);
  const v = best.validation || best.val || {};
  console.log(`  best.json validation: ${JSON.stringify(v).slice(0, 300)}`);
}

// Python per-epoch loss vs the prototype's m_base_s1 log (4-decimal precision there).
const proto = fs.readFileSync(protoLog, 'utf8').split('\n').filter(Boolean).map(l => {
  const m = l.match(/ep(\d+) loss ([\d.]+) recall none ([\d.]+) jump ([\d.]+)/);
  return { epoch: Number(m[1]), loss: Number(m[2]), none: Number(m[3]), jump: Number(m[4]) };
});
const py = jsonl(path.join(dir, 'py-s1', 'bc-metrics.jsonl'));
let maxD = 0;
for (const p of proto) {
  const e = py.find(r => r.epoch === p.epoch);
  const d = Math.abs(Number(e.loss.toFixed(4)) - p.loss);
  maxD = Math.max(maxD, d);
  console.log(`ep${p.epoch} proto ${p.loss.toFixed(4)} py ${e.loss.toFixed(6)} |d| ${d.toFixed(4)}`);
}
console.log(`max |d loss| (py rounded to 1e-4 vs proto) = ${maxD.toFixed(4)}`);
