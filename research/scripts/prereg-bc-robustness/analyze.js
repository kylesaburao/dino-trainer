'use strict';
// Statistics for the robustness scan: per-arm failure counts with two-sided 95%
// Clopper-Pearson intervals, the P gate, exact one-sided McNemar for N1 vs C1, and
// the pairing of failures across arms. Readings (§5) are applied by hand in the doc.
// Usage: node analyze.js <dir with P.jsonl C1.jsonl N1.jsonl C2.jsonl>
const fs = require('fs');
const path = require('path');
const dir = path.resolve(process.argv[2]);

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
// Two-sided 95% CP for k successes in n.
function cp(k, n) {
  const lower = k === 0 ? 0 : bisect(p => 1 - cdf(n, k - 1, p) >= 0.025);   // P(X>=k|p) = 0.025
  const upper = k === n ? 1 : bisect(p => cdf(n, k, p) <= 0.025);            // P(X<=k|p) = 0.025
  return { lower, upper };
}
const oneSidedUpper = (k, n) => bisect(p => cdf(n, k, p) <= 0.05);

const read = (arm) => {
  const f = path.join(dir, `${arm}.jsonl`);
  if (!fs.existsSync(f)) return null;
  const ls = fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  return { header: ls.find(l => l.header), footer: ls.find(l => l.footer), rows: ls.filter(l => l.seed !== undefined) };
};
const arms = {};
for (const a of ['P', 'C1', 'N1', 'C2']) arms[a] = read(a);
const fmt = (x) => (x * 100).toFixed(3) + '%';
for (const [a, r] of Object.entries(arms)) {
  if (!r) continue;
  const n = r.rows.length, fails = r.rows.filter(x => !x.pass);
  const ci = cp(fails.length, n);
  console.log(`${a}: ${fails.length}/${n} failures; 95% CP [${fmt(ci.lower)}, ${fmt(ci.upper)}]; one-sided 95% upper ${fmt(oneSidedUpper(fails.length, n))}; wasm ${r.header.wasm}; ${r.footer ? r.footer.seconds.toFixed(1) + ' s' : 'unfinished'}`);
  for (const x of fails) console.log(`   seed ${x.seed} score ${x.score}`);
}
if (arms.P) {
  const f = arms.P.rows.filter(x => !x.pass);
  const zone = f.filter(x => (x.score >= 460 && x.score <= 740) || x.score >= 990000).length;
  console.log(`P gate: ${f.length} failures -> ${f.length < 3 ? 'STOP' : f.length <= 7 ? 'proceed (rate low)' : 'proceed'}; ${zone}/${f.length} in known zones (460-740 or >=990k)`);
}
if (arms.C1 && arms.N1) {
  const c1 = new Map(arms.C1.rows.map(x => [x.seed, x.pass])), n1 = new Map(arms.N1.rows.map(x => [x.seed, x.pass]));
  let b = 0, c = 0;
  for (const [s, p] of n1) { if (!c1.has(s)) throw new Error('unpaired seed ' + s); if (!p && c1.get(s)) b++; if (p && !c1.get(s)) c++; }
  const nd = b + c;
  let pv = null; if (nd > 0) { pv = 0; for (let i = b; i <= nd; i++) pv += pmf(nd, i, 0.5); }
  console.log(`McNemar N1 vs C1: b (N1 fail, C1 pass) = ${b}, c (C1 fail, N1 pass) = ${c}; one-sided exact p = ${pv === null ? 'undefined (no discordant pairs)' : pv.toPrecision(4)}`);
}
const failSeeds = new Set();
for (const a of ['C1', 'N1', 'C2']) if (arms[a]) for (const x of arms[a].rows) if (!x.pass) failSeeds.add(x.seed);
console.log(`clone-arm failing seeds (for S): ${[...failSeeds].sort().join(',') || 'none'}`);
if (arms.C1 && arms.C2) { const n = arms.C1.rows.length + arms.C2.rows.length; const k = [arms.C1, arms.C2].reduce((t, r) => t + r.rows.filter(x => !x.pass).length, 0); const ci = cp(k, n); console.log(`C1+C2 pooled: ${k}/${n}; 95% CP [${fmt(ci.lower)}, ${fmt(ci.upper)}]`); }
console.log(`check: CP upper 0/2000 = ${fmt(cp(0, 2000).upper)}, 0/150 = ${fmt(cp(0, 150).upper)} (expect ~0.184% and 2.43%)`);
