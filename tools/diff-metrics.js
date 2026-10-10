'use strict';
// Compare two metrics.jsonl files generation by generation.
// Usage: node tools/diff-metrics.js <a/metrics.jsonl> <b/metrics.jsonl> [--fields fitMean,fitMax,valMedian] [--tol 1e-9] [--upto N]
// Reports the first generation and field whose values differ by more than tol
// (absolute, or relative to |a| when |a| > 1), plus per-field mismatch counts.
const fs = require('fs');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const [fa, fb] = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const fields = opt('--fields', 'fitMean,fitMax,valMedian').split(',');
const tol = Number(opt('--tol', '1e-9'));
const upto = Number(opt('--upto', 'Infinity'));

const load = (f) => {
  const m = new Map();
  for (const line of fs.readFileSync(f, 'utf8').split('\n')) if (line.trim()) { const r = JSON.parse(line); m.set(r.generation, r); }
  return m;
};
const A = load(fa), B = load(fb);
const differ = (x, y) => {
  if (x == null || y == null) return (x == null) !== (y == null);
  return Math.abs(x - y) > tol * Math.max(1, Math.abs(x));
};

let compared = 0, first = null;
const counts = Object.fromEntries(fields.map(f => [f, 0]));
const gens = [...A.keys()].filter(g => B.has(g) && g <= upto).sort((x, y) => x - y);
for (const g of gens) {
  compared++;
  for (const f of fields) {
    if (differ(A.get(g)[f], B.get(g)[f])) {
      counts[f]++;
      if (!first) first = { generation: g, field: f, a: A.get(g)[f], b: B.get(g)[f] };
    }
  }
}
console.log(JSON.stringify({ compared, firstGen: gens[0], lastGen: gens[gens.length - 1], tol, mismatches: counts, first }, null, 1));
process.exit(first ? 1 : 0);
