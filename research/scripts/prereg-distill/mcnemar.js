'use strict';
// Exact one-sided McNemar of each D1 arm against arm (a), seed for seed over seeds 1-10
// (research/prereg-distill.md §11b (a)): b = seeds clean in the arm and not in (a), c = the reverse,
// p = P(Bin(b + c, 1/2) >= b). Reads a summarize.js --out file.
// Usage: node mcnemar.js <summary.json> [--out f.json]
const fs = require('fs');
const args = process.argv.slice(2);
const oi = args.indexOf('--out'); const outFile = oi >= 0 ? args.splice(oi, 2)[1] : null;
const arms = JSON.parse(fs.readFileSync(args[0], 'utf8'));
const choose = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; };
const pOne = (b, c) => { const n = b + c; if (!n) return 1; let s = 0; for (let k = b; k <= n; k++) s += choose(n, k); return s / 2 ** n; };
const cleanSet = (a) => new Set(a.rows.filter(r => r.clean).map(r => Number(r.name.match(/-s(\d+)/)[1])));
const res = [];
for (const a of arms) {
  const m = a.arm.match(/^v2-distill-duck-(d\d+w\d+)\|(.+)$/);
  if (!m) continue;
  const base = arms.find(x => x.arm === `v2-bc-duck-${m[1]}`);
  if (!base) throw new Error(`no direct arm for ${a.arm}`);
  const A = cleanSet(a), D = cleanSet(base);
  let b = 0, c = 0;
  for (let s = 1; s <= 10; s++) { if (A.has(s) && !D.has(s)) b++; if (!A.has(s) && D.has(s)) c++; }
  const r = { arm: a.arm, clean: A.size, direct: D.size, b, c, p: pOne(b, c) };
  res.push(r); console.log(JSON.stringify(r));
}
if (outFile) fs.writeFileSync(outFile, JSON.stringify(res, null, 1));
