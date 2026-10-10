'use strict';
// Tier-B pairing for research/prereg-teacher-duck-step2.md §5.3 step 2 and §6.
// For each "duck=twin" pair of scan files: contiguity check (header base/n, n seed rows,
// contiguous, footer), failures, and the exact one-sided McNemar test (H1: duck clone fails
// more): b = only the duck clone fails, c = only the twin fails, p = P(X >= b), X ~ Bin(b+c, 1/2).
// Also pooled per arm (two training seeds, 2,000 pairs). Prints JSON.
// Usage: node pair.js <arm>:<duck.jsonl>:<twin.jsonl> [...]
const fs = require('fs');
function load(f) {
  const L = fs.readFileSync(f, 'utf8').trim().split('\n').map(l => JSON.parse(l));
  const h = L[0], ft = L[L.length - 1], rows = L.slice(1, -1);
  const ok = !!h.header && !!ft.footer && rows.length === h.n && rows.every((r, i) => r.seed === h.base + i);
  const fail = rows.filter(r => !r.pass).map(r => ({ seed: r.seed, score: r.score }));
  return { file: f, ok, n: rows.length, base: h.base, coreHash: h.coreHash, seconds: ft.seconds, fail, map: new Map(rows.map(r => [r.seed, r.pass])) };
}
function binomTail(b, n) { let p = 0; for (let k = b; k <= n; k++) { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); p += c * Math.pow(0.5, n); } return n === 0 ? 1 : p; }
const out = [], pooled = {};
for (const spec of process.argv.slice(2)) {
  const [arm, df, tf] = spec.split(':');
  const d = load(df), t = load(tf);
  let b = 0, c = 0;
  for (const [s, p] of d.map) { const q = t.map.get(s); if (q === undefined) throw new Error('unpaired seed ' + s); if (!p && q) b++; if (p && !q) c++; }
  const key = arm.replace(/-s\d+$/, '');
  pooled[key] = pooled[key] || { b: 0, c: 0, pairs: 0 };
  pooled[key].b += b; pooled[key].c += c; pooled[key].pairs += d.n;
  out.push({ arm, duck: { ok: d.ok, n: d.n, fails: d.fail, coreHash: d.coreHash.slice(0, 12), seconds: d.seconds }, twin: { file: tf, ok: t.ok, n: t.n, fails: t.fail }, b, c, p: binomTail(b, b + c) });
}
for (const k in pooled) pooled[k].p = binomTail(pooled[k].b, pooled[k].b + pooled[k].c);
console.log(JSON.stringify({ perSeed: out, pooled }, null, 1));
