'use strict';
// Seed-survey summary: per-run escape gen (fitMean > 200), first gen with fitMean > 2000,
// wall to escape; Kaplan-Meier over runs plus extra (gen, escaped) points; P(escape by T);
// sd(ln T) over escaped runs; K seeds for 95% racing success.
// Usage: node tools/survey.js <run dir>... [--extra name:gen:0|1,...]
// --extra adds runs not on disk as name:generation:escaped (1) or censored (0).
// Prints a per-run table, then JSON: Kaplan-Meier steps and median, P(escape by 900/1500)
// (KM and naive), sd(ln T) over escaped runs (biased low under censoring), and K95.
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const xi = args.indexOf('--extra');
const extra = xi >= 0 ? args[xi + 1].split(',').map(s => { const [name, g, e] = s.split(':'); return { run: name, t: Number(g), escaped: e === '1' }; }) : [];
const dirs = args.filter((a, i) => a !== '--extra' && args[i - 1] !== '--extra');

const runs = dirs.map((dir) => {
  const rows = fs.readFileSync(path.join(dir, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const last = rows[rows.length - 1];
  const esc = rows.find(r => r.fitMean > 200);
  const k2 = rows.find(r => r.fitMean > 2000);
  return {
    run: path.basename(dir), t: esc ? esc.generation : last.generation, escaped: !!esc, lastGen: last.generation,
    gen2000: k2 ? k2.generation : null, wallToEscapeS: esc ? esc.cumulativeMs / 1000 : null,
    wallS: last.cumulativeMs / 1000, finalFitMean: last.fitMean,
  };
});

console.log('| Run | Escape gen | First fitMean > 2000 | Wall to escape s | Last gen | Total wall s | fitMean end |');
console.log('|---|---|---|---|---|---|---|');
for (const r of runs) {
  console.log(`| ${r.run} | ${r.escaped ? r.t : '>' + r.t} | ${r.gen2000 ?? '-'} | ${r.wallToEscapeS?.toFixed(0) ?? '-'} | ${r.lastGen} | ${r.wallS.toFixed(0)} | ${r.finalFitMean.toFixed(0)} |`);
}

const all = [...runs.map(({ run, t, escaped }) => ({ run, t, escaped })), ...extra];
// Kaplan-Meier: events at escape gens; censored runs leave the risk set after their t.
const times = [...new Set(all.filter(r => r.escaped).map(r => r.t))].sort((a, b) => a - b);
let S = 1;
const km = [];
for (const t of times) {
  const atRisk = all.filter(r => r.t >= t).length;
  const d = all.filter(r => r.escaped && r.t === t).length;
  S *= 1 - d / atRisk;
  km.push({ t, atRisk, d, S });
}
const Sat = (T) => { let s = 1; for (const k of km) if (k.t <= T) s = k.S; return s; };
const kmMedian = km.find(k => k.S <= 0.5)?.t ?? null;
const maxFollow = Math.max(...all.map(r => r.t));

const escapedTs = all.filter(r => r.escaped).map(r => Math.log(r.t));
const m = escapedTs.reduce((a, b) => a + b, 0) / escapedTs.length;
const sd = escapedTs.length > 1 ? Math.sqrt(escapedTs.reduce((a, x) => a + (x - m) ** 2, 0) / (escapedTs.length - 1)) : null;
const K = (p) => (p > 0 ? Math.ceil(Math.log(0.05) / Math.log(1 - p)) : Infinity);

const naive = (T) => {
  const eligible = all.filter(r => r.escaped ? true : r.t >= T); // followed to T, or escaped
  const hit = eligible.filter(r => r.escaped && r.t <= T).length;
  return { hit, n: eligible.length, p: hit / eligible.length };
};
const summary = {
  n: all.length, escaped: all.filter(r => r.escaped).length, maxFollowUp: maxFollow,
  km, kmMedian: kmMedian ?? `not reached (S(${maxFollow}) = ${Sat(maxFollow).toFixed(3)})`,
  P900: { km: 1 - Sat(900), naive: naive(900) }, P1500: { km: 1 - Sat(1500), naive: naive(1500) },
  sdLnT: sd, escapedGens: all.filter(r => r.escaped).map(r => r.t).sort((a, b) => a - b),
  K95: { at900: K(1 - Sat(900)), at1500: K(1 - Sat(1500)) },
};
console.log(JSON.stringify(summary, null, 1));
