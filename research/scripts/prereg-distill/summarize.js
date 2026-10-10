'use strict';
// Per-model and per-arm summary for research/prereg-distill.md §11b (clean definition of ruling (a)).
// Usage: node summarize.js [--out f.json] <arm prefix> [...]   e.g. v2-bc-duck-d2w6 v2-distill-duck-d1w6-*-tau2
// An arm prefix P covers runs/exp-distill/P-s1..s10 (for distilled arms, give "<base>|<suffix>",
// e.g. "v2-distill-duck-d1w6|tau2" for v2-distill-duck-d1w6-s<k>-tau2).
// Per seed: tier A, opening passed / stuck / early (killed by ptero-mid not in a DOWN run) / other,
// tier B failures (if a complete scan exists), duck share, clean. Per arm: clean/10, min duck share
// over clean seeds, min margin over measured models.
const fs = require('fs');
const G = 'runs/exp-distill';
const rd = (f) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return null; } };
function tierB(name) {
  let txt; try { txt = fs.readFileSync(`${G}/scans/${name}.jsonl`, 'utf8'); } catch { return null; }
  const lines = txt.trim().split('\n').map(l => JSON.parse(l));
  const seeds = lines.filter(l => l.seed !== undefined);
  const footer = lines.some(l => l.footer);
  const set = new Set(seeds.map(s => s.seed));
  const contiguous = set.size === 1000 && [...set].every(s => s >= 30003000 && s < 30004000);
  const fails = seeds.filter(s => !s.pass);
  return { n: set.size, contiguous: contiguous && footer, footer, fails: fails.length, failScores: fails.map(f => f.score) };
}
const out = [];
const argv = process.argv.slice(2), oi = argv.indexOf('--out');
const outFile = oi >= 0 ? argv.splice(oi, 2)[1] : null;
for (const spec of argv) {
  const [base, suf] = spec.split('|');
  const rows = [];
  for (let s = 1; s <= 10; s++) {
    const name = suf ? `${base}-s${s}-${suf}` : `${base}-s${s}`;
    const b = rd(`${G}/${name}/best.json`);
    if (!b) { rows.push({ name, missing: true }); continue; }
    const o = rd(`${G}/opening/${name}.json`);
    const d = rd(`${G}/duckshare/${name}.json`);
    const mg = rd(`${G}/margin/${name}.json`);
    const tb = tierB(name);
    const fails = o ? o.fails : [];
    const stuck = fails.filter(f => f.stuck).length;
    const early = fails.filter(f => !f.stuck && f.killer === 'ptero-mid' && !f.finalDuck).length;
    const r = { name, tierA: b.validation.count, tierAmin: b.validation.min, loss: b.config.final.loss,
      opening: o ? o.passed : null, stuck, early, other: fails.length - stuck - early,
      maxLag: o ? o.lag.max : null, tierB: tb ? tb.fails : null, tierBok: tb ? tb.contiguous : null,
      duckShare: d ? d.duckShare : null, margin: mg ? mg.minMargin : null };
    r.clean = r.tierA === 25 && r.opening === 1000 && (r.tierB === null || r.tierB === 0);
    rows.push(r);
  }
  const ok = rows.filter(r => !r.missing);
  const clean = ok.filter(r => r.clean);
  const ds = clean.map(r => r.duckShare).filter(x => x !== null);
  const mg = ok.map(r => r.margin).filter(x => x !== null);
  const arm = { arm: spec, trained: ok.length, clean: clean.length, cleanSeeds: clean.map(r => r.name.match(/-s(\d+)/)[1]),
    stuckSeeds: ok.filter(r => r.stuck > 0).length, earlySeeds: ok.filter(r => r.early > 0).length,
    minDuckShareClean: ds.length ? Math.min(...ds) : null, minMargin: mg.length ? Math.min(...mg) : null };
  out.push({ ...arm, rows });
  console.log(JSON.stringify(arm));
  for (const r of rows) console.log('  ' + JSON.stringify(r));
}
if (outFile) fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
