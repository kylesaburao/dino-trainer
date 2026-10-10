'use strict';
// Summarise replay runs (HANDOFF 9, fix 1) from their metrics.jsonl and checkpoint config.
// Usage: node tools/replay-table.js <run dir>... [--escape 200] [--tie 0.5] [--sigma-max S] [--json]
//
// Per run:
//   escape    first generation with fitMean > --escape; censored (">N") at the last
//             logged generation (--generations or the time budget)
//   q         per generation: tieFrac (unsaturated tied share) when the run's config has
//             tieTarget > 0, otherwise tiedPairs / (popSize / 2). Chosen by config, not by
//             column presence: from ee6995c on, every run logs tieFrac.
//   streak    longest run of consecutive generations with q > --tie
//   frac      share of generations with q > --tie
//   q1-100, q300-400   mean q over those generations (null if the run never reached them)
//   theta     thetaNorm at the last generation
//   wall      cumulativeMs at the last generation (seconds)
//   sigma     effective sigma per generation (sigmaEff, else sigma; includes the stall multiplier); min, max, share of
//             generations at sigma-max (config sigmaMax unless --sigma-max), and the number of
//             times q crosses the tie target (controller runs only)
//   skipped   generations whose step the stall rule skipped
//   fitMax    secondary outcome: highest fitMean and its generation, and the mean fitMean
//             over the last 50 generations (useful when every run is censored)
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const valued = new Set(['--escape', '--tie', '--sigma-max']);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? Number(args[i + 1]) : def; };
const dirs = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && valued.has(args[i - 1])));
const ESCAPE = opt('--escape', 200), TIE = opt('--tie', 0.5);

function configOf(dir) {
  for (const f of ['checkpoint.json', 'best.json']) {
    try { const c = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')); if (c.config) return c.config; } catch (e) { /* next */ }
  }
  throw new Error(`${dir}: no checkpoint.json or best.json with a config`);
}
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

function summarise(dir) {
  const cfg = configOf(dir);
  const rows = fs.readFileSync(path.join(dir, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const pairs = cfg.popSize / 2;
  const controller = cfg.tieTarget > 0;
  const sigmaMax = opt('--sigma-max', cfg.sigmaMax ?? Infinity);
  const qOf = (r) => (controller ? r.tieFrac : r.tiedPairs / pairs);
  const last = rows[rows.length - 1];
  const esc = rows.find(r => r.fitMean > ESCAPE);
  let streak = 0, cur = 0, over = 0, atMax = 0, crossings = 0, prevSide = 0;
  let sMin = Infinity, sMax = -Infinity;
  for (const r of rows) {
    const q = qOf(r);
    if (q > TIE) { over++; cur++; streak = Math.max(streak, cur); } else cur = 0;
    const s = r.sigmaEff ?? r.sigma;
    sMin = Math.min(sMin, s); sMax = Math.max(sMax, s);
    if (s >= sigmaMax * (1 - 1e-9)) atMax++;
    if (controller) {
      const side = Math.sign(q - cfg.tieTarget);
      if (side && prevSide && side !== prevSide) crossings++;
      if (side) prevSide = side;
    }
  }
  const qWin = (a, b) => (last.generation >= b ? mean(rows.filter(r => r.generation >= a && r.generation <= b).map(qOf)) : null);
  return {
    run: path.basename(dir), hidden: cfg.hidden.join(','), seed: cfg.seed,
    arm: controller ? `tie-target ${cfg.tieTarget}` : cfg.stallRule ? 'stall-rule' : 'none',
    qSource: controller ? 'tieFrac' : 'tiedPairs', gens: last.generation,
    escape: esc ? esc.generation : null, censoredAt: esc ? null : last.generation,
    streak, frac: over / rows.length, q1_100: qWin(1, 100), q300_400: qWin(300, 400),
    thetaNorm: last.thetaNorm, wallS: last.cumulativeMs / 1000,
    sigmaMin: sMin, sigmaMax: sMax, fracAtSigmaMax: isFinite(sigmaMax) ? atMax / rows.length : null,
    targetCrossings: controller ? crossings : null,
    skipped: rows.filter(r => r.stepSkipped).length, finalFitMean: last.fitMean,
    peakFitMean: Math.max(...rows.map(r => r.fitMean)),
    peakGen: rows.reduce((b, r) => (r.fitMean > b.fitMean ? r : b)).generation,
    last50FitMean: mean(rows.slice(-50).map(r => r.fitMean)),
  };
}

const out = dirs.map(summarise);
if (args.includes('--json')) { console.log(JSON.stringify(out, null, 1)); process.exit(0); }
const f = (x, d = 2) => (x == null || !isFinite(x) ? '-' : x.toFixed(d));
console.log('| Run | Arm | Escape | Max streak q>0.5 | Frac q>0.5 | q 1–100 | q 300–400 | ‖θ‖ end | Wall s | σ min–max | Frac at σmax | q* crossings | Skipped | fitMean end | Peak fitMean (gen) | fitMean last 50 |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
for (const r of out) {
  console.log(`| ${r.run} | ${r.arm} | ${r.escape ?? '>' + r.censoredAt} | ${r.streak} | ${f(r.frac)} | ${f(r.q1_100)} | ${f(r.q300_400)} | ${f(r.thetaNorm, 1)} | ${f(r.wallS, 0)} | ${f(r.sigmaMin, 3)}–${f(r.sigmaMax, 3)} | ${f(r.fracAtSigmaMax)} | ${r.targetCrossings ?? '-'} | ${r.skipped} | ${f(r.finalFitMean, 0)} | ${f(r.peakFitMean, 0)} (${r.peakGen}) | ${f(r.last50FitMean, 0)} |`);
}
