'use strict';
// Results table for research/prereg-es-curriculum.md (Run B' vs hygiene, paired start thetas).
// Usage: node tools/bp-table.js [--runs runs] [--threshold 2000] [--horizon 2000] [--json]
// Reads runs/exp-bp-s<S>-{hyg,cur}/metrics.jsonl. The resume point G0 is the first logged
// generation - 1.
//   event       first of two consecutive validations with valMedian >= threshold
//   frames      training frames (sum of `frames`) from the resume to the event generation,
//               or over the whole run when censored
//   censored    no event by G0 + horizon (or the run ended early: budget)
//   R           cur frames / hyg frames; bounds when one side is censored
//   opening     lowest valMedian after G0 + 100, and whether two consecutive validations
//               after G0 + 100 fell below 200 (pre-registered harm trigger)
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const RUNS = opt('--runs', 'runs'), THRESHOLD = Number(opt('--threshold', 2000)), HORIZON = Number(opt('--horizon', 2000));
const SEEDS = [12, 4, 5, 7, 8, 9];
const CONTROLS = new Set([4, 12]);

function summarise(s, arm) {
  const f = path.join(RUNS, `exp-bp-s${s}-${arm}`, 'metrics.jsonl');
  if (!fs.existsSync(f)) return null;
  const rows = fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  if (!rows.length) return null;
  const g0 = rows[0].generation - 1;
  const vals = rows.filter(r => r.valMedian != null);
  let event = null;
  for (let i = 1; i < vals.length; i++) if (vals[i - 1].valMedian >= THRESHOLD && vals[i].valMedian >= THRESHOLD) { event = vals[i - 1].generation; break; }
  // The trainer's default stop rule (median) ends a run when a validation median reaches
  // --target and the held-out median (25 seeds) does too, before the watcher's second
  // reading. That held-out check counts as the confirming reading (prereg §8, deviation 1).
  let stopRule = false;
  try {
    const tl = fs.readFileSync(path.join(path.dirname(f), 'train.log'), 'utf8');
    const m = tl.match(/done: held-out median \d+ >= target \d+; generation (\d+)/g);
    if (m && event == null) {
      const g = Number(m[m.length - 1].match(/generation (\d+)/)[1]);
      const v = vals.find(r => r.generation === g);
      if (v && v.valMedian >= THRESHOLD) { event = g; stopRule = true; }
    }
  } catch (e) { /* no log */ }
  const upto = rows.filter(r => event == null || r.generation <= event);
  const late = vals.filter(r => r.generation > g0 + 100);
  let openingLoss = false;
  for (let i = 1; i < late.length; i++) if (late[i - 1].valMedian < 200 && late[i].valMedian < 200) openingLoss = true;
  const last = rows[rows.length - 1];
  return {
    seed: s, arm, g0, lastGen: last.generation, event, gens: event != null ? event - g0 : last.generation - g0,
    frames: upto.reduce((a, r) => a + r.frames, 0), wallS: (upto[upto.length - 1].cumulativeMs - (rows[0].cumulativeMs - rows[0].genMs)) / 1000,
    censored: event == null, stopRule, reachedHorizon: last.generation >= g0 + HORIZON,
    minValAfter100: late.length ? Math.min(...late.map(r => r.valMedian)) : null, openingLoss,
    tied: rows.reduce((a, r) => a + r.tiedPairs, 0) / rows.length / 64,
    windows: [...new Set(rows.map(r => r.window).filter(w => w != null))],
    lastVal: vals.length ? vals[vals.length - 1].valMedian : null,
  };
}

const out = [];
for (const s of SEEDS) {
  const h = summarise(s, 'hyg'), c = summarise(s, 'cur');
  let R = null, Rtxt = '-';
  if (h && c) {
    R = c.frames / h.frames;
    Rtxt = !h.censored && !c.censored ? R.toFixed(2) : h.censored && !c.censored ? `≤ ${R.toFixed(2)}` : !h.censored && c.censored ? `≥ ${R.toFixed(2)}` : 'both censored';
  }
  out.push({ seed: s, control: CONTROLS.has(s), hyg: h, cur: c, R, Rtxt });
}
if (args.includes('--json')) { console.log(JSON.stringify(out, null, 1)); process.exit(0); }
const cell = (r) => (r ? `${r.censored ? '>' + r.gens : r.gens}${r.stopRule ? '*' : ''} gens, ${(r.frames / 1e6).toFixed(0)}M fr, ${r.wallS.toFixed(0)} s` : 'not run');
const extra = (r) => (r ? `${r.tied.toFixed(2)} / ${r.minValAfter100 ?? '-'}${r.openingLoss ? ' LOSS' : ''}${r.windows.length ? ' / win ' + r.windows.join('→') : ''}${r.censored && !r.reachedHorizon ? ' / ended early' : ''}` : '');
console.log('(* = confirmed by the trainer\'s held-out stop-rule check instead of a second validation)');
console.log('| θ | Role | G0 | hyg: to event | cur (B′): to event | R = cur/hyg frames | hyg tied / min val after +100 | cur tied / min val after +100 / window |');
console.log('|---|---|---|---|---|---|---|---|');
for (const o of out) {
  console.log(`| s${o.seed} | ${o.control ? 'control' : 'stuck'} | ${o.hyg?.g0 ?? o.cur?.g0 ?? '-'} | ${cell(o.hyg)} | ${cell(o.cur)} | ${o.Rtxt} | ${extra(o.hyg)} | ${extra(o.cur)} |`);
}
