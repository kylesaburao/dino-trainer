'use strict';
// Results table for research/prereg-es-curriculum.md (Run B' vs hygiene, paired start thetas).
// Usage: node tools/bp-table.js [--runs runs] [--prefix exp-bp] [--seeds 12,4,5,7,8,9] [--controls 4,12]
//                               [--base hyg] [--arms hyg,cur] [--threshold 2000] [--horizon 2000] [--json]
// Reads runs/<prefix>-s<S>-<arm>/metrics.jsonl; R = <arm> frames / <base> frames for each other arm. The resume point G0 is the first logged
// generation - 1.
//   event       first of two consecutive validations with valMedian >= threshold
//   frames      training frames (sum of `frames` + `teacherFrames`, the ramp arm's teacher play)
//               from the resume to the event generation, or over the whole run when censored;
//               studentFrames = sum of `frames` alone
//   censored    no event by G0 + horizon (or the run ended early: budget)
//   R           cur frames / hyg frames; bounds when one side is censored
//   opening     lowest valMedian after G0 + 100, and whether two consecutive validations
//               after G0 + 100 fell below 200 (pre-registered harm trigger)
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const RUNS = opt('--runs', 'runs'), THRESHOLD = Number(opt('--threshold', 2000)), HORIZON = Number(opt('--horizon', 2000));
const PREFIX = opt('--prefix', 'exp-bp');
const SEEDS = opt('--seeds', '12,4,5,7,8,9').split(',').map(Number);
const CONTROLS = new Set(opt('--controls', '4,12').split(',').map(Number));
const BASE = opt('--base', 'hyg');
const ARMS = opt('--arms', 'hyg,cur').split(',');

function summarise(s, arm) {
  const f = path.join(RUNS, `${PREFIX}-s${s}-${arm}`, 'metrics.jsonl');
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
    frames: upto.reduce((a, r) => a + r.frames + (r.teacherFrames || 0), 0), studentFrames: upto.reduce((a, r) => a + r.frames, 0), wallS: (upto[upto.length - 1].cumulativeMs - (rows[0].cumulativeMs - rows[0].genMs)) / 1000,
    censored: event == null, stopRule, reachedHorizon: last.generation >= g0 + HORIZON,
    minValAfter100: late.length ? Math.min(...late.map(r => r.valMedian)) : null, openingLoss,
    tied: rows.reduce((a, r) => a + r.tiedPairs, 0) / rows.length / 64,
    windows: [...new Set(rows.map(r => r.window).filter(w => w != null))],
    lastVal: vals.length ? vals[vals.length - 1].valMedian : null,
  };
}

const out = [];
for (const s of SEEDS) {
  const runs = Object.fromEntries(ARMS.map(a => [a, summarise(s, a)]));
  const b = runs[BASE];
  const ratios = {};
  for (const a of ARMS) {
    if (a === BASE) continue;
    const c = runs[a];
    if (!b || !c) { ratios[a] = { R: null, txt: '-' }; continue; }
    const R = c.frames / b.frames;
    const txt = !b.censored && !c.censored ? R.toFixed(2) : b.censored && !c.censored ? `≤ ${R.toFixed(2)}` : !b.censored && c.censored ? `≥ ${R.toFixed(2)}` : 'both censored';
    // win for the arm: R < 1 (including arm clears, base censored); both censored is not a win
    const win = (!c.censored && (b.censored || R < 1));
    ratios[a] = { R, txt, win };
  }
  out.push({ seed: s, control: CONTROLS.has(s), runs, ratios });
}
if (args.includes('--json')) { console.log(JSON.stringify(out, null, 1)); process.exit(0); }
const cell = (r) => (r ? `${r.censored ? '>' + r.gens : r.gens}${r.stopRule ? '*' : ''} gens, ${(r.frames / 1e6).toFixed(0)}M fr${r.studentFrames !== r.frames ? ` (${(r.studentFrames / 1e6).toFixed(0)}M student)` : ''}, ${r.wallS.toFixed(0)} s` : 'not run');
const extra = (r) => (r ? `${r.tied.toFixed(2)} / ${r.minValAfter100 ?? '-'}${r.openingLoss ? ' LOSS' : ''}${r.windows.length > 1 ? ' / win ' + r.windows.join('→') : ''}${r.censored && !r.reachedHorizon ? ' / ended early' : ''}` : '');
const others = ARMS.filter(a => a !== BASE);
console.log('(* = confirmed by the trainer\'s held-out stop-rule check instead of a second validation)');
console.log(`| θ | Role | G0 | ${ARMS.map(a => a + ': to event').join(' | ')} | ${others.map(a => `R = ${a}/${BASE}`).join(' | ')} | ${ARMS.map(a => a + ' tied / min val after +100').join(' | ')} |`);
console.log(`|${'---|'.repeat(3 + ARMS.length * 2 + others.length)}`);
for (const o of out) {
  const g0 = ARMS.map(a => o.runs[a]?.g0).find(x => x != null) ?? '-';
  console.log(`| s${o.seed} | ${o.control ? 'control' : 'stuck'} | ${g0} | ${ARMS.map(a => cell(o.runs[a])).join(' | ')} | ${others.map(a => o.ratios[a].txt + (o.ratios[a].win ? ' (win)' : '')).join(' | ')} | ${ARMS.map(a => extra(o.runs[a])).join(' | ')} |`);
}
for (const a of others) {
  const stuck = out.filter(o => !o.control && o.runs[a] && o.runs[BASE]);
  const wins = stuck.filter(o => o.ratios[a].win).length;
  const n = stuck.length;
  let p = 0; const C = (n, k) => { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); return c; };
  for (let k = wins; k <= n; k++) p += C(n, k) / 2 ** n;
  const half = stuck.filter(o => o.ratios[a].R != null && (o.ratios[a].win || o.ratios[a].txt.startsWith('≤') || !o.runs[a].censored) && o.ratios[a].R <= 0.5 && !o.runs[a].censored).length;
  console.log(`\n${a} vs ${BASE} on stuck θs: ${wins}/${n} wins (one-sided sign test p = ${p.toFixed(3)}); R ≤ 0.5 on ${half}/${n}`);
}
