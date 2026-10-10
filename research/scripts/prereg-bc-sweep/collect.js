'use strict';
// Collects every sweep measurement (research/prereg-bc-sweep.md §4) into
// runs/exp-bc-sweep/summary.json and prints the report tables (markdown). Optionally writes
// the learning-curve SVG (agreement against training FLOPs) with --svg <file>.
// Reads per run dir runs/exp-bc-sweep/d<d>-w<w>-s<seed>/: best.json (tier A), agreement.json
// (agreement.py), export/model.{js,css} (bytes), train.log (/usr/bin/time CPU, train seconds),
// bc-metrics.jsonl (epoch seconds); scans/<run>.jsonl (tier B); margins/<run>.json;
// bench/<d>-<w>.json (bench policyFps).
// Usage: node collect.js [--svg <file>]
const fs = require('fs');
const path = require('path');
const G = require('./grid.js');
const ROOT = '/Users/kyle/Desktop/dino-game';
const R = path.join(ROOT, 'runs', 'exp-bc-sweep');
const T1 = 0.995, T2 = 0.998;
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };

// Clopper-Pearson (copied from research/scripts/prereg-bc-pipeline-e2e/analyze.js).
function lgamma(x) {
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
const oneSidedUpper = (k, n) => bisect(p => cdf(n, k, p) <= 0.05);

const readJson = (f) => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null);
const jsonl = (f) => (fs.existsSync(f) ? fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : null);
const size = (f) => (fs.existsSync(f) ? fs.statSync(f).size : null);

function scan(run) {
  const ls = jsonl(path.join(R, 'scans', `${run}.jsonl`));
  if (!ls) return null;
  const rows = ls.filter(l => l.seed !== undefined), footer = ls.find(l => l.footer), header = ls.find(l => l.header);
  const fails = rows.filter(r => !r.pass);
  const contiguous = [...rows].map(r => r.seed).sort((a, b) => a - b).every((s, i) => s === header.base + i);
  return { games: rows.length, failures: fails.length, failScores: fails.map(r => [r.seed, r.score]), complete: !!footer && rows.length === header.n && contiguous,
    seconds: footer ? footer.seconds : null, upper95: oneSidedUpper(fails.length, rows.length) };
}

function run(d, w, seed) {
  const name = G.name(d, w, seed), dir = path.join(R, name);
  const best = readJson(path.join(dir, 'best.json'));
  if (!best) return null;
  const ag = readJson(path.join(dir, 'agreement.json'));
  const log = fs.readFileSync(path.join(dir, 'train.log'), 'utf8');
  const tm = log.match(/([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys/);
  const metrics = jsonl(path.join(dir, 'bc-metrics.jsonl')) || [];
  const macs = G.macs(d, w);
  const epochs = ag ? ag.epochs : [];
  const first = (t) => { const e = epochs.find(r => r.agreement >= t); return e ? { epoch: e.epoch, flops: G.flops(d, w, e.epoch) } : null; };
  const last = epochs[epochs.length - 1] || {};
  return {
    run: name, d, w, seed, macs, params: G.params(d, w), layers: best.layers.join('-'),
    modelJsBytes: size(path.join(dir, 'export', 'model.js')), modelCssBytes: size(path.join(dir, 'export', 'model.css')),
    tierA: { count: best.validation.count, median: best.validation.median, min: best.validation.min },
    agreement: { final: last.agreement, recallNone: last.recallNone, recallJump: last.recallJump, max: Math.max(...epochs.map(e => e.agreement)),
      curve: epochs.map(e => e.agreement), predCountsFinal: last.predCounts },
    t1: first(T1), t2: first(T2), trainFlops: G.flops(d, w, G.EPOCHS),
    finalLoss: best.config.final && best.config.final.loss,
    secondary: { trainWallSeconds: best.config.trainSeconds, epochSecondsSum: metrics.reduce((a, r) => a + r.seconds, 0),
      jobCpuSeconds: tm ? Number(tm[2]) + Number(tm[3]) : null, jobWallSeconds: tm ? Number(tm[1]) : null },
    tierB: scan(name), margin: readJson(path.join(R, 'margins', `${name}.json`)),
    benchPolicyFps: (readJson(path.join(R, 'bench', `d${d}-w${w}.json`)) || {}).policyFps ?? null,
  };
}

const all = [];
for (const c of G.configs) for (const seed of [1, 2]) { const r = run(c.d, c.w, seed); if (r) all.push(r); }
fs.writeFileSync(path.join(R, 'summary.json'), JSON.stringify({ T1, T2, samplesPerEpoch: G.SAMPLES_PER_EPOCH, runs: all }, null, 1) + '\n');

const fmtE = (x) => (x == null ? '—' : x.toExponential(2).replace('e+', 'e'));
const pct = (x, dp = 2) => (x == null ? '—' : (x * 100).toFixed(dp) + '%');
const n = (x) => (x == null ? '—' : x.toLocaleString('en-US'));
const tb = (s) => (!s ? '—' : `${s.failures}/${s.games}${s.complete ? '' : ' (incomplete)'}`);

const s1 = all.filter(r => r.seed === 1);
const byName = new Map(all.map(r => [r.run, r]));
console.log('### Per-configuration table (training seed 1; tier B seed 2 where run)\n');
console.log('| d | w | params | MACs/decision | model.js B | model.css B | tier A | held-out agreement (ep 15) | recall none / jump | T1 epoch (FLOPs) | T2 epoch (FLOPs) | train FLOPs (15 ep) | tier B s1 | tier B s2 |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
for (const r of s1.sort((a, b) => a.w - b.w || a.d - b.d)) {
  const r2 = byName.get(G.name(r.d, r.w, 2));
  console.log(`| ${r.d} | ${r.w} | ${n(r.params)} | ${n(r.macs)} | ${n(r.modelJsBytes)} | ${n(r.modelCssBytes)} | ${r.tierA.count}/25 | ${pct(r.agreement.final, 3)} | ${pct(r.agreement.recallNone)} / ${pct(r.agreement.recallJump)} | ` +
    `${r.t1 ? `${r.t1.epoch} (${fmtE(r.t1.flops)})` : 'never'} | ${r.t2 ? `${r.t2.epoch} (${fmtE(r.t2.flops)})` : 'never'} | ${fmtE(r.trainFlops)} | ${tb(r.tierB)} | ${r2 ? tb(r2.tierB) : '—'} |`);
}

console.log('\n### Seed-2 replicates (S2 candidates)\n');
console.log('| config | tier A s2 | agreement s2 (ep 15) | T1 / T2 epoch s2 | tier B s2 | min margin s1 / s2 |');
console.log('|---|---|---|---|---|---|');
for (const r of all.filter(r => r.seed === 2).sort((a, b) => a.macs - b.macs)) {
  const r1 = byName.get(G.name(r.d, r.w, 1));
  console.log(`| d${r.d} w${r.w} | ${r.tierA.count}/25 | ${pct(r.agreement.final, 3)} | ${r.t1 ? r.t1.epoch : 'never'} / ${r.t2 ? r.t2.epoch : 'never'} | ${tb(r.tierB)} | ` +
    `${r1 && r1.margin ? r1.margin.minMargin.toExponential(2) : '—'} / ${r.margin ? r.margin.minMargin.toExponential(2) : '—'} |`);
}

console.log('\n### Learning curves: held-out agreement after each epoch (seed 1)\n');
console.log('| config | FLOPs/epoch | ' + Array.from({ length: 15 }, (_, i) => `ep${i + 1}`).join(' | ') + ' |');
console.log('|---|---|' + '---|'.repeat(15));
for (const r of s1.sort((a, b) => a.w - b.w || a.d - b.d)) {
  console.log(`| d${r.d} w${r.w} | ${fmtE(G.flops(r.d, r.w, 1))} | ` + r.agreement.curve.map(a => (a * 100).toFixed(2)).join(' | ') + ' |');
}

console.log('\n### Secondary (time-dependent, measured under parallel load; not used to rank)\n');
console.log('| config | train wall s | sum of epoch s | job CPU s (train + tier-A validation) | bench JS frames/s (sim + policy, 1 thread) |');
console.log('|---|---|---|---|---|');
for (const r of all.sort((a, b) => a.w - b.w || a.d - b.d || a.seed - b.seed)) {
  console.log(`| ${r.run} | ${r.secondary.trainWallSeconds.toFixed(1)} | ${r.secondary.epochSecondsSum.toFixed(1)} | ${r.secondary.jobCpuSeconds == null ? '—' : r.secondary.jobCpuSeconds.toFixed(0)} | ${r.seed === 1 ? n(r.benchPolicyFps) : ''} |`);
}

// Learning-curve SVG: one panel per width; x = log10 training FLOPs, y = agreement.
const svgFile = opt('--svg');
if (svgFile) {
  const W = 440, H = 300, padL = 52, padR = 12, padT = 26, padB = 40;
  const xMin = 10, xMax = 14, yMin = 0.985, yMax = 1.0;
  const colors = { 1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 16: '#9467bd', 32: '#8c564b', 64: '#e377c2' };
  const panels = G.WIDTHS.map((w, pi) => {
    const ox = (pi % 2) * W, oy = Math.floor(pi / 2) * H;
    const X = (f) => ox + padL + (Math.log10(f) - xMin) / (xMax - xMin) * (W - padL - padR);
    const Y = (a) => oy + padT + (1 - (Math.max(yMin, Math.min(yMax, a)) - yMin) / (yMax - yMin)) * (H - padT - padB);
    let s = `<text x="${ox + W / 2}" y="${oy + 16}" text-anchor="middle" font-size="13">width ${w}</text>`;
    s += `<rect x="${ox + padL}" y="${oy + padT}" width="${W - padL - padR}" height="${H - padT - padB}" fill="none" stroke="#999"/>`;
    for (let e = xMin; e <= xMax; e++) s += `<line x1="${X(10 ** e)}" x2="${X(10 ** e)}" y1="${oy + padT}" y2="${oy + H - padB}" stroke="#eee"/><text x="${X(10 ** e)}" y="${oy + H - padB + 14}" text-anchor="middle" font-size="10">1e${e}</text>`;
    for (const a of [0.985, 0.99, 0.995, 0.998, 1.0]) s += `<line x1="${ox + padL}" x2="${ox + W - padR}" y1="${Y(a)}" y2="${Y(a)}" stroke="${a === T1 || a === T2 ? '#bbb' : '#eee'}" ${a === T1 || a === T2 ? 'stroke-dasharray="4 3"' : ''}/><text x="${ox + padL - 4}" y="${Y(a) + 3}" text-anchor="end" font-size="10">${(a * 100).toFixed(1)}</text>`;
    s += `<text x="${ox + (W + padL) / 2}" y="${oy + H - 8}" text-anchor="middle" font-size="10">training FLOPs (log)</text>`;
    for (const r of s1.filter(r => r.w === w)) {
      const pts = r.agreement.curve.map((a, i) => `${X(G.flops(r.d, r.w, i + 1)).toFixed(1)},${Y(a).toFixed(1)}`).join(' ');
      s += `<polyline points="${pts}" fill="none" stroke="${colors[r.d]}" stroke-width="1.5"/>`;
    }
    return s;
  });
  const legend = G.DEPTHS.map((d, i) => `<rect x="${20 + i * 100}" y="${2 * H + 8}" width="14" height="4" fill="${colors[d]}"/><text x="${38 + i * 100}" y="${2 * H + 13}" font-size="11">depth ${d}</text>`).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${2 * W}" height="${2 * H + 24}" font-family="sans-serif" style="background:#fff">
<rect width="100%" height="100%" fill="#fff"/>
${panels.join('\n')}
${legend}
<text x="${2 * W - 10}" y="${2 * H + 13}" text-anchor="end" font-size="10">held-out agreement (%), clipped to [98.5, 100]; dashed: T1 99.5, T2 99.8</text>
</svg>
`;
  fs.writeFileSync(path.resolve(svgFile), svg);
}
