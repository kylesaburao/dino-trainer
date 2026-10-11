'use strict';
// Summary of research/prereg-v3-clones.md (§4.5 item 5 per-seed record, §5 classes, §6 tests).
// Reads the outputs of run.js under runs/exp-v3-clones/ and writes summary.json there.
// Failure classes (§5), by the failure's score: < 2,000 from the opening classes (stuck: final DOWN
// run mid and >= 20 frames; early: killed by ptero-mid while not holding DOWN; other), 2,000-100,000
// sub-100k-other, 100,000-102,000 step, 102,000-130,000 post-step-other, > 130,000 late.
// Usage: node summarize.js
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const ROOT = path.resolve(__dirname, '../../..');
const G = path.join(ROOT, 'runs/exp-v3-clones');
const ARMS = [['A', 'd1w6'], ['B', 'd2w32'], ['C', 'd2w8']];
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const rd = p => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null);
const lines = p => (fs.existsSync(p) ? fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : null);
const sha = params => crypto.createHash('sha256').update(JSON.stringify(Array.from(params))).digest('hex');

function openingClass(f) {
  if (f.stuck) return 'opening-stuck';
  if (f.killer === 'ptero-mid' && !f.finalDuck) return 'opening-early';
  return 'opening-other';
}
function classOf(f) {
  if (f.score < 2000) return openingClass(f);
  if (f.score < 100000) return 'sub-100k-other';
  if (f.score <= 102000) return 'step';
  if (f.score <= 130000) return 'post-step-other';
  return 'late';
}
const brief = f => ({ seed: f.seed, score: f.score, killer: f.killer, finalDuck: f.finalDuck, class: classOf(f) });

function scanFails(p) {
  const L = lines(p);
  if (!L) return null;
  const rows = L.filter(r => r.seed !== undefined);
  return { rows: rows.length, footer: L.some(r => r.footer), fails: rows.filter(r => !r.pass) };
}

const models = [];
for (const [arm, shape] of ARMS) {
  for (const k of SEEDS) {
    const name = `v3-bc-duck-${shape}-s${k}`;
    const b = rd(path.join(G, name, 'best.json'));
    if (!b) continue;
    const rec = { arm, shape, seed: k, name, paramsSha256: sha(b.params).slice(0, 12), finalLoss: b.config.final.loss, recall: b.config.final.recall,
      trainSeconds: b.config.trainSeconds, tierA: { count: b.validation.count, min: b.validation.min } };
    const classes = new Set();
    const val = rd(path.join(G, 'opening', `${name}-val.json`));
    if (val) { rec.tierA.fails = val.fails.map(brief); val.fails.forEach(f => classes.add(classOf(f))); }
    const op = rd(path.join(G, 'opening', `${name}.json`));
    if (op) {
      const cls = op.fails.map(openingClass);
      rec.opening = { passed: op.passed, stuck: cls.filter(c => c === 'opening-stuck').length, early: cls.filter(c => c === 'opening-early').length,
        other: cls.filter(c => c === 'opening-other').length, maxLag: op.lag.max,
        otherKillers: op.fails.filter(f => openingClass(f) === 'opening-other').map(f => `${f.killer}@${f.score}`).slice(0, 10) };
      cls.forEach(c => classes.add(c));
    }
    const ps = scanFails(path.join(G, 'poststep', `${name}.jsonl`));
    if (ps) {
      rec.postStep = { passed: ps.rows - ps.fails.length, rows: ps.rows, failScores: ps.fails.map(f => f.score).sort((a, b) => a - b) };
      const pc = rd(path.join(G, 'poststep', `${name}-class.json`));
      if (pc) {
        rec.postStep.classReplay = { passed: pc.passed, fails: pc.fails.map(brief) };
        const scanSeeds = ps.fails.map(f => f.seed).sort().join(','), repSeeds = pc.fails.map(f => f.seed).sort().join(',');
        rec.postStep.replayMatchesScan = scanSeeds === repSeeds;
        pc.fails.forEach(f => classes.add(classOf(f)));
      } else ps.fails.forEach(f => classes.add(classOf(f)));
      const byClass = {};
      for (const f of (pc ? pc.fails : ps.fails)) { const c = classOf(f); byClass[c] = (byClass[c] || 0) + 1; }
      rec.postStep.byClass = byClass;
    }
    const tb = scanFails(path.join(G, 'scans', `${name}.jsonl`));
    if (tb) {
      rec.tierB = { passed: tb.rows - tb.fails.length, rows: tb.rows, fails: tb.fails.map(f => {
        const r = rd(path.join(G, 'scans', `${name}-fail-${f.seed}.json`));
        const x = r && r.fails[0] ? brief(r.fails[0]) : { seed: f.seed, score: f.score, class: classOf(f) };
        x.scanScore = f.score;
        return x;
      }) };
      rec.tierB.fails.forEach(f => classes.add(f.class));
    }
    const ds = rd(path.join(G, 'duckshare', `${name}.json`));
    if (ds) rec.duckShare = { share: ds.duckShare, passed: ds.passed, met: ds.met, ducked: ds.ducked, jumped: ds.jumped, mixed: ds.mixed, crashed: ds.crashed, downAir: ds.downAir };
    const mg = rd(path.join(G, 'margin', `${name}.json`));
    if (mg) rec.margin = { min: mg.minMargin, at: mg.at, p01: mg.p01, median: mg.median, crashes: mg.crashes, games: mg.games };
    const ag = rd(path.join(G, name, 'agreement.json'));
    if (ag && ag.epochs.length) {
      const e = ag.epochs[ag.epochs.length - 1];
      rec.agreement = { epoch: e.epoch, overall: e.agreement, recallNone: e.recallNone, recallJump: e.recallJump, recallDuck: e.recallDuck, duckPredShare: e.duckPredShare,
        le: e.split && e.split.le, gt: e.split && e.split.gt, perEpoch: ag.epochs.map(r => [r.epoch, r.agreement, r.split && r.split.le && r.split.le.agreement, r.split && r.split.gt && r.split.gt.agreement]) };
    }
    rec.classes = [...classes].sort();
    rec.clean = rec.tierA.count === 25 && !!op && op.passed === 1000 && !!ps && ps.rows === 1000 && ps.fails.length === 0 && (!tb || tb.fails.length === 0);
    rec.cleanKind = rec.clean ? (tb ? 'clean (tier B)' : 'clean (A + opening + post-step)') : 'unclean';
    models.push(rec);
  }
}

// Exact binomial / McNemar / Fisher
const lchoose = (n, k) => { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; };
const binomUpper = (n, k) => { let p = 0; for (let i = k; i <= n; i++) p += Math.exp(lchoose(n, i) - n * Math.LN2); return Math.min(1, p); };
function mcnemar(x, y) {
  let b = 0, c = 0;
  for (const k of SEEDS) {
    const a = models.find(m => m.arm === x && m.seed === k), d = models.find(m => m.arm === y && m.seed === k);
    if (!a || !d) continue;
    if (a.clean && !d.clean) b++;
    if (!a.clean && d.clean) c++;
  }
  const n = b + c;
  return { pair: `${x} vs ${y}`, b_xCleanOnly: b, c_yCleanOnly: c, pOneSided_xBetter: n ? binomUpper(n, b) : 1, pOneSided_yBetter: n ? binomUpper(n, c) : 1,
    pTwoSided: n ? Math.min(1, 2 * binomUpper(n, Math.max(b, c))) : 1 };
}
function fisher(a, n1, c, n2) { // two-sided, a of n1 vs c of n2
  const K = a + c, N = n1 + n2;
  const pmf = x => Math.exp(lchoose(n1, x) + lchoose(n2, K - x) - lchoose(N, K));
  const p0 = pmf(a); let p = 0;
  for (let x = Math.max(0, K - n2); x <= Math.min(K, n1); x++) { const q = pmf(x); if (q <= p0 * (1 + 1e-9)) p += q; }
  return Math.min(1, p);
}
function cp(k, n) { // two-sided 95 % Clopper-Pearson by bisection
  const cdf = (p, kk) => { let s = 0; for (let i = 0; i <= kk; i++) s += Math.exp(lchoose(n, i) + i * Math.log(p) + (n - i) * Math.log1p(-p)); return s; };
  const bis = f => { let lo = 0, hi = 1; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (f(m)) lo = m; else hi = m; } return lo; };
  const lo = k === 0 ? 0 : bis(p => 1 - cdf(p, k - 1) < 0.025);
  const hi = k === n ? 1 : bis(p => cdf(p, k) > 0.025);
  return [lo, hi];
}
const arms = {};
for (const [arm, shape] of ARMS) {
  const ms = models.filter(m => m.arm === arm);
  if (!ms.length) continue;
  const clean = ms.filter(m => m.clean).length;
  const classSeeds = {};
  for (const m of ms) for (const c of m.classes) classSeeds[c] = (classSeeds[c] || 0) + 1;
  const openingFailSeeds = ms.filter(m => m.opening && m.opening.passed < 1000).length;
  arms[arm] = { shape, models: ms.length, clean, cp95: cp(clean, ms.length), classSeeds, openingFailSeeds,
    tierB: ms.filter(m => m.tierB).map(m => m.name) };
}
const tests = { mcnemar: [mcnemar('A', 'B'), mcnemar('A', 'C'), mcnemar('B', 'C')] };
if (arms.B) tests.fisherOpeningB_vs_v2d2w32 = { v3: `${arms.B.openingFailSeeds}/${arms.B.models}`, v2: '16/20', p: fisher(arms.B.openingFailSeeds, arms.B.models, 16, 20), note: 'descriptive' };
if (arms.A) tests.fisherOpeningA_vs_v2d1w6 = { v3: `${arms.A.openingFailSeeds}/${arms.A.models}`, v2: '0/10', p: fisher(arms.A.openingFailSeeds, arms.A.models, 0, 10), note: 'descriptive' };
const out = { generated: new Date().toISOString(), arms, tests, models };
fs.writeFileSync(path.join(G, 'summary.json'), JSON.stringify(out, null, 1) + '\n');
for (const m of models) {
  console.log([m.name, m.paramsSha256, m.finalLoss.toFixed(5), `A ${m.tierA.count}/25`, m.opening ? `open ${m.opening.passed} s${m.opening.stuck} e${m.opening.early} o${m.opening.other} lag${m.opening.maxLag}` : '-',
    m.postStep ? `post ${m.postStep.passed}` : '-', m.tierB ? `B ${m.tierB.passed}` : '', m.cleanKind, m.classes.join('+')].join(' | '));
}
console.log(JSON.stringify({ arms, tests }, null, 1));
