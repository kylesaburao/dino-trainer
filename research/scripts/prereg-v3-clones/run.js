'use strict';
// Serial job runner for research/prereg-v3-clones.md (§10 steps 3-5). One heavy job at a time.
// Usage: node run.js <phase> [<phase> ...]   phases: demos controls arms tierb measures (or all)
// Every step logs to runs/exp-v3-clones/status.log; each step skips if its output exists, so the
// runner can be restarted. Any non-zero tool exit or failed gate/control STOPs the runner (exit 1).
// Model failures in the arms are data, not stops.
// Env: PYTHON (agreement.py and bc-train --backend python), OMP_NUM_THREADS / VECLIB_MAXIMUM_THREADS.
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const ROOT = path.resolve(__dirname, '../../..');
const G = 'runs/exp-v3-clones';
const GRID = '0,2000,50000,98000,101000,127000,162000,207000,264000,336000,428000,546000,695000,886000,995000';
const MARGIN_LEVELS = '0,2000,50000,98000,101000,207000,546000,886000';
const ARMS = [['A', '6', 'd1w6'], ['B', '32,32', 'd2w32'], ['C', '8,8', 'd2w8']];
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const S = 'research/scripts';
const OPENING = `${S}/prereg-teacher-duck-stuck/opening.js`, SCAN = `${S}/prereg-bc-robustness/scan.js`;
const SURVEY = `${S}/prereg-teacher-duck/survey.js`, DUCKSHARE = `${S}/prereg-teacher-duck-step2/duckshare.js`;
const MARGIN = `${S}/prereg-bc-sweep/margin.js`, AGREE = `${S}/prereg-bc-sweep/agreement.py`;
const PY = process.env.PYTHON || '/Users/kyle/Desktop/dino-game/.venv/bin/python';
const env = { ...process.env, PYTHON: PY, OMP_NUM_THREADS: '2', VECLIB_MAXIMUM_THREADS: '2' };

const abs = p => path.join(ROOT, p);
const exists = p => fs.existsSync(abs(p));
const readJson = p => JSON.parse(fs.readFileSync(abs(p), 'utf8'));
const readLines = p => fs.readFileSync(abs(p), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
fs.mkdirSync(abs(G), { recursive: true });
function status(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  fs.appendFileSync(abs(`${G}/status.log`), line + '\n');
  console.log(line);
}
class Stop extends Error {}
function stop(msg) { status(`STOP: ${msg}`); throw new Stop(msg); }

// Run one command (cwd ROOT), stdout+stderr appended to logFile. Throws Stop on a non-zero exit.
function run(label, cmd, args, logFile) {
  fs.mkdirSync(path.dirname(abs(logFile)), { recursive: true });
  const fd = fs.openSync(abs(logFile), 'a');
  fs.writeSync(fd, `\n### ${new Date().toISOString()} ${cmd} ${args.join(' ')}\n`);
  status(`start ${label}`);
  const t0 = Date.now();
  const r = spawnSync(cmd, args, { cwd: ROOT, env, stdio: ['ignore', fd, fd] });
  fs.closeSync(fd);
  const code = r.status === null ? `signal ${r.signal}` : r.status;
  status(`end ${label} exit ${code} ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (r.status !== 0) stop(`${label} exited ${code} (log ${logFile})`);
}
const node = (label, args, log) => run(label, process.execPath, args, log);

// Scan summary: rows (contiguous from base), footer, failures.
function scanSummary(p, base, n) {
  const L = readLines(p);
  const rows = L.filter(r => r.seed !== undefined);
  const footer = L.some(r => r.footer);
  const seeds = new Set(rows.map(r => r.seed));
  const contiguous = rows.length === n && seeds.size === n && [...seeds].every(s => s >= base && s < base + n);
  return { rows: rows.length, contiguous, footer, fails: rows.filter(r => !r.pass).map(r => ({ seed: r.seed, score: r.score })) };
}
const scanDone = p => exists(p) && readLines(p).some(r => r.footer);

// ---------------------------------------------------------------- §4.1 demos and the gate
function demos() {
  for (const [name, seeds] of [['v3-demos-duck-s1-40', '1-40'], ['v3-demos-duck-s41-60', '41-60']]) {
    const dir = `${G}/${name}`;
    if (!exists(`${dir}/demos.json`)) {
      node(`bc-gen ${name}`, ['trainer/cli.js', 'bc-gen', '--teacher', 'duck', '--game', 'v3', '--seeds', seeds, '--span0', '3000',
        '--span', '5000', '--keep-boring', '10', '--levels', GRID, '--dir', dir], `${dir}/gen.log`);
    }
    node(`gate ${name}`, [`${S}/prereg-v3-clones/gate.js`, dir, '--out', `${dir}/gate.json`], `${dir}/gate.log`);
  }
}

// ---------------------------------------------------------------- §4.4 controls
function controls() {
  const C = `${G}/controls`, log = `${C}/controls.log`;
  const res = {};
  if (!exists(`${C}/pc-open.json`)) node('PC-open', [OPENING, '--teacher', 'duck', '--game', 'v3', '--base', '30003000', '--n', '1000', '--target', '2000', '--out', `${C}/pc-open.json`], log);
  if (!exists(`${C}/pc-post-survey.jsonl`)) node('PC-post survey', [SURVEY, '--solver', 'duck', '--game', 'v3', '--start', '95000', '--cap', '130000', '--base', '30003000', '--n', '1000', '--out', `${C}/pc-post-survey.jsonl`], log);
  if (!exists(`${C}/pc-post-lag.json`)) node('PC-post lag', [OPENING, '--teacher', 'duck', '--game', 'v3', '--start', '95000', '--target', '130000', '--base', '30003000', '--n', '1000', '--out', `${C}/pc-post-lag.json`], log);
  if (!exists(`${C}/pc-b-survey.jsonl`)) node('PC-B survey', [SURVEY, '--solver', 'duck', '--game', 'v3', '--start', '0', '--cap', '1000000', '--base', '30003000', '--n', '1000', '--out', `${C}/pc-b-survey.jsonl`], log);
  if (!scanDone(`${C}/dc-post.jsonl`)) node('DC-post', [SCAN, '--model', 'runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json', '--game', 'v3', '--expect-hash', '65f3546c',
    '--start', '95000', '--target', '130000', '--base', '30003000', '--n', '1000', '--wasm', '--out', `${C}/dc-post.jsonl`], log);

  const op = readJson(`${C}/pc-open.json`);
  res.pcOpen = { passed: op.passed, fatal: op.runs.fatalMid + op.runs.fatalNoMid, maxLag: op.lag.max, ok: op.passed === 1000 && op.runs.fatalMid + op.runs.fatalNoMid === 0 && op.lag.max <= 2 };
  const sv = f => { const r = readLines(f); return { n: r.length, crashes: r.filter(x => x.crashed).length, seeds: new Set(r.map(x => x.seed)).size }; };
  const pp = sv(`${C}/pc-post-survey.jsonl`), pl = readJson(`${C}/pc-post-lag.json`);
  res.pcPost = { survey: pp, lag: { passed: pl.passed, fatal: pl.runs.fatalMid + pl.runs.fatalNoMid, maxLag: pl.lag.max },
    crashOk: pp.n === 1000 && pp.seeds === 1000 && pp.crashes === 0 && pl.passed === 1000, lagOk: pl.lag.max <= 2 };
  const pb = sv(`${C}/pc-b-survey.jsonl`);
  res.pcB = { ...pb, ok: pb.n === 1000 && pb.seeds === 1000 && pb.crashes === 0 };
  const dc = scanSummary(`${C}/dc-post.jsonl`, 30003000, 1000);
  const below = dc.fails.filter(f => f.score < 100000).length, inStep = dc.fails.filter(f => f.score >= 100000 && f.score <= 102000).length;
  const sc = dc.fails.map(f => f.score).sort((a, b) => a - b);
  res.dcPost = { rows: dc.rows, contiguous: dc.contiguous, footer: dc.footer, fails: dc.fails.length, below100k: below, in100to102k: inStep,
    minScore: sc[0], maxScore: sc[sc.length - 1], ok: dc.contiguous && dc.footer && dc.fails.length >= 975 && below === 0 && inStep >= 0.95 * dc.fails.length };
  fs.writeFileSync(abs(`${C}/controls.json`), JSON.stringify(res, null, 1) + '\n');
  status(`controls ${JSON.stringify(res)}`);
  if (!res.pcOpen.ok) stop('PC-open failed');
  if (!res.pcPost.crashOk) stop('PC-post: a crash (or missing rows)');
  if (!res.pcPost.lagOk) stop('PC-post: lag > 2 at v3 speeds (fix the lag definition, record it, then read)');
  if (!res.pcB.ok) stop('PC-B failed');
  if (!res.dcPost.ok) stop('DC-post failed');
}

// ---------------------------------------------------------------- arms
const models = () => ARMS.flatMap(([arm, hidden, shape]) => SEEDS.map(k => ({ arm, hidden, shape, k, name: `v3-bc-duck-${shape}-s${k}` })));
function trainAndScreen(m) {
  const dir = `${G}/${m.name}`, best = `${dir}/best.json`;
  if (!exists(best)) {
    node(`bc-train ${m.name}`, ['trainer/cli.js', 'bc-train', '--backend', 'python', '--hidden', m.hidden, '--seed', String(m.k),
      '--demos', `${G}/v3-demos-duck-s1-40`, '--snapshots', '--wasm', '--dir', dir], `${dir}/train.log`);
  }
  const b = readJson(best);
  status(`tierA ${m.name} ${b.validation.count}/25 min ${b.validation.min}`);
  if (b.validation.count < 25 && !exists(`${G}/opening/${m.name}-val.json`)) {
    node(`tierA replay ${m.name}`, [OPENING, '--model', best, '--val', '--out', `${G}/opening/${m.name}-val.json`], `${G}/opening/${m.name}.log`);
  }
  if (!exists(`${G}/opening/${m.name}.json`)) {
    node(`opening ${m.name}`, [OPENING, '--model', best, '--base', '30003000', '--n', '1000', '--target', '2000', '--out', `${G}/opening/${m.name}.json`], `${G}/opening/${m.name}.log`);
  }
  const ps = `${G}/poststep/${m.name}.jsonl`;
  if (!scanDone(ps)) {
    node(`poststep ${m.name}`, [SCAN, '--model', best, '--start', '95000', '--target', '130000', '--base', '30003000', '--n', '1000', '--wasm', '--out', ps], `${G}/poststep/${m.name}.log`);
  }
  const s = scanSummary(ps, 30003000, 1000);
  if (!s.contiguous || !s.footer) stop(`poststep ${m.name}: ${s.rows} rows, contiguous ${s.contiguous}, footer ${s.footer}`);
  status(`poststep ${m.name} ${1000 - s.fails.length}/1000`);
  if (s.fails.length && !exists(`${G}/poststep/${m.name}-class.json`)) {
    node(`poststep class ${m.name}`, [OPENING, '--model', best, '--start', '95000', '--target', '130000', '--base', '30003000', '--n', '1000',
      '--out', `${G}/poststep/${m.name}-class.json`], `${G}/poststep/${m.name}.log`);
  }
}
function arms() { for (const m of models()) trainAndScreen(m); }

// ---------------------------------------------------------------- tier B
function eligible(m) {
  const b = readJson(`${G}/${m.name}/best.json`), op = readJson(`${G}/opening/${m.name}.json`);
  const ps = scanSummary(`${G}/poststep/${m.name}.jsonl`, 30003000, 1000);
  return b.validation.count === 25 && op.passed === 1000 && ps.fails.length === 0;
}
const tierBModels = () => ARMS.flatMap(([arm]) => models().filter(m => m.arm === arm && eligible(m)).slice(0, 2));
function tierb() {
  for (const m of tierBModels()) {
    const f = `${G}/scans/${m.name}.jsonl`;
    if (!scanDone(f)) node(`tierB ${m.name}`, [SCAN, '--model', `${G}/${m.name}/best.json`, '--base', '30003000', '--n', '1000', '--target', '1000000', '--wasm', '--out', f], `${G}/scans/${m.name}.log`);
    const s = scanSummary(f, 30003000, 1000);
    if (!s.contiguous || !s.footer) stop(`tierB ${m.name}: ${s.rows} rows, contiguous ${s.contiguous}, footer ${s.footer}`);
    status(`tierB ${m.name} ${1000 - s.fails.length}/1000`);
    // Each failure: replay that seed's game from 0 (same game; killer and final DOWN run).
    for (const x of s.fails) {
      const o = `${G}/scans/${m.name}-fail-${x.seed}.json`;
      if (!exists(o)) node(`tierB replay ${m.name} ${x.seed}`, [OPENING, '--model', `${G}/${m.name}/best.json`, '--base', String(x.seed), '--n', '1', '--target', '1000000', '--out', o], `${G}/scans/${m.name}.log`);
    }
  }
}

// ---------------------------------------------------------------- §4.5 measures
function measures() {
  for (const m of models()) {
    const b = readJson(`${G}/${m.name}/best.json`);
    if (b.validation.count >= 20 && !exists(`${G}/duckshare/${m.name}.json`)) {
      node(`duckshare ${m.name}`, [DUCKSHARE, '--model', `${G}/${m.name}/best.json`, '--base', '30003000', '--n', '10', '--out', `${G}/duckshare/${m.name}.json`], `${G}/duckshare/${m.name}.log`);
    }
  }
  for (const m of tierBModels()) {
    if (!exists(`${G}/margin/${m.name}.json`)) node(`margin ${m.name}`, [MARGIN, '--model', `${G}/${m.name}/best.json`, '--levels', MARGIN_LEVELS, '--out', `${G}/margin/${m.name}.json`], `${G}/margin/${m.name}.log`);
  }
  const dirs = models().map(m => `${G}/${m.name}`).filter(d => !exists(`${d}/agreement.json`));
  if (dirs.length) run('agreement', PY, [AGREE, '--split-x4', '2.5', `${G}/v3-demos-duck-s41-60/demos.bin`, ...dirs], `${G}/agree/agreement.log`);
}

const PHASES = { demos, controls, arms, tierb, measures };
let list = process.argv.slice(2);
if (list.includes('all')) list = ['demos', 'controls', 'arms', 'tierb', 'measures'];
status(`runner pid ${process.pid}: ${list.join(' ')}`);
try {
  for (const p of list) { if (!PHASES[p]) stop(`unknown phase ${p}`); status(`phase ${p}`); PHASES[p](); }
  status('runner done');
} catch (e) {
  if (!(e instanceof Stop)) status(`STOP: error ${e.stack}`);
  process.exit(1);
}
