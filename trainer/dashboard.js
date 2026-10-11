#!/usr/bin/env node
'use strict';
// Live training dashboard: `node trainer/dashboard.js [--port 8080] [--runs runs]`
// then open http://127.0.0.1:8080. Read-only: it never writes into the runs
// directory, so it is safe to run alongside (and independently of) any trainer.
//
// Runs: every directory under --runs, plus run directories one level down
// (runs/<group>/<run>, shown as "group/run"). A directory is a run if it holds any of
// metrics.jsonl, checkpoint.json, best.json, train.lock, bc-metrics.jsonl, demos.json or a
// bc-gen temp file (demos.bin.<pid>.tmp). A top-level directory whose only training
// output is in its subdirectories is a group, not a run; any other top-level directory
// is listed as before.
//   ES runs (train): metrics.jsonl by generation, checkpoint.json, best.json, assess.jsonl (+ legacy evals.jsonl),
//     train.lock and the newest *.log; unchanged from the ES-only dashboard.
//   Behaviour-cloning runs (bc-train; bc-metrics.jsonl, or best.json whose config.method
//     says behaviour cloning): per-epoch loss, recall for every class key in the rows, and
//     seconds per epoch; the demos manifest (config.demos, or the bc-train log line while
//     training); best.json's validation; assess.jsonl / evals.jsonl; snapshots/epoch-NN.json. Teacher =
//     config.teacherName / the manifest's teacherName, else teacher-jump. Distilled runs
//     (config.distillFrom) and quantized runs (config.quant: bc-train --quant-bits, or
//     bc-quantize with config.quantFrom) also carry bc.distill / bc.quant (teacher model
//     path, sha256, tau, mode, alpha; bits, scheme, levels, source; k per layer for
//     scheme pow2), live from the log line.
//   Demos runs (bc-gen; demos.json only): the manifest summary and teacher playback.
//     DAgger rollouts (bc-gen --student; manifest source 'dagger'): the student is named,
//     its crashes are shown as student crashes (not solver crashes), and playback offers the
//     student model first ('student', from the manifest's student.path). A --score-window is
//     shown. BC runs trained with --extra-demos list each extra manifest, and
//     --transition-weight with its row counts.
// Running = train.lock with a live pid (train, bc-train), or a bc-gen temp file
// demos.bin.<pid>.tmp with a live pid (bc-gen takes no lock). A nested run with no *.log
// of its own shows <group>/<run>.log (where detached runs often log).
//
// Playback plays the run's weights with core.js + policy.js: ES latest checkpoint or best;
// BC best or any snapshot epoch. "Teacher" playback runs the solver itself (test/solver.js,
// every create*Solver it exports at page load) through /teacher.js, a shim that wraps the
// file's source and maps require('../core.js') to the page's DinoCore (with _internal).
// Only this local dev page sees the solver; game.js and index.html never load it.
// Parsed best.json / checkpoint.json / demos.json are cached by mtime and size, without
// their weights, so polling dozens of runs stays cheap.
//
// Scans (trainer/scans.js): every scan.js / survey.js JSONL under --runs, any depth up to 4,
// with games done, passes, failure scores and a Clopper-Pearson interval on the failure
// rate; done (footer) / running / stalled from the pid recorded in the file, a <name>.pid
// file, or the mtime. Files are tailed incrementally. /api/scans lists them; a run's detail
// carries the scans whose header model is that run's best.json.
const http = require('http');
const fs = require('fs');
const path = require('path');
const { pidAlive, ROOT, ASSESS_FILES, readAssessLines } = require('./store.js');
const { listScans } = require('./scans.js');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const PORT = Number(opt('--port', 8080));
const RUNS = path.resolve(opt('--runs', path.join(ROOT, 'runs')));
const PAGE = path.join(__dirname, 'dashboard.html');
const SOLVER = path.join(ROOT, 'test', 'solver.js');

const readJson = (f) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } };
const readLines = (f) => { try { return fs.readFileSync(f, 'utf8').split('\n').filter(Boolean); } catch (e) { return []; } };
const mtime = (f) => { try { return fs.statSync(f).mtimeMs; } catch (e) { return 0; } };
const listDir = (d) => { try { return fs.readdirSync(d, { withFileTypes: true }); } catch (e) { return []; } };

// Parsed JSON, re-read only when a file's mtime or size changes; `strip` drops bulky
// keys (weights) the list and detail views never use.
const jsonCache = new Map();
function cachedJson(file, strip) {
  let st;
  try { st = fs.statSync(file); } catch (e) { jsonCache.delete(file); return null; }
  const key = `${st.mtimeMs}:${st.size}`;
  const hit = jsonCache.get(file);
  if (hit && hit.key === key) return hit.value;
  let value = readJson(file);
  if (value && strip) { value = { ...value }; for (const k of strip) delete value[k]; }
  jsonCache.set(file, { key, value });
  return value;
}
const bestSummary = (dir) => cachedJson(path.join(dir, 'best.json'), ['params']);
const ckSummary = (dir) => cachedJson(path.join(dir, 'checkpoint.json'), ['theta', 'adam', 'best']);

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------
const MARKERS = ['metrics.jsonl', 'checkpoint.json', 'best.json', 'train.lock', 'bc-metrics.jsonl', 'demos.json'];
const GEN_TMP = /^demos\.bin\.(\d+)\.tmp$/;
const NOT_RUNS = new Set(['snapshots', 'export']);   // per-run subdirectories, never runs

const isRunDir = (entries) => entries.some(e => e.isFile() && (MARKERS.includes(e.name) || GEN_TMP.test(e.name)));

function listRunNames() {
  const names = [];
  for (const top of listDir(RUNS)) {
    if (!top.isDirectory()) continue;
    const entries = listDir(path.join(RUNS, top.name));
    const children = entries.filter(e => e.isDirectory() && !NOT_RUNS.has(e.name))
      .filter(e => isRunDir(listDir(path.join(RUNS, top.name, e.name)))).map(e => `${top.name}/${e.name}`);
    if (isRunDir(entries) || !children.length) names.push(top.name);   // plain top-level dirs as before
    names.push(...children);
  }
  return names;
}

const BC_METHOD = /behaviou?r[ -]clon/i;
function runKind(name, best) {
  const dir = path.join(RUNS, name);
  const entries = listDir(dir);
  const has = (f) => entries.some(e => e.name === f);
  if (has('bc-metrics.jsonl') || (best && best.config && BC_METHOD.test(best.config.method || ''))) return 'bc';
  if (has('metrics.jsonl') || has('checkpoint.json') || best) return 'es';
  if (has('demos.json') || entries.some(e => GEN_TMP.test(e.name))) return 'demos';
  if (has('train.lock') && trainLine(name)) return 'bc';   // bc-train before its first epoch
  return 'es';   // a plain directory: the ES view, exactly as before
}

// A running bc-gen (no lock): demos.bin.<pid>.tmp with a live pid.
function genProcess(dir) {
  for (const e of listDir(dir)) {
    const m = GEN_TMP.exec(e.name);
    if (m && pidAlive(Number(m[1]))) {
      let t = 0;
      try { t = fs.statSync(path.join(dir, e.name)).birthtimeMs; } catch (err) { /* finished meanwhile */ }
      return { pid: Number(m[1]), startedAt: new Date(t || Date.now()).toISOString() };
    }
  }
  return null;
}

function newestLogIn(dir, name) {
  const logs = listDir(dir).filter(e => e.isFile() && e.name.endsWith('.log')).map(e => path.join(dir, e.name));
  if (logs.length) return logs.sort((a, b) => mtime(b) - mtime(a))[0];
  if (name && name.includes('/')) {   // nested run logging next to itself: <group>/<run>.log
    const f = path.join(path.dirname(dir), path.basename(dir) + '.log');
    if (fs.existsSync(f)) return f;
  }
  return null;
}
const newestLog = (name) => newestLogIn(path.join(RUNS, name), name);

// ---------------------------------------------------------------------------
// ES runs (unchanged fields, plus kind)
// ---------------------------------------------------------------------------
function esInfo(name) {
  const dir = path.join(RUNS, name);
  const lock = readJson(path.join(dir, 'train.lock'));
  const ck = ckSummary(dir);
  const best = bestSummary(dir);
  const metrics = readLines(path.join(dir, 'metrics.jsonl'));
  const last = metrics.length ? JSON.parse(metrics[metrics.length - 1]) : null;
  const evals = readAssessLines(dir).map(l => { const e = JSON.parse(l); delete e.scores; return e; });
  const updated = Math.max(...['metrics.jsonl', 'checkpoint.json', 'train.lock'].map(f => mtime(path.join(dir, f))));
  return {
    name,
    kind: 'es',
    running: !!(lock && pidAlive(lock.pid)),
    pid: lock ? lock.pid : null,
    startedAt: lock ? lock.startedAt : null,
    generation: last ? last.generation : ck ? ck.generation : 0,
    checkpoint: ck ? { generation: ck.generation, savedAt: ck.savedAt, reason: ck.reason, cumulativeMs: ck.cumulativeMs, coreHash: ck.coreHash } : null,
    config: ck ? ck.config : best ? best.config : null,
    best: best ? { generation: best.generation, validation: best.validation } : null,
    heldOut: (ck && ck.heldOut) || null,
    lastEval: evals.length ? evals[evals.length - 1] : null,
    evals,
    last,
    updated,
  };
}

// ---------------------------------------------------------------------------
// Behaviour-cloning and demos runs
// ---------------------------------------------------------------------------
// The bc-train start line, for a run still training (no best.json yet):
// "bc-train: dir=... backend=python layers=44-8-8-3 epochs 15 ... demos runs/x/demos.json (...".
function trainLineIn(dir, name) {
  const f = newestLogIn(dir, name);
  if (!f) return null;
  const line = readLines(f).find(l => l.includes('bc-train: dir='));
  if (!line) return null;
  const g = (re) => { const m = re.exec(line); return m ? m[1] : null; };
  return { backend: g(/backend=(\S+)/), layers: g(/layers=([\d-]+)/), epochs: Number(g(/ epochs (\d+)/)) || null, demos: g(/ demos (\S+)/),
    distill: g(/ distill=(\S+)/) ? { mode: g(/ distill=(\S+)/), tau: g(/ tau=(\S+)/) !== null ? Number(g(/ tau=(\S+)/)) : null,
      alpha: g(/ alpha=(\S+)/) !== null ? Number(g(/ alpha=(\S+)/)) : null, from: g(/ from=(\S+)/) } : null,
    quant: g(/ quant=int(\d+)\//) ? { bits: Number(g(/ quant=int(\d+)\//)), scheme: g(/ quant=int\d+\/(\S+)/), mode: 'qat' } : null };
}
const trainLine = (name) => trainLineIn(path.join(RUNS, name), name);

// config.demos is relative to the repo root that trained it; with --runs pointing at
// another checkout's runs/, try that checkout too.
function resolveDemos(p) {
  if (!p) return null;
  for (const f of [path.resolve(ROOT, p), path.resolve(path.dirname(RUNS), p)]) if (fs.existsSync(f)) return f;
  return null;
}

function demosSummary(file) {
  const m = file && cachedJson(file, ['episodes']);
  if (!m) return null;
  return {
    file: path.relative(path.dirname(RUNS), file),
    rows: m.rows, counts: m.counts, teacher: m.teacher || null, teacherName: m.teacherName || null,
    solverCrashes: m.solverCrashes, coreHash: m.coreHash, gameVersion: m.gameVersion, createdAt: m.createdAt,
    seeds: Array.isArray(m.seeds) && m.seeds.length ? `${m.seeds[0]}-${m.seeds[m.seeds.length - 1]}` : null,
    levels: m.levels, span0: m.span0, span: m.span, keepBoring: m.keepBoring, sha256: m.sha256,
    // Release-fix keys (bc-gen --score-window / --student; absent in older manifests).
    scoreWindow: m.scoreWindow || null, source: m.source || 'teacher', student: m.student || null,
    studentCrashes: m.studentCrashes, undefinedAir: m.undefinedAir,
  };
}

// The student model file of a DAgger demos manifest (path relative to the repo that made it).
function studentFile(dm) {
  return dm && dm.source === 'dagger' && dm.student && dm.student.path ? resolveDemos(dm.student.path) : null;
}

function snapshotList(dir) {
  return listDir(path.join(dir, 'snapshots')).filter(e => e.isFile() && /^epoch-\d+\.json$/.test(e.name))
    .map(e => ({ name: e.name.slice(0, -5), epoch: Number(e.name.slice(6, -5)) })).sort((a, b) => a.epoch - b.epoch);
}

// Distillation (bc-train --distill-from) and quantization (--quant-bits, bc-quantize):
// from best.json's config, or from the bc-train start line while training.
function distillSummary(cfg, live) {
  if (cfg && cfg.distillFrom) {
    return { mode: cfg.distillMode || 'soft', tau: cfg.tau ?? null, alpha: cfg.alpha ?? null, from: cfg.distillFrom.path,
      paramsSha256: cfg.distillFrom.paramsSha256 || null, layers: cfg.distillFrom.layers || null, teacherName: cfg.distillFrom.teacherName || null };
  }
  return live && live.distill ? { ...live.distill, paramsSha256: null, layers: null, teacherName: null } : null;
}
function quantSummary(cfg, live) {
  if (cfg && cfg.quant) {
    const q = cfg.quant;
    return { bits: q.bits, scheme: q.scheme, mode: q.mode, levels: q.levels ?? null, biases: q.biases || null,
      from: cfg.quantFrom ? cfg.quantFrom.path : null, fromSha256: cfg.quantFrom ? cfg.quantFrom.paramsSha256 : null,
      ...(Array.isArray(q.k) ? { k: q.k } : {}) };
  }
  return live && live.quant ? { ...live.quant, levels: null, biases: live.quant.scheme === 'pow2' ? 'int' : 'float', from: null, fromSha256: null } : null;
}

function bcInfo(name, kind) {
  const dir = path.join(RUNS, name);
  const lock = readJson(path.join(dir, 'train.lock'));
  const gen = !(lock && pidAlive(lock.pid)) && genProcess(dir);
  const best = bestSummary(dir);
  const cfg = (best && best.config) || null;
  const rows = readLines(path.join(dir, 'bc-metrics.jsonl')).map(l => JSON.parse(l));
  const last = rows.length ? rows[rows.length - 1] : null;
  const evals = readAssessLines(dir).map(l => { const e = JSON.parse(l); delete e.scores; return e; });
  const live = !cfg && kind === 'bc' ? trainLine(name) : null;
  const demosFile = kind === 'demos' ? path.join(dir, 'demos.json') : resolveDemos((cfg && cfg.demos) || (live && live.demos));
  const demos = demosSummary(demosFile);
  const layers = best ? best.layers : live && live.layers ? live.layers.split('-').map(Number) : null;
  const updated = Math.max(...['bc-metrics.jsonl', 'best.json', 'train.lock', 'demos.json', ...ASSESS_FILES].map(f => mtime(path.join(dir, f))),
    ...listDir(dir).filter(e => GEN_TMP.test(e.name)).map(e => mtime(path.join(dir, e.name))));
  // teacherName is additive (duck generation); runs without it are the jump generation.
  const teacher = (cfg && cfg.teacherName) || (best && best.teacherName) || (demos && demos.teacherName) || 'teacher-jump';
  return {
    name,
    kind,
    running: !!((lock && pidAlive(lock.pid)) || gen),
    pid: lock && pidAlive(lock.pid) ? lock.pid : gen ? gen.pid : lock ? lock.pid : null,
    startedAt: lock && pidAlive(lock.pid) ? lock.startedAt : gen ? gen.startedAt : lock ? lock.startedAt : null,
    process: lock && pidAlive(lock.pid) ? 'bc-train' : gen ? 'bc-gen' : null,
    generation: null,
    config: cfg,
    best: best ? { generation: best.generation, validation: best.validation, coreHash: best.coreHash, gameVersion: best.gameVersion } : null,
    lastEval: evals.length ? evals[evals.length - 1] : null,
    evals,
    bc: {
      teacher,
      teacherSource: (demos && demos.teacher) || (cfg && /test\/solver\.js/.test(cfg.method || '') ? 'test/solver.js' : null),
      backend: (cfg && cfg.backend) || (last && last.backend) || (live && live.backend) ||
        (cfg && /trainB\.py/.test(cfg.trainer || '') ? 'python (trainB.py)' : null),
      layers,
      epochs: (cfg && cfg.epochs) || (live && live.epochs) || null,
      epochsDone: rows.length,
      demos,
      demosRef: (cfg && cfg.demos) || (live && live.demos) || null,
      // bc-train --extra-demos / --transition-weight (absent for runs that did not use them).
      extraDemos: ((cfg && cfg.extraDemos) || []).map(e => ({ ref: e.demos, summary: demosSummary(resolveDemos(e.demos)) })),
      transitionWeight: (cfg && cfg.transitionWeight) || null,
      transitions: (cfg && cfg.transitions) || null,
      // DAgger demos: the student that played the rollouts, if its best.json is on this computer.
      student: demos && demos.source === 'dagger' ? { ...(demos.student || {}), available: !!studentFile(demos) } : null,
      snapshots: snapshotList(dir),
      distill: distillSummary(cfg, live),
      quant: quantSummary(cfg, live),
    },
    last,
    updated,
  };
}

function runInfo(name) {
  const kind = runKind(name, bestSummary(path.join(RUNS, name)));
  return kind === 'es' ? esInfo(name) : bcInfo(name, kind);
}

function listRuns() {
  return listRunNames().map(runInfo).sort((a, b) => (b.running - a.running) || (b.updated - a.updated));
}

// Metrics for charts; long runs are thinned to ~maxPoints, always keeping
// validation rows (unless validation is folded into every generation: valFrames > 0)
// and the latest row. BC runs: every bc-metrics.jsonl row (one per epoch).
function runMetrics(name, kind, maxPoints = 1500) {
  if (kind !== 'es') return readLines(path.join(RUNS, name, 'bc-metrics.jsonl')).map(l => JSON.parse(l));
  const rows = readLines(path.join(RUNS, name, 'metrics.jsonl')).map(l => JSON.parse(l));
  if (rows.length <= maxPoints) return rows;
  const step = Math.ceil(rows.length / maxPoints);
  return rows.filter((r, i) => i % step === 0 || (r.valMedian !== null && !r.valFrames) || i === rows.length - 1);
}

function logTail(name, n = 40) {
  const f = newestLog(name);
  return f ? readLines(f).slice(-n) : [];
}

// Weights for live playback: the latest checkpoint's center ('current', falling back to
// best.json when there is no checkpoint), best.json, or a BC snapshot ('epoch-NN').
function runModel(name, which) {
  const dir = path.join(RUNS, name);
  if (/^epoch-\d+$/.test(which)) {
    if (!snapshotList(dir).some(s => s.name === which)) return null;
    const s = readJson(path.join(dir, 'snapshots', which + '.json'));
    const best = bestSummary(dir);
    const demos = best ? null : demosSummary(resolveDemos((trainLine(name) || {}).demos));
    return s && { format: 'dino-mlp-v1', layers: s.layers, actionRepeat: 1, gameVersion: (best && best.gameVersion) || (demos && demos.gameVersion) || 'v2',
      params: s.params, generation: null, epoch: s.epoch, which };
  }
  if (which === 'student') {
    const f = studentFile(demosSummary(path.join(dir, 'demos.json')));
    const b = f && readJson(f);
    return b && { format: b.format, layers: b.layers, actionRepeat: b.actionRepeat, gameVersion: b.gameVersion || 'v1', params: b.params, generation: null, which, student: true };
  }
  if (which === 'best') {
    const b = readJson(path.join(dir, 'best.json'));
    return b && { format: b.format, layers: b.layers, actionRepeat: b.actionRepeat, gameVersion: b.gameVersion || 'v1', params: b.params, generation: b.generation, which };
  }
  const ck = readJson(path.join(dir, 'checkpoint.json'));
  if (!ck) return which === 'current' ? runModel(name, 'best') : null;   // a run with only best.json
  return { layers: ck.layers, actionRepeat: ck.config.actionRepeat, gameVersion: ck.config.gameVersion || 'v1', params: ck.theta, generation: ck.generation, which: 'current' };
}

// The solver as a classic script for this page only. Read on every request, so solver
// factories added to test/solver.js appear on the next page load; its require of
// core.js resolves to the page's DinoCore (which carries _internal), policy.js to DinoPolicy.
function teacherShim() {
  const src = fs.readFileSync(SOLVER, 'utf8');
  return `// Generated by trainer/dashboard.js from test/solver.js (dashboard-only teacher shim).
(function (root) {
  var module = { exports: {} };
  function require(p) {
    var base = String(p).split('/').pop().replace(/\\.js$/, '');
    if (base === 'core') return root.DinoCore;
    if (base === 'policy') return root.DinoPolicy;
    throw new Error('teacher shim: test/solver.js requires ' + p + ', which the dashboard does not provide');
  }
  (function (require, module, exports) {
${src}
  })(require, module, module.exports);
  root.DinoTeacher = module.exports;
})(globalThis);
`;
}

// The page runs the real game code, served straight from the repo.
const SHARED = { '/core.js': 'core.js', '/policy.js': 'policy.js', '/render.js': 'render.js', '/netview.js': 'netview.js' };

const send = (res, code, type, body) => {
  res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store' });
  res.end(body);
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const m = url.pathname.match(/^\/api\/runs\/([^/]+)(\/model)?$/);
  try {
    if (url.pathname === '/') return send(res, 200, 'text/html; charset=utf-8', fs.readFileSync(PAGE));
    if (SHARED[url.pathname]) return send(res, 200, 'text/javascript; charset=utf-8', fs.readFileSync(path.join(ROOT, SHARED[url.pathname])));
    if (url.pathname === '/teacher.js') return send(res, 200, 'text/javascript; charset=utf-8', teacherShim());
    if (url.pathname === '/api/runs') return send(res, 200, 'application/json', JSON.stringify(listRuns()));
    if (url.pathname === '/api/scans') return send(res, 200, 'application/json', JSON.stringify(listScans(RUNS, new Set(listRunNames()))));
    if (m) {
      const name = decodeURIComponent(m[1]);   // "run" or "group/run" (sent as group%2Frun)
      if (!listRunNames().includes(name)) return send(res, 404, 'text/plain', 'no such run');
      if (m[2]) {
        const w = url.searchParams.get('which') || '';
        const model = runModel(name, w === 'best' || w === 'student' || /^epoch-\d+$/.test(w) ? w : 'current');
        return model ? send(res, 200, 'application/json', JSON.stringify(model)) : send(res, 404, 'text/plain', 'no weights yet');
      }
      const info = runInfo(name);
      const scans = listScans(RUNS, new Set(listRunNames())).filter(s => s.owner === name);
      return send(res, 200, 'application/json', JSON.stringify({ info, metrics: runMetrics(name, info.kind), log: logTail(name), scans }));
    }
    send(res, 404, 'text/plain', 'not found');
  } catch (e) {
    send(res, 500, 'text/plain', String(e && e.message));
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`dino training dashboard: http://127.0.0.1:${PORT}  (watching ${path.relative(process.cwd(), RUNS) || RUNS})`);
});
