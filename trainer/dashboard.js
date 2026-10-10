#!/usr/bin/env node
'use strict';
// Live training dashboard: `node trainer/dashboard.js [--port 8080] [--runs runs]`
// then open http://127.0.0.1:8080. Read-only: it only reads each run directory's
// metrics.jsonl, checkpoint.json, best.json, evals.jsonl, train.lock and *.log,
// so it is safe to run alongside (and independently of) a trainer.
const http = require('http');
const fs = require('fs');
const path = require('path');
const { pidAlive, ROOT } = require('./store.js');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const PORT = Number(opt('--port', 8080));
const RUNS = path.resolve(opt('--runs', path.join(ROOT, 'runs')));
const PAGE = path.join(__dirname, 'dashboard.html');

const readJson = (f) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } };
const readLines = (f) => { try { return fs.readFileSync(f, 'utf8').split('\n').filter(Boolean); } catch (e) { return []; } };
const mtime = (f) => { try { return fs.statSync(f).mtimeMs; } catch (e) { return 0; } };

function runInfo(name) {
  const dir = path.join(RUNS, name);
  const lock = readJson(path.join(dir, 'train.lock'));
  const ck = readJson(path.join(dir, 'checkpoint.json'));
  const best = readJson(path.join(dir, 'best.json'));
  const metrics = readLines(path.join(dir, 'metrics.jsonl'));
  const last = metrics.length ? JSON.parse(metrics[metrics.length - 1]) : null;
  const evals = readLines(path.join(dir, 'evals.jsonl')).map(l => JSON.parse(l));
  const updated = Math.max(...['metrics.jsonl', 'checkpoint.json', 'train.lock'].map(f => mtime(path.join(dir, f))));
  return {
    name,
    running: !!(lock && pidAlive(lock.pid)),
    pid: lock ? lock.pid : null,
    startedAt: lock ? lock.startedAt : null,
    generation: last ? last.generation : ck ? ck.generation : 0,
    checkpoint: ck ? { generation: ck.generation, savedAt: ck.savedAt, reason: ck.reason, cumulativeMs: ck.cumulativeMs, coreHash: ck.coreHash } : null,
    config: ck ? ck.config : best ? best.config : null,
    best: best ? { generation: best.generation, validation: best.validation } : null,
    heldOut: (ck && ck.heldOut) || null,
    lastEval: evals.length ? evals[evals.length - 1] : null,
    last,
    updated,
  };
}

function listRuns() {
  let names = [];
  try { names = fs.readdirSync(RUNS).filter(n => fs.statSync(path.join(RUNS, n)).isDirectory()); } catch (e) { /* no runs yet */ }
  return names.map(runInfo).sort((a, b) => (b.running - a.running) || (b.updated - a.updated));
}

// Metrics for charts; long runs are thinned to ~maxPoints, always keeping
// validation rows and the latest row.
function runMetrics(name, maxPoints = 1500) {
  const rows = readLines(path.join(RUNS, name, 'metrics.jsonl')).map(l => JSON.parse(l));
  if (rows.length <= maxPoints) return rows;
  const step = Math.ceil(rows.length / maxPoints);
  return rows.filter((r, i) => i % step === 0 || r.valMedian !== null || i === rows.length - 1);
}

function logTail(name, n = 40) {
  const dir = path.join(RUNS, name);
  let logs = [];
  try { logs = fs.readdirSync(dir).filter(f => f.endsWith('.log')); } catch (e) { return []; }
  if (!logs.length) return [];
  const newest = logs.map(f => path.join(dir, f)).sort((a, b) => mtime(b) - mtime(a))[0];
  return readLines(newest).slice(-n);
}

// Weights for live playback: the latest checkpoint's center ('current') or best.json.
function runModel(name, which) {
  const dir = path.join(RUNS, name);
  if (which === 'best') {
    const b = readJson(path.join(dir, 'best.json'));
    return b && { format: b.format, layers: b.layers, actionRepeat: b.actionRepeat, params: b.params, generation: b.generation, which };
  }
  const ck = readJson(path.join(dir, 'checkpoint.json'));
  return ck && { layers: ck.layers, actionRepeat: ck.config.actionRepeat, params: ck.theta, generation: ck.generation, which: 'current' };
}

// The page runs the real game code, served straight from the repo.
const SHARED = { '/core.js': 'core.js', '/policy.js': 'policy.js', '/render.js': 'render.js' };

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
    if (url.pathname === '/api/runs') return send(res, 200, 'application/json', JSON.stringify(listRuns()));
    if (m) {
      const name = decodeURIComponent(m[1]);
      if (name.includes('/') || name.includes('..') || !fs.existsSync(path.join(RUNS, name))) return send(res, 404, 'text/plain', 'no such run');
      if (m[2]) {
        const model = runModel(name, url.searchParams.get('which') === 'best' ? 'best' : 'current');
        return model ? send(res, 200, 'application/json', JSON.stringify(model)) : send(res, 404, 'text/plain', 'no weights yet');
      }
      return send(res, 200, 'application/json', JSON.stringify({ info: runInfo(name), metrics: runMetrics(name), log: logTail(name) }));
    }
    send(res, 404, 'text/plain', 'not found');
  } catch (e) {
    send(res, 500, 'text/plain', String(e && e.message));
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`dino training dashboard: http://127.0.0.1:${PORT}  (watching ${path.relative(process.cwd(), RUNS) || RUNS})`);
});
