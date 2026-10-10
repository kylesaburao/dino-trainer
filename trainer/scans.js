'use strict';
// Game-scan files for the dashboard (read-only): discovery, incremental parsing and status.
//
// A scan file is any *.jsonl under the runs root (nested to MAX_DEPTH) whose first line is
//   - a research/scripts/prereg-bc-robustness/scan.js header:
//       {"header":true,"model":...,"base":B,"n":N,"target":T,...,"pid":P,"host":H}
//     (newer headers add game, modelGameVersion and startScore: the rules played, the model's
//     own rules and the score every game starts at; `start` is the ISO start time)
//     then one {"seed","score","pass","frames"} line per seed (not necessarily in seed order),
//     a {"resume":true,"pid","host","start"} line when a later scan.js resumes the file, and a
//     {"footer":true,"seconds","end"} line at the end of every completed pass; or
//   - a research/scripts/prereg-teacher-duck/survey.js row (no header or footer): one
//     {"seed","crashed","score","frames",...,"obstaclesSeen"} line per seed, in completion
//     order. base/n come from the default file name survey-<solver>-<base>-<n>.jsonl or the
//     "survey solver=... seeds A..B cap C" line of a sibling <name>.log.
// Files are tailed: only bytes appended since the last read are parsed (a trailing partial
// line waits for its newline), and a file that shrinks or is replaced is re-read from the
// start. Non-scan files are classified once per mtime and size from their first 64 KB.
//
// Status: done = the last record is a footer (scan) or every seed is in (survey). Otherwise
// running/stalled from, in order: the pid in the newest header/resume line (alive on this
// host = running, exited = stalled); a sibling <name>.pid file (launchers write one); else
// the file's mtime (written within STALL_MS = running, else stalled). `how` says which.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { pidAlive } = require('./store.js');
const { clopperPearson } = require('./rules.js');

const SKIP_FILES = new Set(['metrics.jsonl', 'bc-metrics.jsonl', 'evals.jsonl', 'assess.jsonl']);
const SKIP_DIRS = new Set(['snapshots', 'export', 'node_modules']);
const MAX_DEPTH = 4;
const STALL_MS = 15 * 60 * 1000;
const HEAD_BYTES = 64 * 1024;
const FAIL_LIST = 200;          // lowest failure scores sent to the page
const HOST = os.hostname();
const SURVEY_NAME = /^survey-(\w+)-(\d+)-(\d+)\.jsonl$/;
const SURVEY_LOG = /survey solver=(\w+) seeds (\d+)\.\.(\d+) cap (\d+)/;

function findJsonl(root) {
  const out = [];
  const walk = (dir, depth) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return; }
    for (const e of entries) {
      if (e.isDirectory()) { if (depth < MAX_DEPTH && !SKIP_DIRS.has(e.name)) walk(path.join(dir, e.name), depth + 1); }
      else if (e.isFile() && e.name.endsWith('.jsonl') && !SKIP_FILES.has(e.name)) out.push(path.join(dir, e.name));
    }
  };
  walk(root, 0);
  return out;
}

function readRange(file, start, end) {
  const fd = fs.openSync(file, 'r');
  try {
    const buf = Buffer.alloc(end - start);
    let got = 0;
    while (got < buf.length) {
      const n = fs.readSync(fd, buf, got, buf.length - got, start + got);
      if (!n) break;
      got += n;
    }
    return buf.subarray(0, got);
  } finally {
    fs.closeSync(fd);
  }
}

function firstLine(file, size) {
  const buf = readRange(file, 0, Math.min(size, HEAD_BYTES));
  const i = buf.indexOf(10);
  return i < 0 ? null : buf.subarray(0, i).toString('utf8');
}

// 'scan' | 'survey' | null, from the first line.
function classify(line) {
  let r;
  try { r = JSON.parse(line); } catch (e) { return null; }
  if (!r || typeof r !== 'object') return null;
  if (r.header === true && 'model' in r && 'base' in r && 'n' in r && 'target' in r) return 'scan';
  if (r.seed !== undefined && 'crashed' in r && 'obstaclesSeen' in r) return 'survey';
  return null;
}

function surveyMeta(file) {
  const m = SURVEY_NAME.exec(path.basename(file));
  if (m) return { solver: m[1], base: Number(m[2]), n: Number(m[3]), target: null };
  try {
    const log = file.replace(/\.jsonl$/, '.log');
    const st = fs.statSync(log);
    const l = SURVEY_LOG.exec(readRange(log, 0, Math.min(st.size, 4096)).toString('utf8'));
    if (l) return { solver: l[1], base: Number(l[2]), n: Number(l[3]) - Number(l[2]) + 1, target: Number(l[4]) };
  } catch (e) { /* no log */ }
  return { solver: null, base: null, n: null, target: null };
}

function freshState(kind, file, ino) {
  return {
    kind, ino, offset: 0, rest: '', header: null, pid: null, host: null, startedAt: null,
    seeds: new Set(), dupes: 0, passes: 0, failScores: [], frames: 0, last: null, footer: null, badLines: 0,
    meta: kind === 'survey' ? surveyMeta(file) : null, ci: null,
  };
}

function consume(st, line) {
  let r;
  try { r = JSON.parse(line); } catch (e) { st.badLines++; return; }
  if (r.header) {
    st.header = r; st.last = 'header';
    if (r.pid !== undefined) { st.pid = r.pid; st.host = r.host || null; }
    st.startedAt = r.start || null;
  } else if (r.resume) {
    st.last = 'resume'; st.pid = r.pid; st.host = r.host || null;
  } else if (r.footer) {
    st.last = 'footer'; st.footer = r;
  } else if (r.seed !== undefined) {
    st.last = 'seed';
    if (st.seeds.has(r.seed)) { st.dupes++; return; }
    st.seeds.add(r.seed);
    if (r.game && !st.game) st.game = r.game;          // survey.js --game (rules other than v2)
    if (r.start && st.startScore === undefined) st.startScore = r.start;   // survey.js --start
    st.frames += r.frames || 0;
    const pass = st.kind === 'survey' ? !r.crashed : r.pass === true;
    if (pass) st.passes++; else st.failScores.push(r.score);
    st.ci = null;
  }
}

// Bring a file's state up to date; parses only appended bytes.
const states = new Map();      // file -> state (scan files) or { key, kind: null } (others)
function update(file) {
  let s;
  try { s = fs.statSync(file); } catch (e) { states.delete(file); return null; }
  const key = `${s.mtimeMs}:${s.size}`;
  let st = states.get(file);
  if (st && st.kind === null) {
    if (st.key === key) return null;
    st = null;
  }
  // Replaced, truncated, or rewritten in place (survey.js opens its output with 'w'): the
  // first line no longer matches.
  if (st && st.key !== key && (s.ino !== st.ino || s.size < st.offset ||
      readRange(file, 0, st.head.length + 1).toString('utf8') !== st.head + '\n')) st = null;
  if (!st) {
    const head = s.size ? firstLine(file, s.size) : null;
    const kind = head ? classify(head) : null;
    if (!kind) { states.set(file, { key, kind: null }); return null; }
    st = freshState(kind, file, s.ino);
    st.head = head;
    states.set(file, st);
  }
  st.key = key;
  st.mtimeMs = s.mtimeMs;
  if (s.size > st.offset) {
    const text = st.rest + readRange(file, st.offset, s.size).toString('utf8');
    st.offset = s.size;
    const lines = text.split('\n');
    st.rest = lines.pop();
    for (const l of lines) if (l) consume(st, l);
  }
  return st;
}

function pidFile(file) {
  try {
    const pid = Number(fs.readFileSync(file.replace(/\.jsonl$/, '.pid'), 'utf8').trim());
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  } catch (e) { return null; }
}

function lastLogLine(file) {
  const log = file.replace(/\.jsonl$/, '.log');
  try {
    const s = fs.statSync(log);
    const lines = readRange(log, Math.max(0, s.size - 2048), s.size).toString('utf8').split('\n').filter(Boolean);
    return lines.length ? lines[lines.length - 1] : null;
  } catch (e) { return null; }
}

function status(st, n, file, now) {
  const games = st.seeds.size;
  if (st.kind === 'scan' ? st.last === 'footer' : n && games >= n) return { state: 'done', how: st.kind === 'scan' ? 'footer' : 'all seeds in' };
  if (st.pid !== null && (!st.host || st.host === HOST)) {
    return pidAlive(st.pid) ? { state: 'running', how: `pid ${st.pid} alive` } : { state: 'stalled', how: `pid ${st.pid} exited without a footer` };
  }
  const side = pidFile(file);
  if (side) return pidAlive(side) ? { state: 'running', how: `pid ${side} (.pid file) alive` } : { state: 'stalled', how: `pid ${side} (.pid file) exited` };
  const idle = now - st.mtimeMs;
  const other = st.pid !== null ? `pid ${st.pid} is on ${st.host}; ` : 'no pid recorded; ';
  return idle < STALL_MS ? { state: 'running', how: `${other}written ${Math.round(idle / 1000)} s ago` }
    : { state: 'stalled', how: `${other}no writes for ${Math.round(idle / 60000)} min` };
}

// Decade bins up to the target: failure scores are few, so the page lists them too.
function histogram(scores, target) {
  const top = Math.max(target || 0, ...scores, 1);
  const edges = [0];
  for (let e = 1000; e < top; e *= 10) edges.push(e);
  edges.push(top + 1);
  return edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], count: scores.filter(s => s >= lo && s < edges[i + 1]).length }));
}

// The model's run name, when the header's model path is <...>/runs/<name>/best.json.
function ownerOf(model, root, runNames) {
  if (!model) return null;
  const cands = [path.relative(root, model)];
  const i = model.lastIndexOf('/runs/');
  if (i >= 0) cands.push(model.slice(i + 6));
  for (const c of cands) {
    if (!c.endsWith('best.json') || c.startsWith('..')) continue;
    const name = path.dirname(c).split(path.sep).join('/');
    if (runNames.has(name)) return name;
  }
  return null;
}

function summary(file, st, root, runNames, now) {
  const h = st.header || {}, meta = st.meta || {};
  const n = st.kind === 'scan' ? h.n : meta.n;
  const target = st.kind === 'scan' ? h.target : meta.target;
  const games = st.seeds.size, failures = st.failScores.length;
  if (!st.ci && games) st.ci = { k: failures, n: games, ...clopperPearson(failures, games) };
  const fails = st.failScores.slice().sort((a, b) => a - b);
  let lo = Infinity, hi = -Infinity;
  for (const s of st.seeds) { if (s < lo) lo = s; if (s > hi) hi = s; }
  const { state, how } = status(st, n, file, now);
  const model = st.kind === 'scan' ? h.model : null;
  return {
    id: path.relative(root, file).split(path.sep).join('/'),
    kind: st.kind,
    model,
    modelRel: model && model.lastIndexOf('/runs/') >= 0 ? model.slice(model.lastIndexOf('/runs/') + 1) : model,
    owner: ownerOf(model, root, runNames),
    solver: meta.solver || null,
    layers: h.layers || null, gameVersion: h.game || h.gameVersion || st.game || null, wasm: h.wasm ?? null, workers: h.workers ?? null,
    // scan.js --game (cross-rules) and --start, survey.js --start: rules played, the model's own
    // rules, and the score every game starts at (0 = from the start; old headers have none).
    modelGameVersion: h.modelGameVersion || null, startScore: (st.kind === 'scan' ? h.startScore : st.startScore) || 0,
    coreHash: h.coreHash || null,
    base: st.kind === 'scan' ? h.base : meta.base, n: n ?? null, target: target ?? null,
    games, passes: st.passes, failures, frames: st.frames, dupes: st.dupes, badLines: st.badLines,
    seedRange: games ? [lo, hi] : null,
    failMin: fails.length ? fails[0] : null,
    failScores: fails.slice(0, FAIL_LIST),
    failHist: histogram(fails, target),
    failRate: games ? failures / games : null,
    failCI95: st.ci && st.ci.n === games && st.ci.k === failures ? { lower: st.ci.lower, upper: st.ci.upper } : null,
    state, how, pid: st.pid, host: st.host,
    startedAt: st.startedAt, endedAt: st.footer ? st.footer.end : null, seconds: st.footer ? st.footer.seconds : null,
    updated: st.mtimeMs,
    logLine: lastLogLine(file),
  };
}

// Every scan file under root, running first, then most recently written.
function listScans(root, runNames = new Set()) {
  const now = Date.now();
  const files = findJsonl(root);
  const keep = new Set(files);
  for (const f of states.keys()) if (!keep.has(f)) states.delete(f);
  const out = [];
  for (const f of files) {
    const st = update(f);
    if (st) out.push(summary(f, st, root, runNames, now));
  }
  const rank = { running: 0, stalled: 1, done: 2 };
  return out.sort((a, b) => (rank[a.state] - rank[b.state]) || (b.updated - a.updated));
}

module.exports = { listScans, classify, histogram, STALL_MS, _states: states };
