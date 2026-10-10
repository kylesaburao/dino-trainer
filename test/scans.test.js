'use strict';
// trainer/scans.js: scan/survey JSONL discovery, incremental tailing and status.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { listScans, classify } = require('../trainer/scans.js');
const { clopperPearson } = require('../trainer/rules.js');

const header = (o) => JSON.stringify({ header: true, model: '/x/runs/grp/m/best.json', layers: [44, 8, 3], gameVersion: 'v2', actionRepeat: 1,
  generation: null, coreHash: 'c'.repeat(64), base: 100, n: 4, target: 1000, wasm: false, workers: 2, start: '2026-10-10T00:00:00.000Z', ...o });
const seed = (s, score) => JSON.stringify({ seed: s, score, pass: score >= 1000, frames: 10 });
const deadPid = () => spawnSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))']).stdout.toString() | 0;

test('classify: scan headers and survey rows, nothing else', () => {
  assert.strictEqual(classify(header({})), 'scan');
  assert.strictEqual(classify(JSON.stringify({ seed: 1, crashed: false, score: 5, frames: 1, obstaclesSeen: 3 })), 'survey');
  for (const l of ['{"generation":1,"fitMean":2}', '{"seed":1,"score":2}', 'not json', '', '[1,2]']) assert.strictEqual(classify(l), null, l);
});

test('tailing: partial lines wait, appends are parsed once, out-of-order seeds, footer = done', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-scans-'));
  const f = path.join(dir, 'grp', 'scans', 'a.jsonl');
  fs.mkdirSync(path.dirname(f), { recursive: true });
  fs.writeFileSync(f, [header({ pid: process.pid, host: os.hostname() }), seed(102, 1000), seed(100, 470)].join('\n') + '\n' + seed(101, 10).slice(0, 9));
  fs.writeFileSync(path.join(dir, 'grp', 'metrics.jsonl'), '{"generation":1}\n');
  fs.writeFileSync(path.join(dir, 'grp', 'other.jsonl'), '{"foo":1}\n');
  let [s] = listScans(dir, new Set(['grp/m']));
  assert.strictEqual(s.id, 'grp/scans/a.jsonl');
  assert.strictEqual(s.owner, 'grp/m');
  assert.deepStrictEqual([s.games, s.passes, s.failures, s.failMin, s.state], [2, 1, 1, 470, 'running']);
  assert.match(s.how, new RegExp(`pid ${process.pid} alive`));
  fs.appendFileSync(f, seed(101, 10).slice(9) + '\n' + seed(103, 1000) + '\n' + seed(103, 1000) + '\n');
  [s] = listScans(dir, new Set());
  assert.deepStrictEqual([s.games, s.passes, s.failures, s.dupes, s.owner], [4, 2, 2, 1, null]);
  assert.deepStrictEqual(s.failScores, [10, 470]);
  assert.deepStrictEqual(s.seedRange, [100, 103]);
  const cp = clopperPearson(2, 4);
  assert.deepStrictEqual(s.failCI95, { lower: cp.lower, upper: cp.upper });
  assert.strictEqual(s.failHist.reduce((a, b) => a + b.count, 0), 2);
  fs.appendFileSync(f, JSON.stringify({ footer: true, seconds: 1, end: '2026-10-10T00:01:00.000Z' }) + '\n');
  [s] = listScans(dir, new Set());
  assert.deepStrictEqual([s.state, s.how, s.seconds], ['done', 'footer', 1]);
  // Resume after a footer: running again under the resumer's pid.
  fs.appendFileSync(f, JSON.stringify({ resume: true, pid: process.pid, host: os.hostname(), start: 'x' }) + '\n');
  [s] = listScans(dir, new Set());
  assert.strictEqual(s.state, 'running');
  // Truncated and rewritten: re-read from the start.
  fs.writeFileSync(f, header({}) + '\n' + seed(100, 1000) + '\n');
  [s] = listScans(dir, new Set());
  assert.deepStrictEqual([s.games, s.failures, s.pid], [1, 0, null]);
  // Rewritten in place with more bytes than were read: the changed first line forces a re-read.
  fs.writeFileSync(f, header({ start: '2026-10-11T00:00:00.000Z' }) + '\n' + [seed(100, 5), seed(101, 6), seed(102, 7)].join('\n') + '\n');
  [s] = listScans(dir, new Set());
  assert.deepStrictEqual([s.games, s.failures, s.dupes, s.startedAt], [3, 3, 0, '2026-10-11T00:00:00.000Z']);
});

test('status: exited pid is stalled, other host and no pid fall back to mtime, .pid file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-scans-'));
  const w = (name, lines, ageMin = 0) => {
    const f = path.join(dir, name);
    fs.writeFileSync(f, lines.join('\n') + '\n');
    if (ageMin) { const t = new Date(Date.now() - ageMin * 60000); fs.utimesSync(f, t, t); }
    return f;
  };
  const dead = deadPid();
  w('dead.jsonl', [header({ pid: dead, host: os.hostname() }), seed(100, 1000)]);
  w('foreign.jsonl', [header({ pid: process.pid, host: 'some-other-host' }), seed(100, 1000)], 60);
  w('old-recent.jsonl', [header({}), seed(100, 1000)]);
  w('old-idle.jsonl', [header({}), seed(100, 1000)], 60);
  w('side.jsonl', [header({}), seed(100, 1000)], 60);
  fs.writeFileSync(path.join(dir, 'side.pid'), `${process.pid}\n`);
  w('survey-duck-40000000-2.jsonl', [JSON.stringify({ seed: 40000001, crashed: true, score: 812, frames: 5, obstaclesSeen: 1 }),
    JSON.stringify({ seed: 40000000, crashed: false, score: 1000000, frames: 9, obstaclesSeen: 2 })], 60);
  const by = Object.fromEntries(listScans(dir).map(s => [s.id, s]));
  assert.deepStrictEqual([by['dead.jsonl'].state, by['dead.jsonl'].how], ['stalled', `pid ${dead} exited without a footer`]);
  assert.strictEqual(by['foreign.jsonl'].state, 'stalled');
  assert.match(by['foreign.jsonl'].how, /some-other-host/);
  assert.strictEqual(by['old-recent.jsonl'].state, 'running');
  assert.match(by['old-recent.jsonl'].how, /no pid recorded; written/);
  assert.strictEqual(by['old-idle.jsonl'].state, 'stalled');
  assert.strictEqual(by['side.jsonl'].state, 'running');
  const sv = by['survey-duck-40000000-2.jsonl'];
  assert.deepStrictEqual([sv.kind, sv.solver, sv.base, sv.n, sv.games, sv.passes, sv.failMin, sv.state], ['survey', 'duck', 40000000, 2, 2, 1, 812, 'done']);
});
