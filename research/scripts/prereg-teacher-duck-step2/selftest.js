'use strict';
// Self-tests for the step-2 tools (research/prereg-teacher-duck-step2.md §5.4). A few seconds,
// one thread. Run from anywhere: node research/scripts/prereg-teacher-duck-step2/selftest.js
// Needs the jump-generation sweep outputs in /Users/kyle/Desktop/dino-game/runs/exp-bc-sweep.
//   1. duckshare.js counter control on the duck teacher (2 seeds to 100k): exit 0, share 1.
//   2. duckshare.js negative control: a jump-generation clone (d2-w8-s1) ducks nothing and
//      meets the same duckable obstacles as the teacher (it jumps them all).
//   3. diag.js with the default teacher reproduces the sweep's recorded d8-w8-s1 diagnosis
//      byte for byte (the three-action extension leaves the default unchanged).
//   4. diag.js --teacher duck runs on the same seeds and emits a classHint per failure.
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');
const HERE = __dirname;
const SWEEP = '/Users/kyle/Desktop/dino-game/runs/exp-bc-sweep';
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'duck2-selftest-'));
const node = (script, args) => execFileSync(process.execPath, [script, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });

const t = JSON.parse(node(path.join(HERE, 'duckshare.js'), ['--teacher', 'duck', '--n', '2', '--target', '100000']));
assert.strictEqual(t.duckShare, 1, 'teacher duck share');
assert.ok(t.met > 0 && t.downAir === 0 && t.mixed === 0 && t.crashed === 0);
console.log(`ok 1 teacher counter control: ${t.ducked}/${t.met} ducked, ${t.downFrames} DOWN frames`);

const j = JSON.parse(node(path.join(HERE, 'duckshare.js'), ['--model', path.join(SWEEP, 'd2-w8-s1', 'best.json'), '--n', '2', '--target', '100000']));
assert.strictEqual(j.ducked, 0, 'jump clone ducks');
assert.strictEqual(j.jumped, t.met, 'jump clone meets and jumps the same duckable obstacles');
console.log(`ok 2 jump-clone negative control: ${j.jumped}/${j.met} jumped, ${j.ducked} ducked`);

const DIAG = path.join(HERE, '..', 'prereg-bc-robustness', 'diag.js');
const common = ['--arm', 'd8-w8-s1', '--scan', path.join(SWEEP, 'scans', 'd8-w8-s1.jsonl'), '--model', path.join(SWEEP, 'd8-w8-s1', 'best.json')];
node(DIAG, [...common, '--out', path.join(tmp, 'jump.json')]);
assert.ok(fs.readFileSync(path.join(tmp, 'jump.json')).equals(fs.readFileSync(path.join(SWEEP, 'diag', 'd8-w8-s1.json'))), 'diag default output changed');
console.log('ok 3 diag.js default output byte-identical to the sweep record');

node(DIAG, [...common, '--teacher', 'duck', '--out', path.join(tmp, 'duck.json')]);
const d = JSON.parse(fs.readFileSync(path.join(tmp, 'duck.json'), 'utf8'));
assert.ok(d.length === 2 && d.every(r => r.teacher === 'duck' && typeof r.duck.classHint === 'string'));
console.log(`ok 4 diag.js --teacher duck: ${d.map(r => `${r.seed} ${r.duck.classHint}`).join('; ')}`);
fs.rmSync(tmp, { recursive: true, force: true });
