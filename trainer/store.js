'use strict';
// Training-directory persistence: core hash, atomic JSON writes, the PID lockfile.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.join(__dirname, '..');

// Hash of the simulation the weights were trained against: the core rules and
// the shared observation/inference code (an observation change also invalidates them).
function coreHash() {
  const h = crypto.createHash('sha256');
  for (const f of ['core.js', 'policy.js']) h.update(f + '\0').update(fs.readFileSync(path.join(ROOT, f)));
  return h.digest('hex');
}

// Write to a temp file in the same directory, fsync, then rename over the target.
function writeAtomic(file, text) {
  const tmp = `${file}.${process.pid}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeSync(fd, text);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, file);
}
const writeJsonAtomic = (file, obj) => writeAtomic(file, JSON.stringify(obj) + '\n');
const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));

function pidAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === 'EPERM';     // exists but owned by someone else
  }
}

// Exclusive trainer lock for a training directory. Throws if a live process holds
// it; silently replaces a lock whose PID is dead (stale). Returns release().
function acquireLock(dir) {
  const file = path.join(dir, 'train.lock');
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const fd = fs.openSync(file, 'wx');
      fs.writeSync(fd, JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString() }) + '\n');
      fs.closeSync(fd);
      let released = false;
      return function release() {
        if (released) return;
        released = true;
        try {
          if (readJson(file).pid === process.pid) fs.unlinkSync(file);
        } catch (e) { /* already gone */ }
      };
    } catch (e) {
      if (e.code !== 'EEXIST') throw e;
      let holder = null;
      try { holder = readJson(file); } catch (err) { /* unreadable: treat as stale */ }
      if (holder && holder.pid !== process.pid && pidAlive(holder.pid)) {
        throw new Error(`${file} is held by running trainer PID ${holder.pid} (since ${holder.startedAt}); refusing to start a second trainer`);
      }
      console.log(`removing stale lock ${file}${holder ? ` (PID ${holder.pid} is not running)` : ''}`);
      fs.unlinkSync(file);
    }
  }
  throw new Error(`could not acquire ${file}`);
}

// Held-out assessment records (`cli.js assess`). New records go to assess.jsonl; runs
// assessed before the rename (2026-10-10) also have evals.jsonl. Readers merge both,
// legacy file first (records are append-only, so that is chronological for any run
// that was assessed under both names).
const ASSESS_FILE = 'assess.jsonl';
const ASSESS_FILES = ['evals.jsonl', ASSESS_FILE];
function readAssessLines(dir) {
  const out = [];
  for (const f of ASSESS_FILES) {
    let text;
    try { text = fs.readFileSync(path.join(dir, f), 'utf8'); } catch (e) { continue; }
    for (const l of text.split('\n')) if (l.trim()) out.push(l);
  }
  return out;
}

module.exports = { ROOT, coreHash, writeAtomic, writeJsonAtomic, readJson, acquireLock, pidAlive, ASSESS_FILE, ASSESS_FILES, readAssessLines };
