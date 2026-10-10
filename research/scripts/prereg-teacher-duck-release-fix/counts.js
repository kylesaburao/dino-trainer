'use strict';
// C-a1 / C-c1 counts (research/prereg-teacher-duck-release-fix.md §4.2): duck-transition rows of a
// demos set, with the same rule as bc-train --transition-weight (trainer/bc.js transitionMask;
// trainer/bc_train.py): grounded, and onset = not ducking with label duck, release = ducking with
// label not duck (release into a jump counted separately as well). Reads demos.bin in chunks and
// splits the counts by the manifest's episodes (per start level). Static: no game is played.
// Usage: node counts.js --demos <dir or demos.json> [--out f.json]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const bc = require(path.join(ROOT, 'trainer', 'bc.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
if (!opt('--demos')) throw new Error('usage: node counts.js --demos <dir or demos.json> [--out f.json]');
const demos = bc.resolveDemos(path.resolve(opt('--demos')));
const { manifest, binFile } = demos;
const OBS = manifest.obsDim, ROW = manifest.rowFloats;
const size = fs.statSync(binFile).size;
if (size !== manifest.rows * ROW * 4) throw new Error(`${binFile}: ${size} bytes, manifest says ${manifest.rows} rows`);

// Episode boundaries (row index -> level); a manifest without episodes counts as one block.
const eps = Array.isArray(manifest.episodes) && manifest.episodes.length ? manifest.episodes : [{ level: 'all', rows: manifest.rows }];
const bounds = [];
let at = 0;
for (const e of eps) { bounds.push({ from: at, to: at + e.rows, level: e.level }); at += e.rows; }
if (at !== manifest.rows) throw new Error(`episodes cover ${at} rows, manifest says ${manifest.rows}`);

const blank = () => ({ rows: 0, duckRows: 0, onset: 0, release: 0, releaseToNone: 0, releaseToJump: 0 });
const total = blank(), byLevel = {};
const fd = fs.openSync(binFile, 'r');
const CHUNK = 65536;
const buf = Buffer.alloc(CHUNK * ROW * 4);
let row = 0, b = 0;
try {
  for (;;) {
    const n = fs.readSync(fd, buf, 0, buf.length, null);
    if (n <= 0) break;
    const f = new Float32Array(buf.buffer, buf.byteOffset, n / 4);
    for (let r = 0; r < n / 4 / ROW; r++, row++) {
      while (row >= bounds[b].to) b++;
      const lv = (byLevel[bounds[b].level] = byLevel[bounds[b].level] || blank());
      const o = r * ROW, label = f[o + OBS];
      const kind = bc.transitionMask(f[o + 2], f[o + 3], label);
      for (const c of [total, lv]) {
        c.rows++;
        if (label === 2) c.duckRows++;
        if (kind === 1) c.onset++;
        if (kind === 2) { c.release++; if (label === 1) c.releaseToJump++; else c.releaseToNone++; }
      }
    }
  }
} finally {
  fs.closeSync(fd);
}
const report = { demos: path.relative(ROOT, demos.manifestFile), sha256: manifest.sha256, teacherName: manifest.teacherName || 'jump',
  source: manifest.source || 'teacher', scoreWindow: manifest.scoreWindow || null, total, byLevel };
const out = opt('--out', null);
if (out) fs.writeFileSync(out, JSON.stringify(report) + '\n');
console.log(JSON.stringify(report));
