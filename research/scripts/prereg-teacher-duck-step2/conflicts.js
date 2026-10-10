'use strict';
// Observability probe (research/prereg-teacher-duck-step2.md §5.1).
// Reads a bc-gen demos file (float32 rows [44 features, label, weight]) and asks:
// how much of the teacher's labelling is NOT a function of the observed features?
//
// Rows are grouped into cells by their 44 features, either exactly (float32 bits)
// or quantised to a grid g (round(x / g)). Per cell we sum row weights per label.
// Reported, all weighted by row weight:
//   - irreducible[c]: share of class-c weight that sits in cells whose majority label
//     is not c. For --grid exact this is a lower bound on the error of ANY
//     deterministic function of the 44 features on these rows (recall ceiling 1 - it).
//     For a grid it is a near-identity proxy: cells merge distinct inputs, so it can
//     only overstate the exact figure.
//   - pair[a,b]: sum over cells of min(w_a, w_b), divided by the smaller of W_a, W_b:
//     the share of the rarer class's weight that is "contested" by the other class.
//   - splits: the same statistics restricted to subsets of rows (airborne / grounded;
//     third obstacle slot present / absent; nearest obstacle a pterodactyl of each
//     height class; grounded with a mid pterodactyl nearest; ducking flag set).
//     Splits select on the UNMASKED features.
// --mask a,b,c zeroes those feature indices before grouping (probe positive control:
// masking the pterodactyl height one-hots and box y must create none/jump conflicts
// on teacher-jump demos, because high pterodactyls are "none" and low/mid are "jump").
// Usage: node conflicts.js --demos <dir or demos.json> [--grid exact|1e-3|1e-2] [--mask 14,15,16,7] [--out f.json]
// Single-threaded; about 10-30 s on 2.1M rows.
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
let man = path.resolve(opt('--demos'));
if (fs.statSync(man).isDirectory()) man = path.join(man, 'demos.json');
const M = JSON.parse(fs.readFileSync(man, 'utf8'));
const DIM = M.obsDim, ROW = M.rowFloats;
const gridArg = opt('--grid', 'exact');
const grid = gridArg === 'exact' ? 0 : Number(gridArg);
const mask = (opt('--mask', '') || '').split(',').filter(Boolean).map(Number);
const out = opt('--out', null);

const buf = fs.readFileSync(path.join(path.dirname(man), M.file));
const F = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const I = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const n = F.length / ROW;
if (n !== M.rows) throw new Error(`rows ${n} != manifest ${M.rows}`);

// Feature layout (policy.js): 0 dino y, 1 vy, 2 jumping, 3 ducking, 4 speed; obstacle k
// at b = 5 + 13k: b present, b+1 rel x, b+2 box y, b+3 w, b+4 h, b+5..7 type one-hot,
// b+8 size, b+9..11 ptero height one-hot (ys [108, 82, 58] = low, mid, high), b+12 vel.
const SPLITS = {
  all: () => true,
  airborne: (o) => F[o + 2] === 1,
  grounded: (o) => F[o + 2] !== 1,
  slot3present: (o) => F[o + 5 + 26] === 1,
  slot3absent: (o) => F[o + 5 + 26] !== 1,
  near0_ptero_low: (o) => F[o + 5 + 9] === 1,
  near0_ptero_mid: (o) => F[o + 5 + 10] === 1,
  near0_ptero_high: (o) => F[o + 5 + 11] === 1,
  grounded_near0_mid: (o) => F[o + 2] !== 1 && F[o + 5 + 10] === 1,
  ducking: (o) => F[o + 3] === 1,
};
const maskSet = new Set(mask);

// 53-bit cell key from two FNV-1a style 32-bit hashes of the (quantised) features.
function key(o) {
  let h1 = 0x811c9dc5 | 0, h2 = 0x01000193 | 0;
  for (let j = 0; j < DIM; j++) {
    let q;
    if (maskSet.has(j)) q = 0;
    else if (grid) q = Math.round(F[o + j] / grid) | 0;
    else q = I[o + j];
    h1 = Math.imul(h1 ^ q, 0x01000193);
    h2 = Math.imul(h2 ^ (q + j * 0x9e3779b1), 0x85ebca6b) ^ (h2 >>> 13);
  }
  return (h1 >>> 0) * 2097152 + ((h2 >>> 0) & 0x1fffff);
}

const results = {};
const t0 = Date.now();
for (const [name, pred] of Object.entries(SPLITS)) {
  const cells = new Map();
  let cw = new Float64Array(1 << 20), nc = 0;
  const W = [0, 0, 0];
  let rows = 0;
  for (let r = 0; r < n; r++) {
    const o = r * ROW;
    if (!pred(o)) continue;
    rows++;
    const lab = F[o + DIM], w = F[o + DIM + 1];
    W[lab] += w;
    const k = key(o);
    let idx = cells.get(k);
    if (idx === undefined) {
      idx = nc++; cells.set(k, idx);
      if (nc * 3 > cw.length) { const b = new Float64Array(cw.length * 2); b.set(cw); cw = b; }
    }
    cw[idx * 3 + lab] += w;
  }
  const wrong = [0, 0, 0], pair = { 'none,jump': 0, 'none,duck': 0, 'jump,duck': 0 };
  let mixedCells = 0;
  for (let i = 0; i < nc; i++) {
    const a = cw[i * 3], b = cw[i * 3 + 1], c = cw[i * 3 + 2];
    const maj = a >= b && a >= c ? 0 : (b >= c ? 1 : 2);
    if ((a > 0) + (b > 0) + (c > 0) > 1) mixedCells++;
    for (let l = 0; l < 3; l++) if (l !== maj) wrong[l] += cw[i * 3 + l];
    pair['none,jump'] += Math.min(a, b); pair['none,duck'] += Math.min(a, c); pair['jump,duck'] += Math.min(b, c);
  }
  const share = (x, d) => (d > 0 ? x / d : null);
  results[name] = {
    rows, cells: nc, mixedCells, weight: { none: W[0], jump: W[1], duck: W[2] },
    irreducible: { none: share(wrong[0], W[0]), jump: share(wrong[1], W[1]), duck: share(wrong[2], W[2]),
      total: share(wrong[0] + wrong[1] + wrong[2], W[0] + W[1] + W[2]) },
    pair: { 'none,jump': share(pair['none,jump'], Math.min(W[0], W[1])), 'none,duck': share(pair['none,duck'], Math.min(W[0], W[2])),
      'jump,duck': share(pair['jump,duck'], Math.min(W[1], W[2])) },
  };
}
const report = { demos: man, teacher: M.teacher, teacherName: M.teacherName ?? null, sha256: M.sha256, rows: n, grid: gridArg, mask,
  seconds: (Date.now() - t0) / 1000, results };
const fmt = (x) => (x === null ? '-' : (100 * x).toFixed(4) + '%');
console.log(`demos ${man} (${n} rows, sha ${String(M.sha256).slice(0, 16)}), grid ${gridArg}, mask [${mask}]`);
for (const [s, r] of Object.entries(results)) {
  console.log(`${s.padEnd(18)} rows ${String(r.rows).padStart(8)} cells ${String(r.cells).padStart(8)} mixed ${String(r.mixedCells).padStart(6)} | ` +
    `irreducible none ${fmt(r.irreducible.none)} jump ${fmt(r.irreducible.jump)} duck ${fmt(r.irreducible.duck)} total ${fmt(r.irreducible.total)} | ` +
    `pair n/j ${fmt(r.pair['none,jump'])} n/d ${fmt(r.pair['none,duck'])} j/d ${fmt(r.pair['jump,duck'])}`);
}
console.log(`${report.seconds.toFixed(1)} s`);
if (out) fs.writeFileSync(out, JSON.stringify(report, null, 1) + '\n');
