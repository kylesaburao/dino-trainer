#!/usr/bin/env node
'use strict';
// Cross-architecture determinism probe (research/prereg-x86-determinism.md).
//
//   node research/scripts/x86-determinism/probe.js [--out FILE] [--only S1,S4] [--python PATH]
//   node research/scripts/x86-determinism/probe.js --dump <set>:<bucket>
//
// Prints one JSON object (also written to --out), and one summary line per stage on stderr.
// Everything hashed is a pure function of the code and the machine's arithmetic; wall-clock
// times are kept apart in the top-level `timings` object, which compare.js ignores.
// Stages (each in its own try/catch, so one failure does not hide the others):
//   S0 env       arch, node, V8, compiler variables, CPU, git HEAD, coreHash, file sha256s
//   S1 libm      Math.* on fixed input sets (sampled ranges + the game's exhaustive rules
//                tables + Adam's bias-correction powers); per-bucket hashes; negative
//                controls (sqrt, fround, arithmetic); a 1-ulp sensitivity self-test
//   S3 traj      per-frame state hashes of solver / random-key games on rules v1, v2, v3
//   S4 demos     the three committed demo sets regenerated in memory (bc.genSeed, single thread)
//   S5 decide    held-out assess seeds with three models, JS net and wasm kernel
//   S6 train     the test/bc.test.js node-backend pin recipe, a 32x32 arm, ES noise and Adam
//   S7 python    optional (--python or PYTHON with numpy): bc_train.py on the pin recipe
//   S2 capture   runs LAST: Math.* replaced by recording wrappers during short games, policy
//                play and training, to show which functions the job paths really call
// Each item separates `discrete` (outcomes: rows, labels, scores, actions, crash frames)
// from `bits` (float bit patterns); compare.js reads H1 vs H2 from that split.
// --dump <set>:<bucket> prints one S1 bucket's inputs and outputs as hex (diff two machines'
// dumps with `diff`), e.g. --dump tanh:7, --dump rules-v2:37, --dump adam-pow:3.
// It writes nothing except --out and a scratch dir under os.tmpdir() (S6/S7; removed on exit).
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFileSync, spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..', '..', '..');
const C = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));

const FORMAT = 'dino-x86-determinism-probe-v1';
// Hash lengths (hex chars), to keep the output pasteable: LONG for an item's overall hash,
// SHORT for a named sub-hash, PACK for each entry of a packed list (a string of PACK-wide
// hashes in a fixed order: buckets, checkpoints, columns, levels). A packed entry can in
// principle collide (2^-32); the item's LONG hash covers the same bytes.
const LONG = 16, SHORT = 10, PACK = 8;

// ---------------------------------------------------------------------------
// Bits and hashes
// ---------------------------------------------------------------------------
const sha = () => crypto.createHash('sha256');
const bufOf = (ta) => Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength);
// NaN payloads are canonicalized (in place) before hashing float64 results: a freshly
// generated NaN (sqrt(-1), sin(Infinity), ...) takes the hardware's default NaN, whose sign
// bit differs by architecture (x86 SSE 0xFFF8..., arm64 0x7FF8...), which says nothing about
// libm accuracy. Demo rows (S4) are hashed raw, as bc-gen writes them; the game has no NaN.
const CANON_NAN = 0x7FF8000000000000n;
function canon(ta) {
  let u = null;
  for (let i = 0; i < ta.length; i++) if (ta[i] !== ta[i]) { u = u || new BigUint64Array(ta.buffer, ta.byteOffset, ta.length); u[i] = CANON_NAN; }
  return ta;
}
const hashTA = (ta) => sha().update(bufOf(canon(ta))).digest('hex');
const short = (hex) => hex.slice(0, SHORT);
const long = (hex) => hex.slice(0, LONG);
const pack = (hexes) => hexes.map(h => h.slice(0, PACK)).join('');
const DV = new DataView(new ArrayBuffer(8));
const hex64 = (x) => { DV.setFloat64(0, x); return DV.getBigUint64(0).toString(16).padStart(16, '0'); };
const fromBits = (hi, lo) => { DV.setUint32(0, hi >>> 0); DV.setUint32(4, lo >>> 0); return DV.getFloat64(0); };
const fileSha = (rel) => { try { return long(sha().update(fs.readFileSync(path.join(ROOT, rel))).digest('hex')); } catch { return null; } };

// Streaming hash of float64 values (buffered, so a per-value update does not dominate).
function f64Stream() {
  const h = sha(), buf = new Float64Array(4096);
  let k = 0, n = 0;
  return {
    push(x) { buf[k++] = x; n++; if (k === buf.length) { h.update(bufOf(canon(buf))); k = 0; } },
    copyDigest() { const c = h.copy(); if (k) c.update(bufOf(canon(buf.subarray(0, k)))); return c.digest('hex'); },
    get n() { return n; },
  };
}

// ---------------------------------------------------------------------------
// S1: input sets. Inputs are built from Core.createRng (exact k / 2^32) with IEEE + - * /
// and bit construction only, so they are identical on every machine; only the function
// under test can differ. Every range has `buckets` buckets of BUCKET inputs.
// ---------------------------------------------------------------------------
const BUCKET = 3000;
const uni = (lo, hi) => (r) => lo + (hi - lo) * r();
const int = (lo, hi) => (r) => lo + Math.floor(r() * (hi - lo + 1));
// Random float64 with a biased exponent in [e0, e1] (2^(e-1023)), random mantissa, random sign if signed.
const bitsF = (e0, e1, signed) => (r) => {
  const e = e0 + Math.floor(r() * (e1 - e0 + 1));
  const hi = ((signed && r() < 0.5 ? 0x800 : 0) | e) << 20 | Math.floor(r() * 0x100000);
  return fromBits(hi, Math.floor(r() * 4294967296));
};
const E = (p) => 1023 + p;                       // biased exponent of 2^p
const pair = (ga, gb) => (r) => [ga(r), gb(r)];

const EDGES1 = [0, -0, 1, -1, 0.5, -0.5, 1e-8, -1e-8, 1e-300, -1e-300, 5e-324, -5e-324, 2.2250738585072014e-308,
  0.1, 0.7, 2, 3, 10, 18, 19, 20, 22, 30, -30, 100, 700, 709.78, -745, 1e10, -1e10, 1e300, Infinity, -Infinity, NaN, Math.PI, Math.E];
const EDGES2 = [[0, 0], [1, 1], [0.9, 1], [0.999, 1], [0.5, 0.5], [2, 0.5], [10, -3], [0, -1], [-2, 3], [-0.5, 0.5],
  [Infinity, 0], [NaN, 0], [1, Infinity], [0.9, 30000], [0.999, 30000], [1e300, 2]];

const S1_FNS = {
  tanh: { f: Math.tanh, ranges: [['pre-activation u[-30,30]', 12, uni(-30, 30)], ['u[-3,3]', 8, uni(-3, 3)], ['near 0 u[-1e-3,1e-3]', 4, uni(-1e-3, 1e-3)], ['bits 2^-60..2^5 signed', 8, bitsF(E(-60), E(5), true)]] },
  exp: { f: Math.exp, ranges: [['softmax u[-50,0]', 12, uni(-50, 0)], ['u[-745,710]', 8, uni(-745, 710)], ['u[-1,1]', 4, uni(-1, 1)], ['bits 2^-60..2^9 signed', 8, bitsF(E(-60), E(9), true)]] },
  log: { f: Math.log, ranges: [['Box-Muller u = k/2^32', 12, (r) => { let u = 0; while (u === 0) u = r(); return u; }], ['softmax sum u[1,4]', 8, uni(1, 4)], ['bits 2^-60..2^20', 4, bitsF(E(-60), E(20), false)], ['bits full positive', 8, bitsF(1, 2046, false)]] },
  log10: { f: Math.log10, ranges: [['score/1000, score int', 12, (r) => int(1001, 1000000)(r) / 1000], ['score/100000, score int', 8, (r) => int(100001, 1000000)(r) / 100000], ['bits 2^-60..2^60', 12, bitsF(E(-60), E(60), false)]] },
  log2: { f: Math.log2, ranges: [['bits 2^-30..2^30', 16, bitsF(E(-30), E(30), false)], ['u[1e-3,1e3]', 16, uni(1e-3, 1e3)]] },
  pow: { f: Math.pow, two: true, ranges: [['v3 curve (u[0,1], u[0,2])', 8, pair(uni(0, 1), uni(0, 2))], ['Adam (0.9|0.999, int t)', 8, (r) => [r() < 0.5 ? 0.9 : 0.999, int(1, 1000000)(r)]], ['(u[0,10], u[-10,10])', 8, pair(uni(0, 10), uni(-10, 10))], ['(bits 2^-30..2^30, u[-3,3])', 8, pair(bitsF(E(-30), E(30), false), uni(-3, 3))]] },
  sin: { f: Math.sin, ranges: [['Box-Muller 2*PI*v', 16, (r) => 2 * Math.PI * r()], ['u[-100,100]', 8, uni(-100, 100)], ['cosine LR PI*t/steps', 8, (r) => Math.PI * int(1, 30000)(r) / 30000]] },
  cos: { f: Math.cos, ranges: [['Box-Muller 2*PI*v', 16, (r) => 2 * Math.PI * r()], ['u[-100,100]', 8, uni(-100, 100)], ['cosine LR PI*t/steps', 8, (r) => Math.PI * int(1, 30000)(r) / 30000]] },
  expm1: { f: Math.expm1, ranges: [['u[-30,30]', 16, uni(-30, 30)], ['bits 2^-60..2^5 signed', 16, bitsF(E(-60), E(5), true)]] },
  log1p: { f: Math.log1p, ranges: [['u[-1,10]', 16, uni(-1, 10)], ['bits 2^-60..2^30', 16, bitsF(E(-60), E(30), false)]] },
  sinh: { f: Math.sinh, ranges: [['u[-30,30]', 16, uni(-30, 30)], ['bits 2^-60..2^5 signed', 16, bitsF(E(-60), E(5), true)]] },
  cosh: { f: Math.cosh, ranges: [['u[-30,30]', 16, uni(-30, 30)], ['bits 2^-60..2^5 signed', 16, bitsF(E(-60), E(5), true)]] },
  atan: { f: Math.atan, ranges: [['u[-10,10]', 16, uni(-10, 10)], ['bits 2^-60..2^60 signed', 16, bitsF(E(-60), E(60), true)]] },
  atan2: { f: Math.atan2, two: true, ranges: [['(u[-10,10], u[-10,10])', 16, pair(uni(-10, 10), uni(-10, 10))], ['bits pairs signed', 16, pair(bitsF(E(-30), E(30), true), bitsF(E(-30), E(30), true))]] },
  cbrt: { f: Math.cbrt, ranges: [['u[-1e3,1e3]', 16, uni(-1e3, 1e3)], ['bits 2^-60..2^60 signed', 16, bitsF(E(-60), E(60), true)]] },
  // Negative controls: correctly rounded by IEEE 754 / ECMAScript; must match everywhere.
  sqrt: { control: true, f: Math.sqrt, ranges: [['bits full positive', 16, bitsF(1, 2046, false)], ['u[0,1e6]', 16, uni(0, 1e6)]] },
  fround: { control: true, f: Math.fround, ranges: [['bits 2^-150..2^130 signed', 32, bitsF(E(-150), E(130), true)]] },
  arith: { control: true, f: (a, b) => a * b + a / b - b, two: true, ranges: [['bits pairs signed', 32, pair(bitsF(E(-60), E(60), true), bitsF(E(-60), E(60), true))]] },
};

// The fn's inputs (Float64Array; pairs interleaved for two-argument fns) and bucket layout.
function s1Inputs(name) {
  const spec = S1_FNS[name];
  const arity = spec.two ? 2 : 1;
  const nb = spec.ranges.reduce((a, r) => a + r[1], 0);
  const xs = new Float64Array(nb * BUCKET * arity);
  const seedBase = (crypto.createHash('sha256').update(name).digest().readUInt32BE(0)) >>> 0;
  let k = 0, b = 0;
  const bucketRange = [];
  spec.ranges.forEach(([label, nbk, gen], ri) => {
    const r = C.createRng((seedBase + ri * 0x9E3779B1) >>> 0);
    for (let i = 0; i < nbk * BUCKET; i++) {
      const v = gen(r);
      if (arity === 2) { xs[k++] = v[0]; xs[k++] = v[1]; } else xs[k++] = v;
    }
    for (let i = 0; i < nbk; i++) bucketRange[b++] = label;
  });
  return { xs, arity, nb, bucketRange };
}

function s1Eval(name, xs, arity) {
  const f = S1_FNS[name].f;
  const n = xs.length / arity, ys = new Float64Array(n);
  if (arity === 2) for (let i = 0; i < n; i++) ys[i] = f(xs[2 * i], xs[2 * i + 1]);
  else for (let i = 0; i < n; i++) ys[i] = f(xs[i]);
  return ys;
}

const bucketHashes = (ys, per) => Array.from({ length: Math.ceil(ys.length / per) }, (_, b) => hashTA(ys.subarray(b * per, (b + 1) * per)));

// Domain tables, exhaustive. rules-v2 / rules-v3: the rules record core.js uses at every
// integer score (the only place the game calls libm). adam-pow: pow(0.9, t), pow(0.999, t).
const RULE_FIELDS = ['level', 'superLevel', 'maxSpeed', 'gapCoef', 'maxGapCoef', 'minWindowFrames', 'minGroundFrames', 'pteroWeight', 'pteroSpeedOffset', 'maxGroup'];
const DOMAIN_SCORE_BUCKET = 10000;
const DOMAINS = {
  'rules-v2': { buckets: 101, note: 'C.difficulty/superLevel + rulesAt at integer score s, bucket = floor(s / 10000), s = 0..1,000,000', fill: (b) => rulesTable('v2', b) },
  'rules-v3': { buckets: 101, note: 'as rules-v2 on rules v3', fill: (b) => rulesTable('v3', b) },
  'adam-pow': { buckets: 20, note: '[pow(0.9, t), pow(0.999, t)] for t = 1..200,000, bucket = floor((t - 1) / 10000)', fill: (b) => {
    const out = new Float64Array(2 * 10000);
    for (let i = 0; i < 10000; i++) { const t = b * 10000 + i + 1; out[2 * i] = Math.pow(0.9, t); out[2 * i + 1] = Math.pow(0.999, t); }
    return { xs: null, ys: out, rowLen: 2, first: b * 10000 + 1 };
  } },
};
function rulesTable(version, b) {
  const s0 = b * DOMAIN_SCORE_BUCKET, s1 = Math.min(1000000, s0 + DOMAIN_SCORE_BUCKET - 1);
  const L = RULE_FIELDS.length, out = new Float64Array((s1 - s0 + 1) * L), r = {};
  for (let s = s0, k = 0; s <= s1; s++) {
    C._internal.rulesAt(C.difficulty(version, s), r, C.superLevel(version, s));
    for (const f of RULE_FIELDS) out[k++] = r[f];
  }
  return { ys: out, rowLen: L, first: s0 };
}

function stageS1() {
  const fns = {};
  for (const name of Object.keys(S1_FNS)) {
    const { xs, arity, nb } = s1Inputs(name);
    const ys = s1Eval(name, xs, arity);
    const edges = S1_FNS[name].two ? EDGES2 : EDGES1;
    const ye = new Float64Array(edges.map(e => (S1_FNS[name].two ? S1_FNS[name].f(e[0], e[1]) : S1_FNS[name].f(e))));
    fns[name] = { bits: { all: long(hashTA(ys)), inputs: short(hashTA(xs)), edges: short(hashTA(ye)), buckets: pack(bucketHashes(ys, BUCKET)) }, ...(S1_FNS[name].control ? { control: true } : {}) };
  }
  const domains = {};
  for (const [name, d] of Object.entries(DOMAINS)) {
    const all = sha(), buckets = [];
    for (let b = 0; b < d.buckets; b++) { const t = d.fill(b); all.update(bufOf(t.ys)); buckets.push(hashTA(t.ys)); }
    domains[name] = { bits: { all: long(all.digest('hex')), buckets: pack(buckets) } };
  }
  // Positive control: one flipped output bit must change that bucket's hash and the full
  // hash, and nothing else.
  const { xs, arity } = s1Inputs('tanh');
  const ys = s1Eval('tanh', xs, arity), before = bucketHashes(ys, BUCKET), full = hashTA(ys);
  const u = new BigUint64Array(ys.buffer), at = 7 * BUCKET + 123;
  u[at] ^= 1n;
  const after = bucketHashes(ys, BUCKET);
  const changed = after.map((h, i) => (h !== before[i] ? i : -1)).filter(i => i >= 0);
  // NaN canonicalization: the x86 and arm64 default NaNs must hash alike, other values not.
  const nanX86 = new Float64Array(new BigUint64Array([0xFFF8000000000000n, 1n]).buffer), nanArm = new Float64Array(new BigUint64Array([0x7FF8000000000000n, 1n]).buffer);
  const nanCanon = hashTA(nanX86) === hashTA(nanArm);
  const selfTest = { flippedBucket: 7, changedBuckets: changed, fullChanged: hashTA(ys) !== full, nanCanon, pass: changed.length === 1 && changed[0] === 7 && hashTA(ys) !== full && nanCanon };
  return { fns, domains, selfTest };
}

function dump(spec) {
  const [set, bs] = spec.split(':');
  const b = Number(bs);
  if (S1_FNS[set]) {
    const { xs, arity, nb, bucketRange } = s1Inputs(set);
    if (!(b >= 0 && b < nb)) throw new Error(`${set} has buckets 0..${nb - 1}`);
    const ys = s1Eval(set, xs, arity);
    console.log(`# ${set} bucket ${b} (${bucketRange[b]}): index input${arity === 2 ? 's' : ''} output, float64 hex`);
    for (let i = b * BUCKET; i < (b + 1) * BUCKET; i++) {
      const ins = arity === 2 ? `${hex64(xs[2 * i])} ${hex64(xs[2 * i + 1])}` : hex64(xs[i]);
      console.log(`${i} ${ins} ${hex64(ys[i])}`);
    }
    return;
  }
  const d = DOMAINS[set];
  if (!d) throw new Error(`unknown set ${set}; one of ${[...Object.keys(S1_FNS), ...Object.keys(DOMAINS)].join(', ')}`);
  if (!(b >= 0 && b < d.buckets)) throw new Error(`${set} has buckets 0..${d.buckets - 1}`);
  const t = d.fill(b);
  console.log(`# ${set} bucket ${b}: ${d.note}; key then ${t.rowLen === 2 ? 'pow(0.9,t) pow(0.999,t)' : RULE_FIELDS.join(' ')}, float64 hex`);
  for (let i = 0; i * t.rowLen < t.ys.length; i++) {
    console.log(`${t.first + i} ${Array.from(t.ys.subarray(i * t.rowLen, (i + 1) * t.rowLen), hex64).join(' ')}`);
  }
}

// ---------------------------------------------------------------------------
// S3: trajectories
// ---------------------------------------------------------------------------
const { snapshot } = require(path.join(ROOT, 'test', 'golden-format.js'));
const SOLVERS = require(path.join(ROOT, 'test', 'solver.js'));
const TEACH = { jump: SOLVERS.createSolver, duck: SOLVERS.createDuckSolver };
// Random keys: each frame jump with p 0.06, down with p 0.06 (never both), from its own stream.
const randomActor = (seed) => { const r = C.createRng((seed ^ 0x5BD1E995) >>> 0); return () => { const u = r(); return { jump: u < 0.06, down: u >= 0.06 && u < 0.12 }; }; };
const S3_ARMS = [
  ['v1', 'jump', 0], ['v2', 'jump', 0], ['v2', 'duck', 30000], ['v2', 'duck', 148000], ['v2', 'duck', 998000],
  ['v3', 'duck', 0], ['v3', 'duck', 98000], ['v3', 'duck', 150000], ['v3', 'duck', 998000],
  ['v2', 'random', 200000], ['v3', 'random', 150000],
];
const S3_SEEDS = [1, 2], S3_SPAN = 3000, S3_CHECK = 1000;
const TYPE = { small: 0, large: 1, ptero: 2 }, STATE = { idle: 0, running: 1, over: 2 };

function playTrajectory(version, teacher, start, seed) {
  const base = C.createRng(seed);
  let calls = 0;
  const g = C.createGame({ rng: () => { calls++; return base(); }, version, startScore: start || undefined });
  const actor = teacher === 'random' ? randomActor(seed) : TEACH[teacher]();
  const cap = (start + S3_SPAN) / C.CFG.scoreCoef;
  const view = { get state() { return g.state; }, dino: g.dino, get speed() { return g.speed; }, get distance() { return g.distance; }, get obstacles() { return g.obstacles; }, get rngCalls() { return calls; } };
  const hb = f64Stream(), hd = f64Stream(), checks = [];
  let frames = 0;
  do {
    const k = actor(g);
    g.step(k);
    frames++;
    for (const v of snapshot(view)) hb.push(v);
    hd.push(STATE[g.state]); hd.push(g.score()); hd.push(calls); hd.push((k.jump ? 1 : 0) | (k.down ? 2 : 0));
    hd.push((g.dino.jumping ? 1 : 0) | (g.dino.ducking ? 2 : 0) | (g.dino.speedDrop ? 4 : 0)); hd.push(g.obstacles.length);
    for (const o of g.obstacles) { hd.push(TYPE[o.t.name]); hd.push(o.size); hd.push(o.y); }
    if (frames % S3_CHECK === 0) checks.push(hb.copyDigest());
  } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
  return {
    discrete: { frames, score: g.score(), crashed: g.state === 'over', rngCalls: calls, hash: short(hd.copyDigest()) },
    bits: { hash: long(hb.copyDigest()), checkpoints: pack(checks) },
  };
}

function stageS3() {
  const arms = {};
  for (const [v, t, s] of S3_ARMS) for (const seed of S3_SEEDS) arms[`${v}-${t}-${s}-s${seed}`] = playTrajectory(v, t, s, seed);
  return { span: S3_SPAN, checkpointEvery: S3_CHECK, arms };
}

// ---------------------------------------------------------------------------
// S4: demo sets, regenerated in memory with bc.genSeed in seed order (byte for byte what
// bc.generateDemos writes and hashes). Parameters copied from the manifests (the first is
// gitignored on the Mac and absent on the desktop).
// ---------------------------------------------------------------------------
const bc = require(path.join(ROOT, 'trainer', 'bc.js'));
const SEEDS_1_40 = Array.from({ length: 40 }, (_, i) => i + 1);
const DEMO_SETS = {
  // runs/exp-bc-e2e/demos.json (Mac only, gitignored); AGENTS.md "jump demos 3a67bfd4…"
  'v2-demos-jump-s1-40': { gameVersion: 'v2', teacher: 'jump', seeds: SEEDS_1_40, levels: [0, 2000, 50000, 300000, 900000], span0: 3000, span: 5000, keepBoring: 10, expect: { sha256: '3a67bfd4e109d6bbee7a6eb21b6fa7922eb254cc0fc92dc7695538a222bcfd09', rows: 2096268 } },
  // runs/exp-bc-duck/v2-demos-duck-s1-40/demos.json
  'v2-demos-duck-s1-40': { gameVersion: 'v2', teacher: 'duck', seeds: SEEDS_1_40, levels: [0, 2000, 50000, 300000, 900000], span0: 3000, span: 5000, keepBoring: 10, expect: { sha256: '363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7', rows: 2062427 } },
  // runs/exp-v3-clones/v3-demos-duck-s1-40/demos.json
  'v3-demos-duck-s1-40': { gameVersion: 'v3', teacher: 'duck', seeds: SEEDS_1_40, levels: [0, 2000, 50000, 98000, 101000, 127000, 162000, 207000, 264000, 336000, 428000, 546000, 695000, 886000, 995000], span0: 3000, span: 5000, keepBoring: 10, expect: { sha256: 'b1bded8653b3eaf5bc74bd1c3881b60e9ccd43bef1e960ad6e31580dc2ce7154', rows: 3129615 } },
};
// Feature names from policy.js observe() (as of this probe's commit).
const OBS_NAMES = ['height', 'vy', 'jumping', 'ducking', 'speed'];
for (let k = 0; k < P.OBS_K; k++) for (const n of ['present', 'dx', 'y', 'w', 'h', 'type.small', 'type.large', 'type.ptero', 'size', 'ptero.y0', 'ptero.y1', 'ptero.y2', 'vel']) OBS_NAMES.push(`o${k}.${n}`);
const COLUMN_NAMES = [...OBS_NAMES, 'label', 'weight'];

function demoSet(o) {
  const R = bc.ROW, D = P.OBS_DIM;
  const all = sha(), cols = COLUMN_NAMES.map(() => sha()), levels = o.levels.map(() => sha());
  const counts = { rows: [0, 0, 0], weighted: [0, 0, 0] }, eps = sha();
  let rows = 0, crashes = 0;
  for (const seed of o.seeds) {
    const r = bc.genSeed(seed, { levels: o.levels, span0: o.span0, span: o.span, keepBoring: o.keepBoring, gameVersion: o.gameVersion, teacher: o.teacher });
    const n = r.rows.length / R;
    all.update(bufOf(r.rows));
    let at = 0;
    r.episodes.forEach((e, li) => {
      levels[li].update(bufOf(r.rows.subarray(at * R, (at + e.rows) * R)));
      at += e.rows;
      eps.update(`${e.seed},${e.level},${e.rows},${e.crashed};`);
      if (e.crashed) crashes++;
    });
    const col = new Float32Array(n);
    for (let c = 0; c < R; c++) {
      for (let i = 0; i < n; i++) col[i] = r.rows[i * R + c];
      cols[c].update(bufOf(col));
    }
    for (let i = 0; i < n; i++) { const y = r.rows[i * R + D]; counts.rows[y]++; counts.weighted[y] += r.rows[i * R + D + 1]; }
    rows += n;
  }
  const colHex = cols.map(h => h.digest('hex'));
  const sha256 = all.digest('hex');
  return {
    discrete: { rows, counts, solverCrashes: crashes, episodes: short(eps.digest('hex')), label: short(colHex[D]), weight: short(colHex[D + 1]) },
    // levels: packed in o.levels order; features: packed in COLUMN_NAMES order (the 44 features).
    bits: { sha256, levels: pack(levels.map(h => h.digest('hex'))), features: pack(colHex.slice(0, D)) },
    matchesManifest: sha256 === o.expect.sha256,
  };
}

function stageS4() {
  const sets = {};
  for (const [name, o] of Object.entries(DEMO_SETS)) sets[name] = demoSet(o);
  return { sets };
}

// ---------------------------------------------------------------------------
// S5: decisions on the assess held-out seeds (trainer/env.js EVAL_SEEDS, offset 0), in
// process: the JS net (policy.createAgent, as assess) and the wasm kernel (as --wasm),
// whose logits() give the top-2 margin. Plus a late arm from 150k and 998k.
// ---------------------------------------------------------------------------
const env = require(path.join(ROOT, 'trainer', 'env.js'));
const S5_MODELS = { 'root-model.js': 'model.js', 'v2-bc-jump-d2w32-s1': 'runs/v2-bc-jump-d2w32-s1/best.json', 'v3-bc-duck-d2w32-s1': 'runs/exp-v3-clones/v3-bc-duck-d2w32-s1/best.json' };
const S5_CAP = 20000, S5_LATE = { seeds: 5, starts: [150000, 998000], span: 3000 };
const loadModel = (rel) => {
  const f = path.join(ROOT, rel);
  if (rel.endsWith('.json')) return JSON.parse(fs.readFileSync(f, 'utf8'));
  const m = require(f);
  return m && m.params ? m : globalThis.DINO_MODEL;
};

function playModel(model, seeds, starts, span, kind) {
  const wasm = require(path.join(ROOT, 'trainer', 'wasm-net.js'));
  const acts = f64Stream(), logits = f64Stream();
  let minMargin = Infinity, decisions = 0;
  const scores = [];
  let makeAgent;
  if (kind === 'js') {
    makeAgent = (m) => { const a = P.createAgent(m); return { act(g) { const k = a.act(g); acts.push((k.jump ? 1 : 0) | (k.down ? 2 : 0)); return k; } }; };
  } else {
    const net = wasm.createWasmNet(model.layers, model.params);
    const probe = { input: net.input, decide() {
      const b = net.decide(), z = net.logits();
      decisions++;
      let m1 = -Infinity, m2 = -Infinity;
      for (const v of z) { logits.push(v); if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v; }
      if (m1 - m2 < minMargin) minMargin = m1 - m2;
      return b;
    } };
    makeAgent = (m) => { const a = wasm.createWasmAgent(m, probe); return { act(g) { const k = a.act(g); acts.push((k.jump ? 1 : 0) | (k.down ? 2 : 0)); return k; } }; };
  }
  seeds.forEach((s, i) => {
    const start = starts ? starts[i] : 0;
    scores.push(env.runEpisode(model, s, start + span, start || undefined, makeAgent).score);
  });
  const out = { discrete: { scores, actions: short(acts.copyDigest()), frames: acts.n } };
  if (kind === 'wasm') out.bits = { logits: short(logits.copyDigest()), minMargin: hex64(minMargin), minMarginValue: minMargin, decisions };
  return out;
}

function stageS5() {
  const models = {};
  const lateSeeds = [], lateStarts = [];
  for (const st of S5_LATE.starts) for (let i = 0; i < S5_LATE.seeds; i++) { lateSeeds.push(env.EVAL_SEEDS[i]); lateStarts.push(st); }
  for (const [name, rel] of Object.entries(S5_MODELS)) {
    const model = loadModel(rel);
    const r = { gameVersion: model.gameVersion, layers: model.layers.join('-') };
    for (const kind of ['js', 'wasm']) {
      r[kind] = playModel(model, env.EVAL_SEEDS, null, S5_CAP, kind);
      r[`${kind}-late`] = playModel(model, lateSeeds, lateStarts, S5_LATE.span, kind);
    }
    r.jsWasmAgree = r.js.discrete.actions === r.wasm.discrete.actions && r['js-late'].discrete.actions === r['wasm-late'].discrete.actions;
    models[name] = r;
  }
  return { cap: S5_CAP, seeds: 'EVAL_SEEDS[0..24]', late: S5_LATE, models };
}

// ---------------------------------------------------------------------------
// S6: BC node backend (test/bc.test.js pin recipe), ES primitives.
// ---------------------------------------------------------------------------
const es = require(path.join(ROOT, 'trainer', 'es.js'));
const SMALL = { seeds: [1, 2], levels: [0], span0: 300, span: 5000, keepBoring: 10, gameVersion: 'v2' };   // test/bc.test.js
const PIN_OPTS = { hidden: [4], epochs: 1, batch: 64, lr: 1e-2, seed: 1 };                                  // test/bc.test.js
const PINS = { // test/bc.test.js SMALL_JUMP_SHA256, STUB_DUCK_SHA256, PINS
  jump: { demos: 'cf4962aafb79e614013399ffc68ee21181190f0ee92c87953458cb0256463890', node: '27e195823b0d766363eb0e0cda76f037922d9a47a8eba9464cc1d05c71f1b90c', python: 'a21360e68e93af0a23b7b0d47f05b6f4c0c9b7ef993730241bae40165ef9ae67' },
  duck: { demos: 'a468d45a7bc23c3ea78d5ac947b722803ffb9ba42075d9f5e619b8903335e8c1', node: '5674e0f0aecefb95c7e24ea4b986a1f4c8779b46e139280fd5061872e6d378a5', python: 'e4ed1f4e674eda8d08499f9cf32473f75b1f1373a0a6db1a4424a8e54ec9af83' },
};
// Mirrors test/bc.test.js createStubDuck: the jump solver, holding down on every 5th
// grounded no-op running frame.
function createStubDuck() {
  const sol = SOLVERS.createSolver();
  let k = 0;
  return (g) => {
    const act = sol(g);
    if (g.state === 'running' && !g.dino.jumping && !act.jump && k++ % 5 === 0) return { jump: false, down: true };
    return act;
  };
}

let scratch = null;
const scratchDir = () => {
  if (!scratch) {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-x86-probe-'));
    process.on('exit', () => fs.rmSync(scratch, { recursive: true, force: true }));
  }
  return scratch;
};
// SMALL demos (as bc.generateDemos with workers 1) written to the scratch dir; returns the .bin path.
function smallDemos(which) {
  const opt = { levels: SMALL.levels, span0: SMALL.span0, span: SMALL.span, keepBoring: SMALL.keepBoring, gameVersion: SMALL.gameVersion,
    teacher: which, ...(which === 'duck' ? { createTeacher: createStubDuck } : {}) };
  const parts = SMALL.seeds.map(s => bc.genSeed(s, opt).rows);
  const file = path.join(scratchDir(), `small-${which}.bin`);
  fs.writeFileSync(file, Buffer.concat(parts.map(bufOf)));
  return file;
}
const logBits = (log) => log.map(r => ({ loss: hex64(r.loss), recall: Object.fromEntries(Object.entries(r.recall).map(([k, v]) => [k, hex64(v)])) }));

function stageS6() {
  const bcOut = {};
  for (const which of ['jump', 'duck']) {
    const file = smallDemos(which);
    const demosSha = bc.sha256File(file);
    const data = bc.loadDemos(file);
    const pin = bc.trainNode(data, PIN_OPTS);
    const r = {
      discrete: { rows: data.n },
      bits: { demos: demosSha, pin: bc.paramsSha256(pin.params), pinLog: logBits(pin.log) },
      matches: { demos: demosSha === PINS[which].demos, nodePin: bc.paramsSha256(pin.params) === PINS[which].node },
    };
    if (which === 'jump') {
      const big = bc.trainNode(data, { hidden: [32, 32], epochs: 2, batch: 64, lr: 2e-3, seed: 7 });
      r.bits.d2w32 = bc.paramsSha256(big.params);
      r.bits.d2w32Log = logBits(big.log);
    }
    bcOut[which] = r;
  }
  const init = {};
  for (const h of [[4], [32, 32]]) init[h.join('x')] = short(hashTA(P.initParams([P.OBS_DIM, ...h, 3], es.gaussian(C.createRng(1)))));
  const noise = sha();
  for (let s = 1; s <= 8; s++) noise.update(bufOf(es.noise(s, 4096)));
  const theta = es.noise(100, 2048), state = { m: new Float64Array(2048), v: new Float64Array(2048), t: 0 };
  for (let t = 0; t < 50; t++) es.adamStep(theta, es.noise(1000 + t, 2048), state, 0.01, 0.005, t % 2 ? 'decoupled' : 'coupled');
  return { bc: bcOut, es: { bits: { init, noise: short(noise.digest('hex')), adam: short(hashTA(theta)) } } };
}

// ---------------------------------------------------------------------------
// S7: python backend (optional)
// ---------------------------------------------------------------------------
async function stageS7(python) {
  if (!python) return { skipped: 'no --python and no PYTHON env var' };
  const info = spawnSync(python, ['-c', 'import sys, platform, numpy; print(sys.version.split()[0], numpy.__version__, platform.machine())'], { encoding: 'utf8' });
  if (info.error || info.status !== 0) return { skipped: `no numpy at ${python}` };
  process.env.PYTHON = python;   // bc.pythonPath reads it
  const [pyVersion, numpy, machine] = info.stdout.trim().split(' ');
  const out = { env: { python: pyVersion, numpy, machine } };
  for (const which of ['jump', 'duck']) {
    const file = smallDemos(which);
    const r = await bc.trainPython(file, path.join(scratchDir(), `py-${which}.json`), PIN_OPTS, ROOT);
    out[which] = { bits: { pin: bc.paramsSha256(r.params) }, matches: { pythonPin: bc.paramsSha256(r.params) === PINS[which].python } };
  }
  return out;
}

// ---------------------------------------------------------------------------
// S2: in-play capture (runs last: replacing Math.* deoptimizes the builtins' call sites)
// ---------------------------------------------------------------------------
const CAPTURE_FNS = ['tanh', 'exp', 'log', 'log10', 'log2', 'pow', 'sin', 'cos', 'sqrt', 'expm1', 'log1p', 'sinh', 'cosh', 'atan', 'atan2', 'cbrt'];
function capture(run) {
  const orig = {}, rec = {};
  for (const n of CAPTURE_FNS) {
    const f = orig[n] = Math[n];
    const r = rec[n] = { calls: 0, h: f64Stream(), distinct: new Set(), min: Infinity, max: -Infinity };
    Math[n] = function (a, b) {
      const y = f(a, b);
      r.calls++;
      r.h.push(a); if (f.length === 2) r.h.push(b); r.h.push(y);
      if (r.distinct.size < 1e6) r.distinct.add(a);
      if (a < r.min) r.min = a;
      if (a > r.max) r.max = a;
      return y;
    };
  }
  try { run(); } finally { for (const n of CAPTURE_FNS) Math[n] = orig[n]; }
  const out = {};
  for (const [n, r] of Object.entries(rec)) {
    if (!r.calls) continue;
    out[n] = { discrete: { calls: r.calls, distinctInputs: r.distinct.size }, bits: { hash: short(r.h.copyDigest()), min: r.min, max: r.max } };
  }
  return out;
}

function stageS2() {
  const game = (version, teacher, start, frames) => () => {
    const g = C.createGame({ rng: C.createRng(11), version, startScore: start || undefined });
    const sol = TEACH[teacher]();
    for (let f = 0; f < frames && g.state !== 'over'; f++) g.step(sol(g));
  };
  const policy = (rel, start, frames) => () => {
    const model = loadModel(rel);
    const g = C.createGame({ rng: C.createRng(env.EVAL_SEEDS[0]), version: model.gameVersion, startScore: start || undefined });
    const a = P.createAgent(model);
    for (let f = 0; f < frames && g.state !== 'over'; f++) g.step(a.act(g));
  };
  const activities = {
    'solver-jump v2 0 (crosses 1000)': game('v2', 'jump', 0, 8000),
    'solver-duck v2 148k': game('v2', 'duck', 148000, 2000),
    'solver-duck v2 998k': game('v2', 'duck', 998000, 2000),
    'solver-duck v3 0': game('v3', 'duck', 0, 2000),
    'solver-duck v3 101k': game('v3', 'duck', 101000, 2000),
    'solver-duck v3 998k': game('v3', 'duck', 998000, 2000),
    'policy root model v2 0, 400 frames': policy('model.js', 0, 400),
    'policy v3-bc-duck-d2w32-s1 150k, 400 frames': policy(S5_MODELS['v3-bc-duck-d2w32-s1'], 150000, 400),
    'bc trainNode pin (SMALL jump, 1 epoch, [4])': () => bc.trainNode(bc.loadDemos(smallDemos('jump')), PIN_OPTS),
    'es noise (seed 1, 1000)': () => es.noise(1, 1000),
  };
  const out = {};
  for (const [name, run] of Object.entries(activities)) out[name] = capture(run);
  return { activities: out };
}

// ---------------------------------------------------------------------------
// S0 and main
// ---------------------------------------------------------------------------
const TRACKED = ['core.js', 'policy.js', 'test/solver.js', 'test/golden-format.js', 'trainer/bc.js', 'trainer/es.js', 'trainer/env.js', 'trainer/wasm-net.js',
  'trainer/bc_train.py', 'trainer/bc_common.py', 'model.js', ...Object.values(S5_MODELS).filter(f => f !== 'model.js'), 'research/scripts/x86-determinism/probe.js'];

function stageS0() {
  const git = (args) => { try { return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { return null; } };
  const status = git(['status', '--porcelain', '--untracked-files=no']);
  const v = process.config.variables || {};
  return {
    arch: process.arch, platform: process.platform, node: process.version, v8: process.versions.v8,
    build: { clang: v.clang, llvm: v.llvm_version || null, gas: v.gas_version || null, host_arch: v.host_arch, target_arch: v.target_arch },
    osRelease: os.release(), wsl: /microsoft/i.test(os.release()), cpu: (os.cpus()[0] || {}).model || null, cpus: os.cpus().length,
    gitHead: git(['rev-parse', 'HEAD']), gitDirty: status === null ? null : status.length > 0,
    coreHash: require(path.join(ROOT, 'trainer', 'store.js')).coreHash(),
    files: Object.fromEntries(TRACKED.map(f => [f, fileSha(f)])),
  };
}

function summarise(id, s) {
  if (!s || s.error) return `${id}: ERROR ${s && s.error}`;
  switch (id) {
    case 'S0': return `S0 env: ${s.platform} ${s.arch} node ${s.node} V8 ${s.v8} clang=${s.build.clang} | ${s.cpu} | head ${String(s.gitHead).slice(0, 10)}${s.gitDirty ? ' (dirty)' : ''} | core ${s.coreHash.slice(0, 12)}`;
    case 'S1': return `S1 libm: ${Object.entries(s.fns).map(([n, f]) => `${n}=${f.bits.all.slice(0, 8)}`).join(' ')} | ${Object.entries(s.domains).map(([n, d]) => `${n}=${d.bits.all.slice(0, 8)}`).join(' ')} | selfTest ${s.selfTest.pass ? 'pass' : 'FAIL'}`;
    case 'S2': return `S2 capture: ${Object.entries(s.activities).map(([n, a]) => `[${n}] ${Object.entries(a).map(([f, r]) => `${f}:${r.discrete.calls}`).join(',') || '-'}`).join(' ')}`;
    case 'S3': return `S3 traj: ${Object.keys(s.arms).length} arms, ${Object.values(s.arms).reduce((a, r) => a + r.discrete.frames, 0)} frames, crashes ${Object.values(s.arms).filter(r => r.discrete.crashed).length}`;
    case 'S4': return `S4 demos: ${Object.entries(s.sets).map(([n, d]) => `${n} ${d.bits.sha256.slice(0, 8)} rows ${d.discrete.rows} manifest ${d.matchesManifest ? 'MATCH' : 'differs'}`).join(' | ')}`;
    case 'S5': return `S5 decide: ${Object.entries(s.models).map(([n, m]) => `${n} min score ${Math.min(...m.js.discrete.scores)} margin ${m.wasm.bits.minMarginValue.toExponential(2)} js=wasm ${m.jsWasmAgree}`).join(' | ')}`;
    case 'S6': return `S6 train: ${Object.entries(s.bc).map(([n, r]) => `${n} demos ${r.matches.demos ? 'MATCH' : 'differs'} nodePin ${r.matches.nodePin ? 'MATCH' : 'differs'}`).join(' | ')} | es noise ${s.es.bits.noise}`;
    case 'S7': return s.skipped ? `S7 python: skipped (${s.skipped})` : `S7 python: ${s.env.python} numpy ${s.env.numpy} ${s.env.machine} | ${['jump', 'duck'].map(w => `${w} pin ${s[w].matches.pythonPin ? 'MATCH' : 'differs'}`).join(' ')}`;
    default: return id;
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const opt = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
  if (opt('--dump')) return dump(opt('--dump'));
  const only = opt('--only') ? opt('--only').split(',') : null;
  const python = opt('--python') || process.env.PYTHON || null;
  const order = ['S0', 'S1', 'S3', 'S4', 'S5', 'S6', 'S7', 'S2'];
  const run = { S0: stageS0, S1: stageS1, S2: stageS2, S3: stageS3, S4: stageS4, S5: stageS5, S6: stageS6, S7: () => stageS7(python) };
  const result = { format: FORMAT, stages: {}, timings: {} };
  const t0 = Date.now();
  for (const id of order) {
    if (only && id !== 'S0' && !only.includes(id)) continue;
    const t = Date.now();
    try { result.stages[id] = await run[id](); } catch (e) { result.stages[id] = { error: String(e && e.stack || e).split('\n').slice(0, 3).join(' | ') }; }
    result.timings[id] = (Date.now() - t) / 1000;
    console.error(summarise(id, result.stages[id]) + `  (${result.timings[id].toFixed(1)} s)`);
  }
  result.timings.total = (Date.now() - t0) / 1000;
  const text = JSON.stringify(result);
  if (opt('--out')) fs.writeFileSync(opt('--out'), text + '\n');
  console.log(text);
  console.error(`probe done in ${result.timings.total.toFixed(1)} s; ${text.length} bytes of JSON${opt('--out') ? ` written to ${opt('--out')}` : ''}`);
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { S1_FNS, DOMAINS, BUCKET, COLUMN_NAMES, DEMO_SETS, S5_MODELS, TRACKED, FORMAT };
