'use strict';
// Behaviour cloning of the solver (test/solver.js): demonstration generation, the
// node training backend, and the trained-net -> best.json assembly.
//
// Demonstrations (bc-gen): for each seed, then each start level, play the solver
// on a fresh v2 game from that level for `span` points. Every running frame gives
// one Float32 row [policy.observe features (OBS_DIM), label, weight], where label
// is the solver's action (1 = jump key down, 0 = none; it never ducks). The
// solver reads hidden state; the rows hold only what policy.observe exposes.
// Boring frames (grounded, label none, nearest visible obstacle more than 40
// frames away or absent) are kept 1 in keepBoring with weight keepBoring; the
// counter resets every episode, so the output does not depend on worker count.
//
// Training (both backends, same algorithm): tanh MLP in policy.createNet's flat
// layout, init N(0, 1/sqrt(fan_in)) weights and zero biases, softmax cross-entropy
// weighted per row by weight * classWeight[label] (classWeight_c = sqrt(total /
// (2 count_c)) over weighted counts), normalised by the batch weight sum; Adam
// (0.9, 0.999, 1e-8) with a cosine learning rate over epochs * floor(N / batch)
// steps; a fresh permutation each epoch, remainder dropped.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { isMainThread, parentPort, workerData, Worker } = require('worker_threads');
const Core = require('../core.js');
const P = require('../policy.js');
const { createSolver } = require('../test/solver.js');
const { gaussian } = require('./es.js');

const DEMOS_FORMAT = 'dino-bc-demos-v1';
const OBS_DIM = P.OBS_DIM;
const ROW = OBS_DIM + 2;                 // features, label, weight
const GEN_DEFAULTS = { levels: [0, 2000, 50000, 300000, 900000], span0: 3000, span: 5000, keepBoring: 10, gameVersion: 'v2' };
const TRAIN_DEFAULTS = { epochs: 15, batch: 1024, lr: 2e-3, seed: 1 };
const RECALL_SUBSAMPLE = 200000;

// ---------------------------------------------------------------------------
// Demonstrations
// ---------------------------------------------------------------------------
// The prototype's (research/scripts/feature-sufficiency/cloneB.js) sampling rule,
// with its Float32 arithmetic: tau0 (frames until obstacle 0 reaches the dino
// front, /30; absent: 80/30) was stored in a Float32Array before the comparison.
function isBoring(x, label) {
  if (x[2] || label !== 0) return false;
  const tau0 = Math.fround(x[5] ? Math.min(80, (x[6] * Core.W) / (x[17] * 13)) / 30 : 80 / 30);
  return tau0 > 40 / 30;
}

// One episode: the solver from startScore `level` for `span` points. push(x, label, weight).
function genEpisode(seed, level, span, keepBoring, gameVersion, push) {
  const g = Core.createGame({ rng: Core.createRng(seed), version: gameVersion, startScore: level || undefined });
  const sol = createSolver();
  const cap = (level + span) / Core.CFG.scoreCoef;
  const x = new Float32Array(OBS_DIM);
  let boring = 0;
  do {
    const act = sol(g);
    if (g.state === 'running') {
      const label = act.jump ? 1 : 0;
      P.observe(g, x);
      if (!isBoring(x, label)) push(x, label, 1);
      else if (boring++ % keepBoring === 0) push(x, label, keepBoring);
    }
    g.step(act);
  } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
  return { crashed: g.state === 'over' };
}

// All episodes of one seed. Returns { rows: Float32Array, episodes }.
function genSeed(seed, o) {
  let buf = new Float32Array(ROW * 65536), n = 0;
  const episodes = [];
  for (const level of o.levels) {
    const before = n;
    const counts = [0, 0, 0];
    const push = (x, label, w) => {
      if ((n + 1) * ROW > buf.length) { const b = new Float32Array(buf.length * 2); b.set(buf); buf = b; }
      buf.set(x, n * ROW);
      buf[n * ROW + OBS_DIM] = label;
      buf[n * ROW + OBS_DIM + 1] = w;
      counts[label] += w;
      n++;
    };
    const r = genEpisode(seed, level, level === 0 ? o.span0 : o.span, o.keepBoring, o.gameVersion, push);
    episodes.push({ seed, level, rows: n - before, weighted: counts, crashed: r.crashed });
  }
  return { rows: buf.slice(0, n * ROW), episodes };
}

// Run genSeed over seeds on `workers` threads; onSeed(i, result) is called in seed order.
function genSeedsParallel(seeds, o, workers, onSeed) {
  if (workers <= 1) {
    seeds.forEach((s, i) => onSeed(i, genSeed(s, o)));
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const pending = new Map();
    let next = 0, emitted = 0, failed = false;
    const pool = Array.from({ length: Math.min(workers, seeds.length) }, () => new Worker(__filename, { workerData: { bcGen: o } }));
    const done = (err) => {
      if (failed) return;
      if (err) failed = true;
      Promise.all(pool.map(w => w.terminate())).then(() => (err ? reject(err) : resolve()));
    };
    const feed = (w) => { if (next < seeds.length) { w.postMessage({ i: next, seed: seeds[next] }); next++; } };
    for (const w of pool) {
      w.on('message', ({ i, rows, episodes }) => {
        pending.set(i, { rows: new Float32Array(rows), episodes });
        while (pending.has(emitted)) {
          try { onSeed(emitted, pending.get(emitted)); } catch (e) { return done(e); }
          pending.delete(emitted);
          emitted++;
        }
        if (emitted === seeds.length) return done();
        feed(w);
      });
      w.on('error', done);
      feed(w);
    }
  });
}

if (!isMainThread && workerData && workerData.bcGen) {
  parentPort.on('message', ({ i, seed }) => {
    const r = genSeed(seed, workerData.bcGen);
    parentPort.postMessage({ i, rows: r.rows.buffer, episodes: r.episodes }, [r.rows.buffer]);
  });
}

// Write demos.bin and demos.json into dir. o: { seeds, levels, span0, span, keepBoring,
// gameVersion, workers, coreHash }. Returns the manifest.
async function generateDemos(dir, o) {
  const opt = { ...GEN_DEFAULTS, ...o };
  fs.mkdirSync(dir, { recursive: true });
  const binFile = path.join(dir, 'demos.bin');
  const tmp = `${binFile}.${process.pid}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  const sha = crypto.createHash('sha256');
  const episodes = [];
  const weighted = [0, 0, 0], rowsByClass = [0, 0, 0];
  let rows = 0;
  const genOpts = { levels: opt.levels, span0: opt.span0, span: opt.span, keepBoring: opt.keepBoring, gameVersion: opt.gameVersion };
  try {
    await genSeedsParallel(opt.seeds, genOpts, opt.workers || 1, (i, r) => {
      const bytes = Buffer.from(r.rows.buffer, r.rows.byteOffset, r.rows.byteLength);
      fs.writeSync(fd, bytes);
      sha.update(bytes);
      const n = r.rows.length / ROW;
      for (let k = 0; k < n; k++) {
        const label = r.rows[k * ROW + OBS_DIM];
        rowsByClass[label]++;
        weighted[label] += r.rows[k * ROW + OBS_DIM + 1];
      }
      rows += n;
      episodes.push(...r.episodes);
      if (opt.onProgress) opt.onProgress(i + 1, opt.seeds.length, rows);
    });
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, binFile);
  const manifest = {
    format: DEMOS_FORMAT, file: 'demos.bin', obsDim: OBS_DIM, rowFloats: ROW, layout: '[observe features..., label, weight] float32 little-endian',
    coreHash: opt.coreHash, gameVersion: opt.gameVersion, teacher: 'test/solver.js',
    seeds: opt.seeds, levels: opt.levels, span0: opt.span0, span: opt.span, keepBoring: opt.keepBoring,
    rows, counts: { rows: { none: rowsByClass[0], jump: rowsByClass[1], duck: rowsByClass[2] },
      weighted: { none: weighted[0], jump: weighted[1], duck: weighted[2] } },
    solverCrashes: episodes.filter(e => e.crashed).length,
    sha256: sha.digest('hex'), createdAt: new Date().toISOString(),
    episodes: episodes.map(e => ({ seed: e.seed, level: e.level, rows: e.rows, crashed: e.crashed })),
  };
  return manifest;
}

// Resolve --demos (a run dir, a demos.json, or a .bin with a sibling .json) to { manifestFile, binFile, manifest }.
function resolveDemos(p) {
  let manifestFile = p;
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) manifestFile = path.join(p, 'demos.json');
  else if (p.endsWith('.bin')) manifestFile = p.slice(0, -4) + '.json';
  if (!fs.existsSync(manifestFile)) throw new Error(`no demos manifest at ${manifestFile} (run bc-gen first)`);
  const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
  if (manifest.format !== DEMOS_FORMAT) throw new Error(`${manifestFile}: unsupported demos format ${manifest.format}`);
  const binFile = path.join(path.dirname(manifestFile), manifest.file);
  return { manifestFile, binFile, manifest };
}

const sha256File = (file) => {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.alloc(1 << 24);
  try {
    let n;
    while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) h.update(buf.subarray(0, n));
  } finally {
    fs.closeSync(fd);
  }
  return h.digest('hex');
};

// Check a demos set against the current core and its own manifest. Throws unless
// the core hash matches (or force) and the file matches the manifest's size and sha256.
function checkDemos({ manifestFile, binFile, manifest }, hash, force, warn = console.warn) {
  if (manifest.obsDim !== OBS_DIM || manifest.rowFloats !== ROW) throw new Error(`${manifestFile}: obsDim ${manifest.obsDim} does not match policy.js OBS_DIM ${OBS_DIM}`);
  if (manifest.coreHash !== hash) {
    if (!force) {
      throw new Error(`core hash mismatch: demos were generated on ${String(manifest.coreHash).slice(0, 12)}, current core/policy is ${hash.slice(0, 12)}. ` +
        'Regenerate them with bc-gen, or pass --force to train on them anyway.');
    }
    warn(`WARNING: demos core hash mismatch (${String(manifest.coreHash).slice(0, 12)} -> ${hash.slice(0, 12)}), continuing because of --force`);
  }
  const size = fs.statSync(binFile).size;
  if (size !== manifest.rows * ROW * 4) throw new Error(`${binFile}: ${size} bytes, manifest says ${manifest.rows} rows`);
  const sha = sha256File(binFile);
  if (sha !== manifest.sha256) throw new Error(`${binFile}: sha256 ${sha.slice(0, 12)} does not match the manifest's ${String(manifest.sha256).slice(0, 12)}`);
  return sha;
}

// Load demos.bin into { X: Float64Array N x OBS_DIM, y: Uint8Array, sw: Float64Array, n }.
function loadDemos(binFile) {
  const buf = fs.readFileSync(binFile);
  const f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = f.length / ROW;
  const X = new Float64Array(n * OBS_DIM), y = new Uint8Array(n), sw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const r = i * ROW;
    for (let j = 0; j < OBS_DIM; j++) X[i * OBS_DIM + j] = f[r + j];
    y[i] = f[r + OBS_DIM];
    sw[i] = f[r + OBS_DIM + 1];
  }
  return { X, y, sw, n, dim: OBS_DIM };
}

// ---------------------------------------------------------------------------
// Node training backend
// ---------------------------------------------------------------------------
// Per-layer views into a flat params array (policy.createNet layout).
function layerOffsets(layers) {
  const offs = [];
  let k = 0;
  for (let l = 1; l < layers.length; l++) {
    const nin = layers[l - 1], nout = layers[l];
    offs.push({ nin, nout, w: k, b: k + nin * nout });
    k += nin * nout + nout;
  }
  return offs;
}

// Weighted softmax cross-entropy over rows idx (a list of row indices) with per-row
// weights wRow. lossAndGrad returns the UNNORMALISED weighted loss sum S, and writes
// into grad the gradient of the batch loss S / sum(wRow[idx]) (the normalised loss
// the optimiser follows). Float64 throughout.
function createTrainer(layers) {
  const offs = layerOffsets(layers);
  const L = offs.length;
  const maxW = Math.max(...layers);
  const acts = layers.map(n => new Float64Array(n));      // per-row activations
  const delta = [new Float64Array(maxW), new Float64Array(maxW)];

  function lossAndGrad(params, X, dim, y, wRow, idx, grad) {
    grad.fill(0);
    let wsum = 0;
    for (const i of idx) wsum += wRow[i];
    let loss = 0;
    for (const i of idx) {
      const a0 = acts[0];
      for (let j = 0; j < dim; j++) a0[j] = X[i * dim + j];
      for (let l = 0; l < L; l++) {
        const { nin, nout, w, b } = offs[l];
        const a = acts[l], z = acts[l + 1];
        for (let o = 0; o < nout; o++) {
          let s = params[b + o];
          const row = w + o * nin;
          for (let j = 0; j < nin; j++) s += params[row + j] * a[j];
          z[o] = l < L - 1 ? Math.tanh(s) : s;
        }
      }
      const out = acts[L], nOut = layers[L];
      let mx = -Infinity;
      for (let c = 0; c < nOut; c++) if (out[c] > mx) mx = out[c];
      let se = 0;
      for (let c = 0; c < nOut; c++) se += Math.exp(out[c] - mx);
      const wi = wRow[i], yi = y[i];
      loss += wi * -(out[yi] - mx - Math.log(se));
      const scale = wi / wsum;
      let d = delta[0];
      for (let c = 0; c < nOut; c++) d[c] = (Math.exp(out[c] - mx) / se - (c === yi ? 1 : 0)) * scale;
      for (let l = L - 1; l >= 0; l--) {
        const { nin, nout, w, b } = offs[l];
        const a = acts[l];
        const dn = delta[(L - l) & 1];
        for (let o = 0; o < nout; o++) {
          const g = d[o];
          grad[b + o] += g;
          const row = w + o * nin;
          for (let j = 0; j < nin; j++) grad[row + j] += g * a[j];
        }
        if (l > 0) {
          for (let j = 0; j < nin; j++) {
            let s = 0;
            for (let o = 0; o < nout; o++) s += d[o] * params[w + o * nin + j];
            dn[j] = s * (1 - a[j] * a[j]);
          }
          d = dn;
        }
      }
    }
    return loss;
  }
  return { lossAndGrad };
}

// Class weights over weighted counts: sqrt(total / (2 count_c)), 0 for absent classes.
function classWeights(y, sw, nClasses = 3) {
  const counts = new Array(nClasses).fill(0);
  for (let i = 0; i < y.length; i++) counts[y[i]] += sw[i];
  const total = counts.reduce((a, b) => a + b, 0);
  return counts.map(c => (c > 0 ? Math.sqrt(total / (2 * c)) : 0));
}

function predict(layers, params, X, dim, i, net) {
  const x = X.subarray(i * dim, (i + 1) * dim);
  return net(x);
}

// Train on data { X, y, sw, n, dim }. o: { hidden, epochs, batch, lr, seed, onEpoch }.
// Returns { layers, params (Array), log }.
function trainNode(data, o) {
  const { X, y, sw, n, dim } = data;
  const layers = [dim, ...o.hidden, 3];
  const rng = Core.createRng(o.seed);
  const params = P.initParams(layers, gaussian(rng));
  const cw = classWeights(y, sw);
  const wRow = new Float64Array(n);
  let wTotal = 0;
  for (let i = 0; i < n; i++) { wRow[i] = sw[i] * cw[y[i]]; wTotal += wRow[i]; }
  const { lossAndGrad } = createTrainer(layers);
  const np = params.length;
  const grad = new Float64Array(np), m = new Float64Array(np), v = new Float64Array(np);
  const B = o.batch, perEpoch = Math.floor(n / B), steps = o.epochs * perEpoch;
  const perm = new Uint32Array(n);
  const log = [];
  let t = 0;
  for (let ep = 0; ep < o.epochs; ep++) {
    const t0 = Date.now();
    for (let i = 0; i < n; i++) perm[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const s = perm[i]; perm[i] = perm[j]; perm[j] = s; }
    let tot = 0;
    for (let s = 0; s < perEpoch; s++) {
      tot += lossAndGrad(params, X, dim, y, wRow, perm.subarray(s * B, (s + 1) * B), grad);
      t++;
      const lr = o.lr * 0.5 * (1 + Math.cos(Math.PI * t / steps));
      const c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t);
      for (let k = 0; k < np; k++) {
        const g = grad[k];
        m[k] = 0.9 * m[k] + 0.1 * g;
        v[k] = 0.999 * v[k] + 0.001 * g * g;
        params[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
      }
    }
    const recall = recallOn(layers, params, data, rng);
    const row = { epoch: ep, loss: tot / wTotal, recall, seconds: (Date.now() - t0) / 1000 };
    log.push(row);
    if (o.onEpoch) o.onEpoch(row);
  }
  return { layers, params: Array.from(params), log };
}

// Recall of classes none and jump on a random subsample of up to 200k rows.
function recallOn(layers, params, { X, y, n, dim }, rng) {
  const k = Math.min(n, RECALL_SUBSAMPLE);
  const pick = new Uint32Array(n);
  for (let i = 0; i < n; i++) pick[i] = i;
  for (let i = 0; i < k; i++) { const j = i + Math.floor(rng() * (n - i)); const s = pick[i]; pick[i] = pick[j]; pick[j] = s; }
  const net = P.createNet(layers, params);
  const hit = [0, 0], tot = [0, 0];
  for (let r = 0; r < k; r++) {
    const i = pick[r], c = y[i];
    if (c > 1) continue;
    tot[c]++;
    if (predict(layers, params, X, dim, i, net) === c) hit[c]++;
  }
  return { none: hit[0] / Math.max(1, tot[0]), jump: hit[1] / Math.max(1, tot[1]) };
}

// ---------------------------------------------------------------------------
// Python backend (trainer/bc_train.py)
// ---------------------------------------------------------------------------
function pythonPath(root) {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = path.join(root, '.venv', 'bin', 'python');
  return fs.existsSync(venv) ? venv : 'python3';
}

// Run bc_train.py on binFile; resolves to { layers, params, log }. onLine gets each stdout line.
// o.snapshotDir (optional): write each epoch's params to <snapshotDir>/epoch-NN.json.
function trainPython(binFile, outFile, o, root, onLine = () => {}) {
  const { spawn } = require('child_process');
  const args = [path.join(__dirname, 'bc_train.py'), '--demos', binFile, '--obs-dim', String(OBS_DIM), '--hidden', o.hidden.join(','),
    '--epochs', String(o.epochs), '--batch', String(o.batch), '--lr', String(o.lr), '--seed', String(o.seed), '--out', outFile];
  if (o.snapshotDir) args.push('--snapshot-dir', o.snapshotDir);
  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath(root), ['-W', 'ignore', ...args], { stdio: ['ignore', 'pipe', 'inherit'] });
    let buf = '';
    child.stdout.on('data', (d) => {
      buf += d;
      let k;
      while ((k = buf.indexOf('\n')) >= 0) { onLine(buf.slice(0, k)); buf = buf.slice(k + 1); }
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code !== 0) return reject(new Error(`bc_train.py exited with code ${code}`));
      const r = JSON.parse(fs.readFileSync(outFile, 'utf8'));
      resolve(r);
    });
  });
}

// ---------------------------------------------------------------------------
// best.json (Node owns the schema; same as ES runs and runs/bc-v2-h32x32)
// ---------------------------------------------------------------------------
function assembleBest({ layers, params, validation, coreHash, config }) {
  return {
    format: P.FORMAT, layers, actionRepeat: 1, gameVersion: config.gameVersion, generation: null,
    validation, coreHash, config, params,
  };
}

module.exports = {
  DEMOS_FORMAT, ROW, GEN_DEFAULTS, TRAIN_DEFAULTS, isBoring, genEpisode, genSeed, generateDemos, resolveDemos, checkDemos,
  loadDemos, sha256File, layerOffsets, createTrainer, classWeights, trainNode, pythonPath, trainPython, assembleBest,
};
