'use strict';
// Behaviour cloning of a solver teacher (test/solver.js): demonstration generation,
// the node training backend, and the trained-net -> best.json assembly.
//
// Teachers (bc-gen --teacher, TEACHERS below): 'jump' = createSolver (the default;
// it never ducks), 'duck' = createDuckSolver (jumps like createSolver, ducks under
// mid pterodactyls). Each is resolved by name when a demo set is generated, so this
// file loads on trees whose solver.js lacks createDuckSolver.
// Demonstrations (bc-gen): for each seed, then each start level, play the teacher
// on a fresh v2 game from that level for `span` points. Every running frame gives
// one Float32 row [policy.observe features (OBS_DIM), label, weight], where label
// is the teacher's action in policy.js indices (0 none, 1 jump key down, 2 down key
// down); a teacher that returns both keys throws. The teacher reads hidden state;
// the rows hold only what policy.observe exposes.
// Boring frames (grounded, label none, nearest visible obstacle more than 40
// frames away or absent) are kept 1 in keepBoring with weight keepBoring; the
// counter resets every episode, so the output does not depend on worker count.
//
// Training (both backends, same algorithm): tanh MLP in policy.createNet's flat
// layout, init N(0, 1/sqrt(fan_in)) weights and zero biases, softmax cross-entropy
// weighted per row by weight * classWeight[label] (classWeight_c = sqrt(total /
// (2 count_c)) over weighted counts), normalised by the batch weight sum; Adam
// (0.9, 0.999, 1e-8) with a cosine learning rate over epochs * floor(N / batch)
// steps; a fresh permutation each epoch, remainder dropped. The class weights cover
// all three classes (0 for an absent one), so duck demos are sqrt-balanced the same
// way. The constant 2 is kept for 3 classes on purpose: it is a uniform scale on
// every row weight, which the batch-sum normalisation (and the loss / sum(w) report)
// cancels, and keeping it leaves jump-demo training bit-identical.
// Recall (per epoch, on a 200k-row subsample) reports none and jump, plus duck when
// the demo set has any label-2 row; without duck rows the output is unchanged.
//
// Distillation (bc-train --distill-from; off by default, and then nothing below runs).
// The teacher is another trained model (a best.json); its logits z_t on every demo row
// are computed once in Node by teacherLogits (policy.createNet's summation order) and
// fed to both backends (the python one through a float64 file), so both see the same
// teacher. Row weights stay sw * classWeight[solver label] in every mode, so modes
// differ only in the target. Hinton, Vinyals & Dean 2015 (arXiv:1503.02531) section 2:
//   soft: q = softmax(z_t / tau), p_tau = softmax(z / tau);
//         per-row objective = alpha * CE(solver label, softmax(z))
//                           + (1 - alpha) * tau^2 * CE(q, p_tau);
//         gradient w.r.t. z = alpha * (softmax(z) - onehot(solver))
//                           + (1 - alpha) * tau * (p_tau - q).
//         alpha (default 0) is the weight of the solver's hard labels.
//   hard: plain CE on argmax(z_t) (first index wins ties, as policy.js); tau, alpha unused.
// With a one-hot teacher at tau 1 and alpha 0, soft is the hard-label update exactly.
// bc-metrics `loss` is always the weighted CE against the SOLVER labels (comparable
// across modes); distilled runs add `distillLoss`, the objective actually optimised.
//
// Quantization (bc-train --quant-bits, bc-quantize; off by default). Weights only, in
// symmetric integers: qmax = 2^(bits-1) - 1 (bits 2 = ternary {-1, 0, 1}), scale =
// max|w| / qmax per output row (scheme 'row') or per layer ('layer'), q = clamp(
// round-half-up(w / scale), -qmax, qmax), stored as the float64 q * scale. Biases stay
// float. QAT: each step's forward and backward use the quantized weights and the
// gradient updates the float weights (straight-through estimator); the final params are
// the quantized ones. The model stays dino-mlp-v1, so policy.js and CSS run it unchanged.
// Scheme 'pow2' (all params N-bit fixed point): weights AND biases are integers q in
// [-qmax, qmax] with ONE power-of-two scale per layer, value = q * 2^-k. k is the
// largest integer with m * 2^k <= qmax, where m = max(|W|, |b|) over the layer (k = 0
// for an all-zero layer; k may be negative), so no value is clipped; q = round-half-up(
// v * 2^k) (the clamp to +-qmax never binds). QAT fake-quantizes biases too and
// recomputes k every step; the result carries k (one integer per layer) instead of
// scales. q * 2^-k is exact in float64 (and float32), so the saved params are the
// fixed-point values exactly; research/scripts/prereg-distill/pack.js packs them.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { isMainThread, parentPort, workerData, Worker } = require('worker_threads');
const Core = require('../core.js');
const P = require('../policy.js');
const { gaussian } = require('./es.js');

const DEMOS_FORMAT = 'dino-bc-demos-v1';
const OBS_DIM = P.OBS_DIM;
const ROW = OBS_DIM + 2;                 // features, label, weight
const GEN_DEFAULTS = { levels: [0, 2000, 50000, 300000, 900000], span0: 3000, span: 5000, keepBoring: 10, gameVersion: 'v2', teacher: 'jump' };
const TRAIN_DEFAULTS = { epochs: 15, batch: 1024, lr: 2e-3, seed: 1 };
const RECALL_SUBSAMPLE = 200000;

// ---------------------------------------------------------------------------
// Teachers
// ---------------------------------------------------------------------------
// name -> the factory exported by `file` (a path from the repo root, recorded in the
// demos manifest as `teacher`; the name is recorded as `teacherName`).
const TEACHERS = {
  jump: { file: 'test/solver.js', factory: 'createSolver' },
  duck: { file: 'test/solver.js', factory: 'createDuckSolver' },
};

// The teacher factory for `name`; throws for an unknown name or a missing export.
function resolveTeacher(name) {
  const t = TEACHERS[name];
  if (!t) throw new Error(`unknown teacher ${name}; one of ${Object.keys(TEACHERS).join(', ')}`);
  const f = require(path.join(__dirname, '..', t.file))[t.factory];
  if (typeof f !== 'function') throw new Error(`${t.factory} not available on this tree (${t.file}); teacher ${name} needs it`);
  return f;
}

// policy.js action index of a teacher action; a teacher must never hold both keys.
function actionLabel(act) {
  if (act.jump && act.down) throw new Error('teacher returned both jump and down');
  return act.down ? 2 : act.jump ? 1 : 0;
}

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

// One episode: the teacher from startScore `level` for `span` points. push(x, label, weight).
// createTeacher (a factory, as in TEACHERS) defaults to the jump teacher.
// ext (optional; absent = the original rule, byte for byte):
//   scoreWindow [A, B]: keep only running frames with A <= game.score() < B. Frames outside
//     the window are neither pushed nor counted by the boring rule.
//   createStudent: a factory of actors (game -> keys). The student plays (its keys are
//     applied) and the teacher becomes a shadow: one instance, queried on every frame from
//     frame 0 on the student's trajectory, whose answer labels the row (DAgger relabelling;
//     research/prereg-teacher-duck-release-fix.md §4.1, as diag.js's oracle trace). A shadow
//     answer is defined on ground frames, and on airborne frames only after a takeoff the
//     shadow itself chose (its private `hold` is set only then); other airborne frames are
//     dropped and counted in undefinedAir. The episode ends at the student's crash or the cap.
function genEpisode(seed, level, span, keepBoring, gameVersion, push, createTeacher = resolveTeacher('jump'), ext = null) {
  const g = Core.createGame({ rng: Core.createRng(seed), version: gameVersion, startScore: level || undefined });
  const sol = createTeacher();
  const cap = (level + span) / Core.CFG.scoreCoef;
  const x = new Float32Array(OBS_DIM);
  let boring = 0;
  const win = ext && ext.scoreWindow;
  const student = ext && ext.createStudent ? ext.createStudent() : null;
  let undefinedAir = 0, jumpValid = true;
  do {
    const act = sol(g);
    const label = actionLabel(act);
    const keys = student ? student(g) : act;
    const airborne = g.dino.jumping;
    if (g.state === 'running' && (!win || (g.score() >= win[0] && g.score() < win[1]))) {
      if (student && airborne && !jumpValid) undefinedAir++;
      else {
        P.observe(g, x);
        if (!isBoring(x, label)) push(x, label, 1);
        else if (boring++ % keepBoring === 0) push(x, label, keepBoring);
      }
    }
    g.step(keys);
    if (student && !airborne && g.dino.jumping) jumpValid = !!act.jump;
  } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
  return student ? { crashed: g.state === 'over', undefinedAir } : { crashed: g.state === 'over' };
}

// Duck-transition rows (bc-train --transition-weight; research/prereg-teacher-duck-release-fix.md
// §4.1), from the row alone: grounded (x[2] = jumping is 0) and either a duck onset (x[3] =
// ducking is 0, label 2) or a release (x[3] is 1, label not 2; includes release into a jump).
// x[3] is the ducking state left by the previous frame's keys, so no frame order is needed.
// trainer/bc_train.py applies the same rule. Returns 0 (none), 1 (onset) or 2 (release).
function transitionMask(x2, x3, label) {
  if (x2 !== 0) return 0;
  if (x3 === 0 && label === 2) return 1;
  if (x3 === 1 && label !== 2) return 2;
  return 0;
}

// A student actor factory from a dino-mlp-v1 model object (best.json): policy.js createAgent.
const studentFactory = (model) => () => { const a = P.createAgent(model); return (g) => a.act(g); };

// All episodes of one seed. Returns { rows: Float32Array, episodes }. The teacher is
// o.createTeacher if given (in-thread only: tests), else TEACHERS[o.teacher || 'jump'].
function genSeed(seed, o) {
  const createTeacher = o.createTeacher || resolveTeacher(o.teacher || 'jump');
  const createStudent = o.createStudent || (o.studentModel ? studentFactory(o.studentModel) : null);
  const ext = o.scoreWindow || createStudent ? { scoreWindow: o.scoreWindow || null, createStudent } : null;
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
    const r = genEpisode(seed, level, level === 0 ? o.span0 : o.span, o.keepBoring, o.gameVersion, push, createTeacher, ext);
    episodes.push({ seed, level, rows: n - before, weighted: counts, crashed: r.crashed, ...(createStudent ? { undefinedAir: r.undefinedAir } : {}) });
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
// gameVersion, workers, coreHash, teacher (a TEACHERS name, default 'jump'),
// createTeacher (optional factory overriding the table, workers <= 1 only: tests),
// scoreWindow ([A, B], optional: genEpisode), studentModel (a dino-mlp-v1 model object,
// optional: DAgger rollouts, genEpisode) with student ({ path, paramsSha256 }, recorded),
// createStudent (optional actor factory instead of studentModel, workers <= 1 only: tests) }.
// Returns the manifest.
async function generateDemos(dir, o) {
  const opt = { ...GEN_DEFAULTS, ...o };
  opt.teacher = opt.teacher || GEN_DEFAULTS.teacher;
  if (opt.createTeacher && (opt.workers || 1) > 1) throw new Error('createTeacher cannot cross worker threads; use workers 1');
  if (!opt.createTeacher) resolveTeacher(opt.teacher);   // fail before writing anything
  fs.mkdirSync(dir, { recursive: true });
  const binFile = path.join(dir, 'demos.bin');
  const tmp = `${binFile}.${process.pid}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  const sha = crypto.createHash('sha256');
  const episodes = [];
  const weighted = [0, 0, 0], rowsByClass = [0, 0, 0];
  let rows = 0;
  if (opt.createStudent && (opt.workers || 1) > 1) throw new Error('createStudent cannot cross worker threads; use workers 1');
  if (opt.scoreWindow && !(Array.isArray(opt.scoreWindow) && opt.scoreWindow.length === 2 && opt.scoreWindow[0] < opt.scoreWindow[1])) {
    throw new Error('scoreWindow must be [A, B] with A < B');
  }
  const dagger = !!(opt.studentModel || opt.createStudent);
  const genOpts = { levels: opt.levels, span0: opt.span0, span: opt.span, keepBoring: opt.keepBoring, gameVersion: opt.gameVersion,
    teacher: opt.teacher, ...(opt.createTeacher ? { createTeacher: opt.createTeacher } : {}),
    ...(opt.scoreWindow ? { scoreWindow: opt.scoreWindow } : {}),
    ...(opt.studentModel ? { studentModel: opt.studentModel } : {}), ...(opt.createStudent ? { createStudent: opt.createStudent } : {}) };
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
    coreHash: opt.coreHash, gameVersion: opt.gameVersion, teacher: TEACHERS[opt.teacher].file, teacherName: opt.teacher,
    seeds: opt.seeds, levels: opt.levels, span0: opt.span0, span: opt.span, keepBoring: opt.keepBoring,
    rows, counts: { rows: { none: rowsByClass[0], jump: rowsByClass[1], duck: rowsByClass[2] },
      weighted: { none: weighted[0], jump: weighted[1], duck: weighted[2] } },
    ...(dagger ? {} : { solverCrashes: episodes.filter(e => e.crashed).length }),
    sha256: sha.digest('hex'), createdAt: new Date().toISOString(),
    episodes: episodes.map(e => ({ seed: e.seed, level: e.level, rows: e.rows, crashed: e.crashed, ...(dagger ? { undefinedAir: e.undefinedAir } : {}) })),
  };
  // Additive keys, present only when used (a default manifest is unchanged).
  if (opt.scoreWindow) manifest.scoreWindow = opt.scoreWindow;
  if (dagger) {
    // DAgger rollouts: the student played; the teacher (teacherName) labelled as a shadow.
    // Crashes are the student's, so they are not solverCrashes.
    manifest.source = 'dagger';
    manifest.student = opt.student || { path: null, paramsSha256: null };
    manifest.studentCrashes = episodes.filter(e => e.crashed).length;
    manifest.undefinedAir = episodes.reduce((a, e) => a + e.undefinedAir, 0);
  }
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

  // dist (optional; distillation only): { ySolver, Q (n x nOut soft targets at tau, or
  // null for hard mode, where y holds the teacher's labels), tau, alpha, solverLoss }.
  // With dist, the return value is the weighted objective sum and dist.solverLoss
  // accumulates the weighted CE against ySolver.
  function lossAndGrad(params, X, dim, y, wRow, idx, grad, dist) {
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
      const scale = wi / wsum;
      let d = delta[0];
      if (!dist) {
        loss += wi * -(out[yi] - mx - Math.log(se));
        for (let c = 0; c < nOut; c++) d[c] = (Math.exp(out[c] - mx) / se - (c === yi ? 1 : 0)) * scale;
      } else {
        const ys = dist.ySolver[i];
        const hce = -(out[ys] - mx - Math.log(se));
        dist.solverLoss += wi * hce;
        if (dist.Q) {
          const { Q, tau, alpha } = dist;
          let mt = -Infinity;
          for (let c = 0; c < nOut; c++) if (out[c] / tau > mt) mt = out[c] / tau;
          let st = 0;
          for (let c = 0; c < nOut; c++) st += Math.exp(out[c] / tau - mt);
          let sce = 0;
          for (let c = 0; c < nOut; c++) {
            const q = Q[i * nOut + c];
            if (q > 0) sce -= q * (out[c] / tau - mt - Math.log(st));
            const soft = (1 - alpha) * tau * (Math.exp(out[c] / tau - mt) / st - q);
            d[c] = (alpha > 0 ? alpha * (Math.exp(out[c] - mx) / se - (c === ys ? 1 : 0)) + soft : soft) * scale;
          }
          loss += wi * (alpha * hce + (1 - alpha) * tau * tau * sce);
        } else {
          loss += wi * -(out[yi] - mx - Math.log(se));
          for (let c = 0; c < nOut; c++) d[c] = (Math.exp(out[c] - mx) / se - (c === yi ? 1 : 0)) * scale;
        }
      }
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

// ---------------------------------------------------------------------------
// Distillation and quantization helpers
// ---------------------------------------------------------------------------
// A teacher model's logits on every row of data: Float64Array n x nOut. Each unit sums
// bias + w0*x0 + w1*x1 + ... in index order, as policy.createNet does (its 4-row blocks
// keep that order), so argmax(row) is createNet's decision (test/bc.test.js checks it).
// stride (default dim) lets X be the raw Float32 demo rows (stride ROW).
function teacherLogits(model, { X, n, dim, stride = dim }) {
  const L = model.layers, p = model.params;
  if (L[0] !== dim) throw new Error(`teacher expects ${L[0]} inputs, demos have ${dim}`);
  if (p.length !== P.numParams(L)) throw new Error(`teacher: expected ${P.numParams(L)} params, got ${p.length}`);
  const nOut = L[L.length - 1];
  const out = new Float64Array(n * nOut);
  const bufs = L.map(k => new Float64Array(k));
  for (let r = 0; r < n; r++) {
    let input = X.subarray(r * stride, r * stride + dim), k = 0;
    for (let l = 1; l < L.length; l++) {
      const nIn = L[l - 1], nO = L[l], o = bufs[l], bBase = k + nO * nIn, last = l === L.length - 1;
      for (let i = 0; i < nO; i++) {
        let s = p[bBase + i];
        const row = k + i * nIn;
        for (let j = 0; j < nIn; j++) s += p[row + j] * input[j];
        o[i] = last ? s : Math.tanh(s);
      }
      input = o; k = bBase + nO;
    }
    for (let c = 0; c < nOut; c++) out[r * nOut + c] = input[c];
  }
  return out;
}

// Per-row argmax of n x nOut logits (first index wins ties, as policy.js): Uint8Array.
function argmaxLabels(Z, nOut = 3) {
  const n = Z.length / nOut, y = new Uint8Array(n);
  for (let r = 0; r < n; r++) {
    let b = 0;
    for (let c = 1; c < nOut; c++) if (Z[r * nOut + c] > Z[r * nOut + b]) b = c;
    y[r] = b;
  }
  return y;
}

// softmax(Z / tau) per row: Float64Array n x nOut.
function softTargets(Z, tau, nOut = 3) {
  const n = Z.length / nOut, Q = new Float64Array(Z.length);
  for (let r = 0; r < n; r++) {
    let mx = -Infinity;
    for (let c = 0; c < nOut; c++) mx = Math.max(mx, Z[r * nOut + c] / tau);
    let s = 0;
    for (let c = 0; c < nOut; c++) s += (Q[r * nOut + c] = Math.exp(Z[r * nOut + c] / tau - mx));
    for (let c = 0; c < nOut; c++) Q[r * nOut + c] /= s;
  }
  return Q;
}

const QUANT_SCHEMES = ['row', 'layer', 'pow2'];
const quantMax = (bits) => 2 ** (bits - 1) - 1;

// Largest integer k with m * 2^k <= qmax (m >= 0; 0 for m = 0). Powers of two are exact,
// so the fix-up comparisons are exact; bc_train.py pow2_exponent is the same rule.
function pow2Exponent(m, qmax) {
  if (!(m > 0)) return 0;
  let k = Math.floor(Math.log2(qmax / m));
  while (m * 2 ** (k + 1) <= qmax) k++;
  while (m * 2 ** k > qmax) k--;
  return k;
}

// pow2 scheme: quantize weights and biases of every layer into out; returns k per layer.
function quantizePow2Into(layers, params, bits, out) {
  const qmax = quantMax(bits);
  return layerOffsets(layers).map(({ nin, nout, w, b }) => {
    let m = 0;
    for (let i = w; i < b + nout; i++) m = Math.max(m, Math.abs(params[i]));
    const k = pow2Exponent(m, qmax), up = 2 ** k, down = 2 ** -k;
    for (let i = w; i < b + nout; i++) out[i] = Math.min(qmax, Math.max(-qmax, Math.floor(params[i] * up + 0.5))) * down;
    return k;
  });
}
function checkQuantOptions(q) {
  if (!(Number.isInteger(q.bits) && q.bits >= 2 && q.bits <= 8)) throw new Error(`--quant-bits must be an integer in 2..8, got ${q.bits}`);
  if (!QUANT_SCHEMES.includes(q.scheme)) throw new Error(`--quant-scheme must be one of ${QUANT_SCHEMES.join(', ')}, got ${q.scheme}`);
}

// Quantize the weights of params (policy.createNet layout) into out (may be params);
// biases are copied unchanged. Returns scales: one array per layer (one entry per
// output row for 'row', one for 'layer'). See the header for the rule. Scheme 'pow2'
// also quantizes the biases and returns k (one integer per layer) instead.
function quantizeInto(layers, params, q, out) {
  if (q.scheme === 'pow2') return quantizePow2Into(layers, params, q.bits, out);
  const qmax = quantMax(q.bits);
  const scales = [];
  for (const { nin, nout, w, b } of layerOffsets(layers)) {
    const rowMax = [];
    for (let o = 0; o < nout; o++) {
      let m = 0;
      for (let j = 0; j < nin; j++) m = Math.max(m, Math.abs(params[w + o * nin + j]));
      rowMax.push(m);
    }
    const layerMax = Math.max(...rowMax);
    const sc = (m) => (m > 0 ? m / qmax : 1);
    const s = q.scheme === 'row' ? rowMax.map(sc) : [sc(layerMax)];
    for (let o = 0; o < nout; o++) {
      const so = s[q.scheme === 'row' ? o : 0];
      for (let j = 0; j < nin; j++) {
        const k = Math.min(qmax, Math.max(-qmax, Math.floor(params[w + o * nin + j] / so + 0.5)));
        out[w + o * nin + j] = k * so;
      }
      out[b + o] = params[b + o];
    }
    scales.push(s);
  }
  return scales;
}

// { params, scales } for 'row'/'layer'; { params, k } for 'pow2'.
function quantizeParams(layers, params, q) {
  const out = new Float64Array(params.length);
  const r = quantizeInto(layers, params, q, out);
  return q.scheme === 'pow2' ? { params: Array.from(out), k: r } : { params: Array.from(out), scales: r };
}

// The config.quant record for a quantized result r ({ scales } or { k }). checkQuantized
// throws unless the params are exactly on the recorded grid, so no record is written
// for params off it. pow2 records biases 'int', k per layer and checked: true.
function quantRecord(layers, params, q, mode, r) {
  const rec = q.scheme === 'pow2'
    ? { bits: q.bits, scheme: q.scheme, mode, biases: 'int', k: r.k }
    : { bits: q.bits, scheme: q.scheme, mode, biases: 'float', scales: r.scales };
  rec.levels = checkQuantized(layers, params, rec);
  if (q.scheme === 'pow2') rec.checked = true;
  return rec;
}

// Throws unless every weight is exactly k * scale with an integer |k| <= qmax, for the
// recorded scales. Returns the number of distinct integer levels used.
// pow2: every weight AND bias v of layer l must be an integer in [-qmax, qmax] times
// 2^-k[l] (k[l] an integer), exactly.
function checkQuantized(layers, params, quant) {
  const qmax = quantMax(quant.bits);
  const levels = new Set();
  if (quant.scheme === 'pow2') {
    const offs = layerOffsets(layers);
    if (!Array.isArray(quant.k) || quant.k.length !== offs.length || !quant.k.every(Number.isInteger)) throw new Error(`pow2: k must be ${offs.length} integers, got ${JSON.stringify(quant.k)}`);
    if (params.length !== P.numParams(layers)) throw new Error(`pow2: expected ${P.numParams(layers)} params, got ${params.length}`);
    offs.forEach(({ nout, w, b }, l) => {
      const up = 2 ** quant.k[l];
      for (let i = w; i < b + nout; i++) {
        const s = params[i] * up;
        if (!(Number.isInteger(s) && Math.abs(s) <= qmax && s / up === params[i])) throw new Error(`layer ${l} param ${i - w}: ${params[i]} is not an integer in [-${qmax}, ${qmax}] times 2^-${quant.k[l]}`);
        levels.add(s);
      }
    });
    return levels.size;
  }
  layerOffsets(layers).forEach(({ nin, nout, w }, l) => {
    const s = quant.scales[l];
    if (!s || s.length !== (quant.scheme === 'row' ? nout : 1)) throw new Error(`layer ${l}: scales do not match scheme ${quant.scheme}`);
    for (let o = 0; o < nout; o++) {
      const so = s[quant.scheme === 'row' ? o : 0];
      for (let j = 0; j < nin; j++) {
        const v = params[w + o * nin + j], k = Math.round(v / so);
        if (!(Math.abs(k) <= qmax && k * so === v)) throw new Error(`layer ${l} row ${o} col ${j}: ${v} is not an integer multiple (|k| <= ${qmax}) of scale ${so}`);
        levels.add(k);
      }
    }
  });
  return levels.size;
}

// Train on data { X, y, sw, n, dim }. o: { hidden, epochs, batch, lr, seed, onEpoch,
// distill (optional: { logits: Float64Array n x 3, mode: 'soft'|'hard', tau, alpha }),
// quant (optional: { bits, scheme }) }. Returns { layers, params (Array), log[, scales | k] }.
function trainNode(data, o) {
  const { X, y, sw, n, dim } = data;
  const layers = [dim, ...o.hidden, 3];
  const rng = Core.createRng(o.seed);
  const params = P.initParams(layers, gaussian(rng));
  const cw = classWeights(y, sw);
  const hasDuck = y.includes(2);
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
  let dist = null, yTrain = y;
  if (o.distill) {
    const dz = o.distill;
    if (dz.logits.length !== n * 3) throw new Error(`teacher logits: ${dz.logits.length / 3} rows, demos have ${n}`);
    dist = { ySolver: y, Q: null, tau: dz.tau, alpha: dz.alpha || 0, solverLoss: 0 };
    if (dz.mode === 'hard') yTrain = argmaxLabels(dz.logits);
    else dist.Q = softTargets(dz.logits, dz.tau);
  }
  if (o.quant) checkQuantOptions(o.quant);
  const qparams = o.quant ? new Float64Array(np) : null;
  for (let ep = 0; ep < o.epochs; ep++) {
    const t0 = Date.now();
    for (let i = 0; i < n; i++) perm[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const s = perm[i]; perm[i] = perm[j]; perm[j] = s; }
    let tot = 0;
    if (dist) dist.solverLoss = 0;
    for (let s = 0; s < perEpoch; s++) {
      if (qparams) quantizeInto(layers, params, o.quant, qparams);
      if (dist || qparams) tot += lossAndGrad(qparams || params, X, dim, yTrain, wRow, perm.subarray(s * B, (s + 1) * B), grad, dist || undefined);
      else tot += lossAndGrad(params, X, dim, y, wRow, perm.subarray(s * B, (s + 1) * B), grad);
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
    if (qparams) quantizeInto(layers, params, o.quant, qparams);
    const recall = recallOn(layers, qparams || params, data, rng, hasDuck);
    const row = dist
      ? { epoch: ep, loss: dist.solverLoss / wTotal, recall, seconds: (Date.now() - t0) / 1000, distillLoss: tot / wTotal }
      : { epoch: ep, loss: tot / wTotal, recall, seconds: (Date.now() - t0) / 1000 };
    log.push(row);
    if (o.onEpoch) o.onEpoch(row);
  }
  if (qparams) {
    const q = quantizeParams(layers, params, o.quant);
    return { layers, params: q.params, log, ...(q.k ? { k: q.k } : { scales: q.scales }) };
  }
  return { layers, params: Array.from(params), log };
}

// Recall of classes none and jump (and duck when hasDuck: the demo set has label-2
// rows) on a random subsample of up to 200k rows.
function recallOn(layers, params, { X, y, n, dim }, rng, hasDuck = false) {
  const k = Math.min(n, RECALL_SUBSAMPLE);
  const pick = new Uint32Array(n);
  for (let i = 0; i < n; i++) pick[i] = i;
  for (let i = 0; i < k; i++) { const j = i + Math.floor(rng() * (n - i)); const s = pick[i]; pick[i] = pick[j]; pick[j] = s; }
  const net = P.createNet(layers, params);
  const hit = [0, 0, 0], tot = [0, 0, 0];
  for (let r = 0; r < k; r++) {
    const i = pick[r], c = y[i];
    if (c > 2) continue;
    tot[c]++;
    if (predict(layers, params, X, dim, i, net) === c) hit[c]++;
  }
  const recall = { none: hit[0] / Math.max(1, tot[0]), jump: hit[1] / Math.max(1, tot[1]) };
  if (hasDuck) recall.duck = hit[2] / Math.max(1, tot[2]);
  return recall;
}

// ---------------------------------------------------------------------------
// Python backend (trainer/bc_train.py)
// ---------------------------------------------------------------------------
// The configured numpy interpreter: PYTHON, else <root>/.venv/bin/python, else null.
function configuredPython(root) {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = path.join(root, '.venv', 'bin', 'python');
  return fs.existsSync(venv) ? venv : null;
}
// --backend python runs the configured interpreter, else whatever python3 is on PATH.
const pythonPath = (root) => configuredPython(root) || 'python3';

// Run bc_train.py on binFile; resolves to { layers, params, log[, transitions] }. onLine gets each stdout line.
// o.snapshotDir (optional): write each epoch's params to <snapshotDir>/epoch-NN.json.
// o.extraDemos (optional): .bin files appended, in order, after binFile.
// o.transitionWeight (optional): multiply duck-transition rows' weights (transitionMask).
// o.blasThreads (optional): cap the BLAS threads of the child (VECLIB/OMP/OPENBLAS env vars);
// bc-train-batch runs several at once with 1. The output is the same bytes for any thread count
// (research/plan-gpu-trainer.md Phase 1, V1.3). o.onSpawn (optional) gets the child process.
function trainPython(binFile, outFile, o, root, onLine = () => {}) {
  const args = [path.join(__dirname, 'bc_train.py'), ...engineArgs(binFile, o), '--seed', String(o.seed), '--out', outFile];
  if (o.snapshotDir) args.push('--snapshot-dir', o.snapshotDir);
  const t = o.blasThreads !== undefined ? String(o.blasThreads) : null;
  const env = t ? { ...process.env, VECLIB_MAXIMUM_THREADS: t, OMP_NUM_THREADS: t, OPENBLAS_NUM_THREADS: t } : process.env;
  return runEngine(pythonPath(root), args, env, o, onLine, 'bc_train.py').then(() => JSON.parse(fs.readFileSync(outFile, 'utf8')));
}

// The options both python engines take (bc_common.arg_parser), all but the seed and outputs.
function engineArgs(binFile, o) {
  const args = ['--demos', binFile, '--obs-dim', String(OBS_DIM), '--hidden', o.hidden.join(','),
    '--epochs', String(o.epochs), '--batch', String(o.batch), '--lr', String(o.lr)];
  // o.distill: { logitsFile (float64 n x 3, writeLogits), mode, tau, alpha }; o.quant: { bits, scheme }.
  if (o.distill) {
    args.push('--teacher-logits', o.distill.logitsFile, '--distill-mode', o.distill.mode);
    if (o.distill.mode === 'soft') args.push('--tau', String(o.distill.tau), '--alpha', String(o.distill.alpha || 0));
  }
  if (o.quant) { checkQuantOptions(o.quant); args.push('--quant-bits', String(o.quant.bits), '--quant-scheme', o.quant.scheme); }
  for (const f of o.extraDemos || []) args.push('--extra-demos', f);
  if (o.transitionWeight !== undefined) args.push('--transition-weight', String(o.transitionWeight));
  return args;
}

// Spawn an engine; onLine gets each stdout line; resolves on exit 0.
function runEngine(python, args, env, o, onLine, name) {
  const { spawn } = require('child_process');
  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-W', 'ignore', ...args], { stdio: ['ignore', 'pipe', 'inherit'], env });
    if (o.onSpawn) o.onSpawn(child);
    let buf = '';
    child.stdout.on('data', (d) => {
      buf += d;
      let k;
      while ((k = buf.indexOf('\n')) >= 0) { onLine(buf.slice(0, k)); buf = buf.slice(k + 1); }
    });
    child.on('error', reject);
    child.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${name} exited with code ${code}`))));
  });
}

// ---------------------------------------------------------------------------
// MLX backend (trainer/bc_train_mlx.py; Apple GPU; research/plan-gpu-trainer.md)
// ---------------------------------------------------------------------------
// The interpreter with mlx: PYTHON_MLX, else <root>/.venv-mlx/bin/python (pyenv Python 3.14.8,
// mlx 0.32.3, numpy 2.5.4). A worktree has no .venv-mlx of its own, so set PYTHON_MLX there.
function configuredMlxPython(root) {
  if (process.env.PYTHON_MLX) return process.env.PYTHON_MLX;
  const venv = path.join(root, '.venv-mlx', 'bin', 'python');
  return fs.existsSync(venv) ? venv : null;
}
function mlxPythonPath(root) {
  const p = configuredMlxPython(root);
  if (p) return p;
  throw new Error(`--backend mlx needs ${path.join(root, '.venv-mlx', 'bin', 'python')} (research/plan-gpu-trainer.md Phase 0) or the PYTHON_MLX env var`);
}

// The --backend used when none is given (user's choices: bc-train-batch 2026-10-11, bc-train
// "same as batch" 2026-10-12): mlx when an mlx interpreter is configured and --jobs is not
// given; else python. With nodeFallback (bc-train; bc-train-batch has no node backend), python
// only when a numpy interpreter is configured, else node. "Configured" means the env var is set
// or the venv exists; neither is probed. Returns { backend, why } (why: for the log line).
function defaultBackend(root, { jobs, nodeFallback = false } = {}) {
  const mlx = !!configuredMlxPython(root);
  if (mlx && jobs === undefined) return { backend: 'mlx', why: 'mlx interpreter found' };
  const noMlx = mlx ? 'mlx interpreter found, but --jobs given' : 'no .venv-mlx or PYTHON_MLX';
  if (!nodeFallback) return { backend: 'python', why: noMlx };
  if (configuredPython(root)) return { backend: 'python', why: `${noMlx}; numpy interpreter found` };
  return { backend: 'node', why: `${noMlx}, no .venv or PYTHON` };
}

// Train several models of ONE shape (o.hidden) at once on the GPU, one per o.seeds[k], into
// outFiles[k]; o.snapshotDirs (optional) one per seed. Resolves to the results in seed order
// ({ layers, params, log[, scales|k, transitions], engine }). onLine(k, line) gets each model's
// epoch lines (k = index into o.seeds) and onLine(null, line) any other stdout line.
function trainMlx(binFile, outFiles, o, root, onLine = () => {}) {
  if (outFiles.length !== o.seeds.length) throw new Error('trainMlx: one output file per seed');
  const args = [path.join(__dirname, 'bc_train_mlx.py'), ...engineArgs(binFile, o), '--seeds', o.seeds.join(','), '--out-files', outFiles.join(',')];
  if (o.snapshotDirs) args.push('--snapshot-dirs', o.snapshotDirs.join(','));
  const route = (line) => {
    if (!line.startsWith('{')) return line && onLine(null, line);
    const { model, ...row } = JSON.parse(line);
    onLine(model, row);
  };
  return runEngine(mlxPythonPath(root), args, process.env, o, route, 'bc_train_mlx.py')
    .then(() => outFiles.map(f => JSON.parse(fs.readFileSync(f, 'utf8'))));
}

// Write n x 3 logits as little-endian float64 for bc_train.py --teacher-logits.
function writeLogits(file, Z) {
  const b = Buffer.alloc(Z.length * 8);
  for (let i = 0; i < Z.length; i++) b.writeDoubleLE(Z[i], i * 8);
  fs.writeFileSync(file, b);
}

// Identifies a model's weights: sha256 of JSON.stringify(params) (a plain array).
const paramsSha256 = (params) => crypto.createHash('sha256').update(JSON.stringify(Array.from(params))).digest('hex');

// ---------------------------------------------------------------------------
// best.json (Node owns the schema; same as ES runs and runs/v2-bc-jump-d2w32-s1)
// ---------------------------------------------------------------------------
function assembleBest({ layers, params, validation, coreHash, config }) {
  return {
    format: P.FORMAT, layers, actionRepeat: 1, gameVersion: config.gameVersion, generation: null,
    validation, coreHash, config, params,
  };
}

module.exports = {
  DEMOS_FORMAT, ROW, GEN_DEFAULTS, TRAIN_DEFAULTS, TEACHERS, resolveTeacher, actionLabel, isBoring, genEpisode, genSeed, studentFactory, transitionMask, generateDemos, resolveDemos, checkDemos,
  loadDemos, sha256File, layerOffsets, createTrainer, classWeights, trainNode, configuredPython, pythonPath, trainPython, configuredMlxPython, mlxPythonPath, defaultBackend, trainMlx, assembleBest,
  teacherLogits, argmaxLabels, softTargets, QUANT_SCHEMES, quantMax, checkQuantOptions, pow2Exponent, quantizeInto, quantizeParams, checkQuantized, quantRecord,
  writeLogits, paramsSha256,
};
