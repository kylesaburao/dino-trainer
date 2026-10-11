'use strict';
// Behaviour cloning pipeline (trainer/bc.js, trainer/bc_train.py, cli bc-gen/bc-train).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const Core = require('../core.js');
const P = require('../policy.js');
const { createSolver } = require('./solver.js');
const bc = require('../trainer/bc.js');
const { coreHash, ROOT } = require('../trainer/store.js');

const SMALL = { seeds: [1, 2], levels: [0], span0: 300, span: 5000, keepBoring: 10, gameVersion: 'v2' };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-bc-'));
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));

async function gen(name, workers) {
  const dir = path.join(tmp, name);
  const manifest = await bc.generateDemos(dir, { ...SMALL, workers, coreHash: coreHash() });
  fs.writeFileSync(path.join(dir, 'demos.json'), JSON.stringify(manifest));
  return { dir, manifest, rows: new Float32Array(fs.readFileSync(path.join(dir, 'demos.bin')).buffer.slice(0)) };
}

let small = null;
const smallDemos = async () => (small = small || await gen('w1', 1));

test('bc-gen rows are policy.observe features labelled by the solver, in replay order', async () => {
  const { manifest, rows } = await smallDemos();
  const R = bc.ROW, D = P.OBS_DIM;
  assert.strictEqual(R, D + 2);
  let r = 0, ep = 0;
  for (const seed of SMALL.seeds) {
    const start = r;
    // Replay the episode: every running frame's (observe, solver label); rows must be
    // an in-order subsequence, and every skipped frame must be a boring one.
    const g = Core.createGame({ rng: Core.createRng(seed), version: 'v2' });
    const sol = createSolver();
    const cap = SMALL.span0 / Core.CFG.scoreCoef;
    const x = new Float32Array(D);
    do {
      const act = sol(g);
      if (g.state === 'running') {
        const label = act.jump ? 1 : 0;
        P.observe(g, x);
        const row = rows.subarray(r * R, (r + 1) * R);
        const same = r < manifest.rows && row[D] === label && x.every((v, j) => Object.is(v, row[j]));
        if (same) {
          assert.ok(row[D + 1] === 1 || row[D + 1] === SMALL.keepBoring, `weight ${row[D + 1]}`);
          if (row[D + 1] !== 1) assert.ok(!x[2] && label === 0, 'a down-weighted row must be a grounded no-op');
          r++;
        } else {
          assert.ok(!x[2] && label === 0, `frame at distance ${g.distance} was dropped but is not boring`);
        }
      }
      g.step(act);
    } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap);
    assert.notStrictEqual(g.state, 'over', 'the solver must not crash');
    assert.strictEqual(r - start, manifest.episodes[ep++].rows);
  }
  assert.strictEqual(r, manifest.rows, 'every row is accounted for by the replay');
});

test('bc-gen manifest matches its file, and the output does not depend on worker count', async () => {
  const { dir, manifest, rows } = await smallDemos();
  const R = bc.ROW, D = P.OBS_DIM;
  assert.strictEqual(manifest.format, bc.DEMOS_FORMAT);
  assert.strictEqual(manifest.obsDim, D);
  assert.strictEqual(manifest.coreHash, coreHash());
  assert.strictEqual(manifest.solverCrashes, 0);
  assert.strictEqual(rows.length, manifest.rows * R);
  assert.strictEqual(bc.sha256File(path.join(dir, 'demos.bin')), manifest.sha256);
  const c = { rows: [0, 0, 0], weighted: [0, 0, 0] };
  for (let i = 0; i < manifest.rows; i++) {
    const y = rows[i * R + D], w = rows[i * R + D + 1];
    assert.ok(y === 0 || y === 1);
    assert.ok(w === 1 || w === SMALL.keepBoring);
    c.rows[y]++; c.weighted[y] += w;
  }
  assert.deepStrictEqual(manifest.counts, { rows: { none: c.rows[0], jump: c.rows[1], duck: 0 }, weighted: { none: c.weighted[0], jump: c.weighted[1], duck: 0 } });
  assert.ok(c.rows[1] > 0 && c.weighted[0] > c.rows[0], 'both classes present and some boring rows down-weighted');
  const two = await gen('w2', 2);
  assert.strictEqual(two.manifest.sha256, manifest.sha256);
});

test('node backend: gradient matches finite differences (44-4-3, Float64)', () => {
  const layers = [P.OBS_DIM, 4, 3];
  const rng = Core.createRng(7);
  const n = 12, D = P.OBS_DIM;
  const X = Float64Array.from({ length: n * D }, () => rng() * 2 - 1);
  const y = Uint8Array.from({ length: n }, (_, i) => i % 3);
  const w = Float64Array.from({ length: n }, () => 0.5 + rng());
  const idx = Uint32Array.from({ length: n }, (_, i) => i);
  const wsum = w.reduce((a, b) => a + b, 0);
  const params = Float64Array.from({ length: P.numParams(layers) }, () => rng() - 0.5);
  const { lossAndGrad } = bc.createTrainer(layers);
  const grad = new Float64Array(params.length), scratch = new Float64Array(params.length);
  lossAndGrad(params, X, D, y, w, idx, grad);
  const f = (p) => lossAndGrad(p, X, D, y, w, idx, scratch) / wsum;
  const h = 1e-6;
  let worst = 0;
  for (let k = 0; k < params.length; k++) {
    const p = Float64Array.from(params);
    p[k] += h; const up = f(p);
    p[k] -= 2 * h; const dn = f(p);
    const fd = (up - dn) / (2 * h);
    const rel = Math.abs(fd - grad[k]) / Math.max(1e-8, Math.abs(fd) + Math.abs(grad[k]));
    worst = Math.max(worst, rel);
  }
  assert.ok(worst < 1e-5, `worst relative error ${worst}`);
});

test('node backend: deterministic per seed, loads as a policy model, learns the small demos', async () => {
  const { dir } = await smallDemos();
  const data = bc.loadDemos(path.join(dir, 'demos.bin'));
  const o = { hidden: [16], epochs: 40, batch: 64, lr: 1e-2, seed: 3 };
  const a = bc.trainNode(data, o), b = bc.trainNode(data, o);
  assert.deepStrictEqual(a.params, b.params);
  assert.deepStrictEqual(a.layers, [P.OBS_DIM, 16, 3]);
  assert.strictEqual(a.log.length, o.epochs);
  const decide = P.createNet(a.layers, a.params);
  assert.ok([0, 1, 2].includes(decide(new Float64Array(P.OBS_DIM))));
  const agent = P.createAgent({ format: P.FORMAT, layers: a.layers, params: a.params, actionRepeat: 1, gameVersion: 'v2' });
  const g = Core.createGame({ rng: Core.createRng(1), version: 'v2' });
  for (let i = 0; i < 50; i++) g.step(agent.act(g));
  assert.strictEqual(g.state, 'running');
  const last = a.log[a.log.length - 1];
  assert.ok(last.recall.jump > 0.9, `jump recall ${last.recall.jump}`);
  const c = bc.trainNode(data, { ...o, seed: 4, epochs: 1 });
  assert.notDeepStrictEqual(c.params, bc.trainNode(data, { ...o, epochs: 1 }).params);
});

test('python backend: learns the small demos (skipped without numpy)', async (t) => {
  const py = bc.pythonPath(ROOT);
  const probe = spawnSync(py, ['-c', 'import numpy'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    t.skip(`python backend not tested: ${py} has no numpy (set PYTHON to an interpreter with numpy, e.g. the repo's .venv)`);
    return;
  }
  const { dir } = await smallDemos();
  const out = path.join(tmp, 'py.json');
  const o = { hidden: [16], epochs: 40, batch: 64, lr: 1e-2, seed: 3 };
  const r = await bc.trainPython(path.join(dir, 'demos.bin'), out, o, ROOT);
  assert.deepStrictEqual(r.layers, [P.OBS_DIM, 16, 3]);
  assert.strictEqual(r.params.length, P.numParams(r.layers));
  assert.strictEqual(r.log.length, o.epochs);
  P.createNet(r.layers, r.params);
  const last = r.log[r.log.length - 1];
  assert.ok(last.recall.jump > 0.9, `jump recall ${last.recall.jump}`);
});

test('python backend: --snapshots leaves params unchanged and writes one file per epoch (skipped without numpy)', async (t) => {
  const py = bc.pythonPath(ROOT);
  const probe = spawnSync(py, ['-c', 'import numpy'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    t.skip(`python backend not tested: ${py} has no numpy (set PYTHON to an interpreter with numpy, e.g. the repo's .venv)`);
    return;
  }
  const { dir } = await smallDemos();
  const bin = path.join(dir, 'demos.bin');
  const o = { hidden: [8, 8], epochs: 5, batch: 64, lr: 1e-2, seed: 2 };
  const plain = await bc.trainPython(bin, path.join(tmp, 'py-plain.json'), o, ROOT);
  const snapshotDir = path.join(tmp, 'snaps');
  const snap = await bc.trainPython(bin, path.join(tmp, 'py-snap.json'), { ...o, snapshotDir }, ROOT);
  assert.deepStrictEqual(snap.params, plain.params, '--snapshots must not change training');
  assert.deepStrictEqual(snap.log.map(r => r.loss), plain.log.map(r => r.loss));
  const files = fs.readdirSync(snapshotDir).sort();
  assert.deepStrictEqual(files, ['epoch-01.json', 'epoch-02.json', 'epoch-03.json', 'epoch-04.json', 'epoch-05.json']);
  const snaps = files.map(f => JSON.parse(fs.readFileSync(path.join(snapshotDir, f), 'utf8')));
  snaps.forEach((s, i) => {
    assert.strictEqual(s.epoch, i + 1);
    assert.deepStrictEqual(s.layers, plain.layers);
    assert.strictEqual(s.params.length, P.numParams(s.layers));
    P.createNet(s.layers, s.params);
  });
  assert.deepStrictEqual(snaps[snaps.length - 1].params, plain.params, 'the last snapshot is the final params');
  assert.notDeepStrictEqual(snaps[0].params, plain.params);
});

test('bc-train --snapshots is rejected with the node backend', () => {
  const run = spawnSync(process.execPath, [path.join(ROOT, 'trainer', 'cli.js'), 'bc-train', '--dir', path.join(tmp, 'snaprun'), '--hidden', '4', '--snapshots'], { encoding: 'utf8' });
  assert.notStrictEqual(run.status, 0);
  assert.match(run.stderr, /--snapshots is only supported with --backend python/);
  assert.ok(!fs.existsSync(path.join(tmp, 'snaprun')));
});

test('bc-train refuses demos from a different core hash unless --force', async () => {
  const { dir, manifest } = await smallDemos();
  const bad = path.join(tmp, 'badhash');
  fs.mkdirSync(bad, { recursive: true });
  fs.copyFileSync(path.join(dir, 'demos.bin'), path.join(bad, 'demos.bin'));
  fs.writeFileSync(path.join(bad, 'demos.json'), JSON.stringify({ ...manifest, coreHash: '0'.repeat(64) }));
  const run = spawnSync(process.execPath, [path.join(ROOT, 'trainer', 'cli.js'), 'bc-train', '--dir', path.join(tmp, 'badrun'), '--demos', bad, '--hidden', '4'], { encoding: 'utf8' });
  assert.notStrictEqual(run.status, 0);
  assert.match(run.stderr, /core hash mismatch/);
  assert.ok(!fs.existsSync(path.join(tmp, 'badrun', 'best.json')));
  const demos = bc.resolveDemos(bad);
  const warnings = [];
  assert.strictEqual(bc.checkDemos(demos, coreHash(), true, (m) => warnings.push(m)), manifest.sha256);
  assert.strictEqual(warnings.length, 1);
});

// ---------------------------------------------------------------------------
// Teacher selection (bc-gen --teacher)
// ---------------------------------------------------------------------------
// SMALL demos sha256 with the jump teacher, recorded on 700e8fb (before teacher
// selection existed): the default must stay byte-identical.
const SMALL_JUMP_SHA256 = 'cf4962aafb79e614013399ffc68ee21181190f0ee92c87953458cb0256463890';

// A stand-in duck teacher, independent of createDuckSolver: the jump solver, but it
// holds the down key on every 5th grounded no-op running frame.
function createStubDuck() {
  const sol = createSolver();
  let k = 0;
  return (g) => {
    const act = sol(g);
    if (g.state === 'running' && !g.dino.jumping && !act.jump && k++ % 5 === 0) return { jump: false, down: true };
    return act;
  };
}

let duckSmall = null;
async function duckDemos() {
  if (duckSmall) return duckSmall;
  const dir = path.join(tmp, 'duck');
  const manifest = await bc.generateDemos(dir, { ...SMALL, workers: 1, coreHash: coreHash(), teacher: 'duck', createTeacher: createStubDuck });
  fs.writeFileSync(path.join(dir, 'demos.json'), JSON.stringify(manifest));
  duckSmall = { dir, manifest, rows: new Float32Array(fs.readFileSync(path.join(dir, 'demos.bin')).buffer.slice(0)) };
  return duckSmall;
}

const cli = (...args) => spawnSync(process.execPath, [path.join(ROOT, 'trainer', 'cli.js'), ...args], { encoding: 'utf8' });

test('teacher table: jump is createSolver, duck is createDuckSolver or a clear error, unknown names throw', () => {
  assert.deepStrictEqual(Object.keys(bc.TEACHERS), ['jump', 'duck']);
  assert.strictEqual(bc.GEN_DEFAULTS.teacher, 'jump');
  assert.strictEqual(bc.resolveTeacher('jump'), createSolver);
  const solver = require('./solver.js');
  if (typeof solver.createDuckSolver === 'function') assert.strictEqual(bc.resolveTeacher('duck'), solver.createDuckSolver);
  else assert.throws(() => bc.resolveTeacher('duck'), /createDuckSolver not available on this tree/);
  assert.throws(() => bc.resolveTeacher('hop'), /unknown teacher hop/);
  const run = cli('bc-gen', '--dir', path.join(tmp, 'badteacher'), '--teacher', 'hop');
  assert.notStrictEqual(run.status, 0);
  assert.match(run.stderr, /--teacher must be one of jump, duck/);
  assert.ok(!fs.existsSync(path.join(tmp, 'badteacher')));
});

test('default (jump) demos are byte-identical to before teacher selection, and record teacherName jump', async () => {
  const { dir, manifest } = await smallDemos();
  assert.strictEqual(manifest.sha256, SMALL_JUMP_SHA256);
  assert.strictEqual(manifest.teacher, 'test/solver.js');
  assert.strictEqual(manifest.teacherName, 'jump');
  const explicit = await bc.generateDemos(path.join(tmp, 'jump-explicit'), { ...SMALL, workers: 1, coreHash: coreHash(), teacher: 'jump' });
  assert.strictEqual(explicit.sha256, SMALL_JUMP_SHA256);
  const r = bc.trainNode(bc.loadDemos(path.join(dir, 'demos.bin')), { hidden: [4], epochs: 1, batch: 64, lr: 1e-2, seed: 1 });
  assert.deepStrictEqual(Object.keys(r.log[0].recall), ['none', 'jump'], 'no duck key without duck rows');
});

test('a duck teacher gives label-2 rows (never boring), counted in the manifest, with recall.duck', async () => {
  const { dir, manifest, rows } = await duckDemos();
  const R = bc.ROW, D = P.OBS_DIM;
  assert.strictEqual(manifest.teacherName, 'duck');
  let duck = 0;
  for (let i = 0; i < manifest.rows; i++) {
    if (rows[i * R + D] !== 2) continue;
    duck++;
    assert.strictEqual(rows[i * R + D + 1], 1, 'a duck row is never down-weighted as boring');
  }
  assert.ok(duck > 0);
  assert.strictEqual(manifest.counts.rows.duck, duck);
  assert.strictEqual(manifest.counts.weighted.duck, duck);
  const far = new Float32Array(D);   // grounded, no visible obstacle: boring for label 0 only
  assert.strictEqual(bc.isBoring(far, 0), true);
  assert.strictEqual(bc.isBoring(far, 2), false);
  const data = bc.loadDemos(path.join(dir, 'demos.bin'));
  assert.ok(bc.classWeights(data.y, data.sw).every(w => w > 0), 'all three classes weighted');
  const r = bc.trainNode(data, { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1 });
  assert.deepStrictEqual(Object.keys(r.log[1].recall), ['none', 'jump', 'duck']);
  assert.ok(r.log[1].recall.duck >= 0 && r.log[1].recall.duck <= 1);
  await assert.rejects(bc.generateDemos(path.join(tmp, 'duck-w2'), { ...SMALL, workers: 2, coreHash: coreHash(), teacher: 'duck', createTeacher: createStubDuck }),
    /createTeacher cannot cross worker threads/);
});

test('a teacher that holds both jump and down throws', async () => {
  assert.throws(() => bc.actionLabel({ jump: true, down: true }), /both jump and down/);
  assert.deepStrictEqual([{ jump: false, down: false }, { jump: true, down: false }, { jump: false, down: true }].map(bc.actionLabel), [0, 1, 2]);
  const both = () => { const sol = createSolver(); let f = 0; return (g) => (++f > 20 ? { jump: true, down: true } : sol(g)); };
  await assert.rejects(bc.generateDemos(path.join(tmp, 'both'), { ...SMALL, workers: 1, coreHash: coreHash(), teacher: 'duck', createTeacher: both }),
    /both jump and down/);
});

test('bc-train on duck demos: config.teacherName duck, recall.duck in bc-metrics, validation runs; old manifests read as jump', async () => {
  const { dir } = await duckDemos();
  const out = path.join(tmp, 'duckrun');
  const run = cli('bc-train', '--dir', out, '--demos', dir, '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1');
  assert.strictEqual(run.status, 0, run.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  assert.strictEqual(best.config.teacherName, 'duck');
  assert.ok('duck' in best.config.final.recall);
  assert.strictEqual(typeof best.validation.median, 'number');
  const metrics = fs.readFileSync(path.join(out, 'bc-metrics.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
  assert.ok('duck' in metrics[0].recall);
  // A manifest from before teacher selection (no teacherName) trains as jump.
  const { dir: jdir, manifest } = await smallDemos();
  const old = path.join(tmp, 'oldmanifest');
  fs.mkdirSync(old, { recursive: true });
  fs.copyFileSync(path.join(jdir, 'demos.bin'), path.join(old, 'demos.bin'));
  const { teacherName, ...legacy } = manifest;
  assert.strictEqual(teacherName, 'jump');
  fs.writeFileSync(path.join(old, 'demos.json'), JSON.stringify(legacy));
  const out2 = path.join(tmp, 'oldrun');
  const run2 = cli('bc-train', '--dir', out2, '--demos', old, '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1');
  assert.strictEqual(run2.status, 0, run2.stderr);
  const best2 = JSON.parse(fs.readFileSync(path.join(out2, 'best.json'), 'utf8'));
  assert.strictEqual(best2.config.teacherName, 'jump');
  assert.deepStrictEqual(Object.keys(best2.config.final.recall), ['none', 'jump']);
});

test('python backend: recall.duck on duck demos, absent on jump demos (skipped without numpy)', async (t) => {
  const py = bc.pythonPath(ROOT);
  const probe = spawnSync(py, ['-c', 'import numpy'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    t.skip(`python backend not tested: ${py} has no numpy (set PYTHON to an interpreter with numpy, e.g. the repo's .venv)`);
    return;
  }
  const o = { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1 };
  const d = await bc.trainPython(path.join((await duckDemos()).dir, 'demos.bin'), path.join(tmp, 'py-duck.json'), o, ROOT);
  assert.deepStrictEqual(Object.keys(d.log[1].recall), ['none', 'jump', 'duck']);
  const j = await bc.trainPython(path.join((await smallDemos()).dir, 'demos.bin'), path.join(tmp, 'py-jump.json'), o, ROOT);
  assert.deepStrictEqual(Object.keys(j.log[1].recall), ['none', 'jump']);
});

// ---------------------------------------------------------------------------
// Distillation (bc-train --distill-from) and quantization (--quant-bits, bc-quantize)
// ---------------------------------------------------------------------------
// sha256 of JSON.stringify(params) for 1-epoch runs ({hidden [4], epochs 1, batch 64,
// lr 1e-2, seed 1}) on the SMALL jump demos and the stub-duck demos, recorded on main
// b057ec4 (before distillation and quantization existed). With the new flags off, both
// backends must reproduce them. The python values depend on numpy/BLAS: recorded on the
// Mac (darwin arm64, numpy 2.0.2), checked only there.
const PIN_OPTS = { hidden: [4], epochs: 1, batch: 64, lr: 1e-2, seed: 1 };
const PINS = {
  jump: { node: '27e195823b0d766363eb0e0cda76f037922d9a47a8eba9464cc1d05c71f1b90c', python: 'a21360e68e93af0a23b7b0d47f05b6f4c0c9b7ef993730241bae40165ef9ae67' },
  duck: { node: '5674e0f0aecefb95c7e24ea4b986a1f4c8779b46e139280fd5061872e6d378a5', python: 'e4ed1f4e674eda8d08499f9cf32473f75b1f1373a0a6db1a4424a8e54ec9af83' },
};
const STUB_DUCK_SHA256 = 'a468d45a7bc23c3ea78d5ac947b722803ffb9ba42075d9f5e619b8903335e8c1';
const hasNumpy = () => {
  const probe = spawnSync(bc.pythonPath(ROOT), ['-c', 'import numpy'], { encoding: 'utf8' });
  return !probe.error && probe.status === 0;
};
const PY_PINNED = process.platform === 'darwin' && process.arch === 'arm64';
const smallDir = async (which) => (which === 'duck' ? await duckDemos() : await smallDemos()).dir;
const loadSmall = async (which) => bc.loadDemos(path.join(await smallDir(which), 'demos.bin'));
// A one-hot "teacher": logit 1000 on the solver's label, 0 elsewhere (exp(-1000) is 0 in float64).
const oneHotLogits = (y) => { const Z = new Float64Array(y.length * 3); y.forEach((c, i) => { Z[i * 3 + c] = 1000; }); return Z; };

test('flags off: 1-epoch params on both backends are identical to main b057ec4 (jump and duck demos)', async (t) => {
  assert.strictEqual((await duckDemos()).manifest.sha256, STUB_DUCK_SHA256);
  for (const which of ['jump', 'duck']) {
    const r = bc.trainNode(await loadSmall(which), PIN_OPTS);
    assert.strictEqual(bc.paramsSha256(r.params), PINS[which].node, `node ${which}`);
    assert.deepStrictEqual(Object.keys(r.log[0]), ['epoch', 'loss', 'recall', 'seconds'], 'bc-metrics keys unchanged');
  }
  if (!hasNumpy()) return t.skip('python pins not checked: no numpy');
  if (!PY_PINNED) return t.skip(`python pins recorded on darwin arm64, not ${process.platform} ${process.arch}`);
  for (const which of ['jump', 'duck']) {
    const r = await bc.trainPython(path.join(await smallDir(which), 'demos.bin'), path.join(tmp, `pin-${which}.json`), PIN_OPTS, ROOT);
    assert.strictEqual(bc.paramsSha256(r.params), PINS[which].python, `python ${which}`);
    assert.deepStrictEqual(Object.keys(r.log[0]), ['epoch', 'loss', 'recall', 'seconds']);
    assert.ok(!('scales' in r));
  }
});

test('teacherLogits: argmax is policy.createNet decision on every row; Float32 rows (stride) give the same logits', async () => {
  const data = await loadSmall('duck');
  const teacher = bc.trainNode(data, { hidden: [8, 8], epochs: 3, batch: 64, lr: 1e-2, seed: 5 });
  const Z = bc.teacherLogits(teacher, data);
  const ya = bc.argmaxLabels(Z);
  const decide = P.createNet(teacher.layers, teacher.params);
  let agreeSolver = 0;
  for (let i = 0; i < data.n; i++) {
    assert.strictEqual(ya[i], decide(data.X.subarray(i * data.dim, (i + 1) * data.dim)), `row ${i}`);
    if (ya[i] === data.y[i]) agreeSolver++;
  }
  assert.ok(agreeSolver > 0.5 * data.n, 'a trained teacher mostly agrees with the solver (stub-duck labels are partly arbitrary)');
  const raw = new Float32Array(fs.readFileSync(path.join(await smallDir('duck'), 'demos.bin')).buffer.slice(0));
  const Z32 = bc.teacherLogits(teacher, { X: raw, n: data.n, dim: P.OBS_DIM, stride: bc.ROW });
  assert.deepStrictEqual(Array.from(Z32), Array.from(Z));
  assert.throws(() => bc.teacherLogits({ layers: [5, 3], params: new Array(18).fill(0) }, data), /teacher expects 5 inputs/);
});

test('soft targets: tau -> 0 gives the one-hot argmax target (the hard-mode target); large tau tends to uniform', () => {
  const rng = Core.createRng(11);
  const n = 200, Z = new Float64Array(n * 3);
  for (let r = 0; r < n; r++) {
    const top = r % 3, base = rng() * 4 - 2;
    for (let c = 0; c < 3; c++) Z[r * 3 + c] = c === top ? base + 0.1 + rng() : base - rng();   // margins >= 0.1
  }
  const y = bc.argmaxLabels(Z);
  const Q = bc.softTargets(Z, 1e-4);
  for (let r = 0; r < n; r++) for (let c = 0; c < 3; c++) assert.ok(Math.abs(Q[r * 3 + c] - (c === y[r] ? 1 : 0)) < 1e-9);
  const U = bc.softTargets(Z, 1e6);
  for (const q of U) assert.ok(Math.abs(q - 1 / 3) < 1e-5);
  assert.deepStrictEqual(Array.from(bc.argmaxLabels(Float64Array.from([1, 1, 0, 0, 2, 2]))), [0, 1], 'first index wins ties, as policy.js');
});

test('node backend: distillation gradient matches finite differences (soft, tau 2, alpha 0.3; hard)', () => {
  const layers = [P.OBS_DIM, 4, 3];
  const rng = Core.createRng(9);
  const n = 12, D = P.OBS_DIM;
  const X = Float64Array.from({ length: n * D }, () => rng() * 2 - 1);
  const y = Uint8Array.from({ length: n }, (_, i) => i % 3);
  const w = Float64Array.from({ length: n }, () => 0.5 + rng());
  const idx = Uint32Array.from({ length: n }, (_, i) => i);
  const wsum = w.reduce((a, b) => a + b, 0);
  const params = Float64Array.from({ length: P.numParams(layers) }, () => rng() - 0.5);
  const Zt = Float64Array.from({ length: n * 3 }, () => rng() * 6 - 3);
  const { lossAndGrad } = bc.createTrainer(layers);
  const modes = [() => ({ ySolver: y, Q: bc.softTargets(Zt, 2), tau: 2, alpha: 0.3, solverLoss: 0 }), () => ({ ySolver: y, Q: null, solverLoss: 0 })];
  for (const mk of modes) {
    const yT = mk().Q ? y : bc.argmaxLabels(Zt);
    const grad = new Float64Array(params.length), scratch = new Float64Array(params.length);
    lossAndGrad(params, X, D, yT, w, idx, grad, mk());
    const f = (p) => lossAndGrad(p, X, D, yT, w, idx, scratch, mk()) / wsum;
    let worst = 0;
    for (let k = 0; k < params.length; k++) {
      const p = Float64Array.from(params);
      p[k] += 1e-6; const up = f(p);
      p[k] -= 2e-6; const dn = f(p);
      const fd = (up - dn) / 2e-6;
      if (Math.abs(fd - grad[k]) > 1e-9) worst = Math.max(worst, Math.abs(fd - grad[k]) / Math.max(1e-8, Math.abs(fd) + Math.abs(grad[k])));   // FD noise ~1e-10
    }
    assert.ok(worst < 1e-5, `worst relative error ${worst}`);
  }
  // The solver-label CE is reported alongside: equal to the plain loss on the same rows.
  const g = new Float64Array(params.length);
  const plain = lossAndGrad(params, X, D, y, w, idx, g);
  const dist = { ySolver: y, Q: bc.softTargets(Zt, 2), tau: 2, alpha: 0, solverLoss: 0 };
  lossAndGrad(params, X, D, y, w, idx, g, dist);
  assert.ok(Math.abs(dist.solverLoss - plain) < 1e-12 * Math.abs(plain));
});

test('a one-hot teacher at tau 1 reproduces hard-label training bit for bit (soft and hard modes, node and python)', async (t) => {
  for (const which of ['jump', 'duck']) {
    const data = await loadSmall(which);
    const logits = oneHotLogits(data.y);
    const plainLoss = bc.trainNode(data, PIN_OPTS).log[0].loss;
    for (const distill of [{ logits, mode: 'soft', tau: 1, alpha: 0 }, { logits, mode: 'hard' }]) {
      const r = bc.trainNode(data, { ...PIN_OPTS, distill });
      assert.strictEqual(bc.paramsSha256(r.params), PINS[which].node, `node ${which} ${distill.mode}`);
      assert.deepStrictEqual(Object.keys(r.log[0]), ['epoch', 'loss', 'recall', 'seconds', 'distillLoss']);
      assert.ok(Math.abs(r.log[0].loss - plainLoss) < 1e-12, 'loss is the CE against the solver labels');
    }
  }
  if (!hasNumpy()) return t.skip('python not checked: no numpy');
  for (const which of ['jump', 'duck']) {
    const bin = path.join(await smallDir(which), 'demos.bin');
    const plain = await bc.trainPython(bin, path.join(tmp, `oh-plain-${which}.json`), PIN_OPTS, ROOT);
    const logitsFile = path.join(tmp, `oh-${which}.f64`);
    bc.writeLogits(logitsFile, oneHotLogits(bc.loadDemos(bin).y));
    for (const distill of [{ logitsFile, mode: 'soft', tau: 1, alpha: 0 }, { logitsFile, mode: 'hard' }]) {
      const r = await bc.trainPython(bin, path.join(tmp, `oh-${which}-${distill.mode}.json`), { ...PIN_OPTS, distill }, ROOT);
      assert.deepStrictEqual(r.params, plain.params, `python ${which} ${distill.mode}`);
      assert.ok('distillLoss' in r.log[0]);
      assert.strictEqual(r.log[0].loss, plain.log[0].loss);
    }
  }
});

test('quantization: q * scale exactly, levels within qmax, ternary at 2 bits; QAT on both backends gives a grid that loads', async (t) => {
  const data = await loadSmall('duck');
  const fl = bc.trainNode(data, { hidden: [8, 8], epochs: 2, batch: 64, lr: 1e-2, seed: 2 });
  for (const bits of [8, 4, 2]) for (const scheme of ['row', 'layer']) {
    const q = bc.quantizeParams(fl.layers, fl.params, { bits, scheme });
    const levels = bc.checkQuantized(fl.layers, q.params, { bits, scheme, scales: q.scales });
    assert.ok(levels <= 2 * bc.quantMax(bits) + 1);
    if (bits === 2) assert.ok(levels <= 3, 'ternary');
    assert.deepStrictEqual(q.scales.map(s => s.length), scheme === 'row' ? fl.layers.slice(1) : [1, 1, 1]);
    bc.layerOffsets(fl.layers).forEach(({ nout, b }) => { for (let o = 0; o < nout; o++) assert.strictEqual(q.params[b + o], fl.params[b + o], 'biases stay float'); });
    assert.deepStrictEqual(JSON.parse(JSON.stringify(q.params)), q.params, 'JSON round-trips the dequantized doubles');
  }
  const q4 = bc.quantizeParams(fl.layers, fl.params, { bits: 4, scheme: 'row' });
  const bad = q4.params.slice(); bad[0] += 1e-9;
  assert.throws(() => bc.checkQuantized(fl.layers, bad, { bits: 4, scheme: 'row', scales: q4.scales }), /not an integer multiple/);
  assert.throws(() => bc.checkQuantOptions({ bits: 1, scheme: 'row' }), /2\.\.8/);
  assert.throws(() => bc.checkQuantOptions({ bits: 4, scheme: 'channel' }), /row, layer/);
  const quant = { bits: 3, scheme: 'row' };
  const qn = bc.trainNode(data, { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1, quant });
  assert.ok(bc.checkQuantized(qn.layers, qn.params, { ...quant, scales: qn.scales }) <= 7);
  assert.deepStrictEqual(Object.keys(qn.log[0]), ['epoch', 'loss', 'recall', 'seconds']);
  P.createNet(qn.layers, qn.params);
  if (!hasNumpy()) return t.skip('python QAT not checked: no numpy');
  const bin = path.join(await smallDir('duck'), 'demos.bin');
  const snapshotDir = path.join(tmp, 'qat-snaps');
  const qp = await bc.trainPython(bin, path.join(tmp, 'qat.json'), { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1, quant, snapshotDir }, ROOT);
  assert.ok(bc.checkQuantized(qp.layers, qp.params, { ...quant, scales: qp.scales }) <= 7);
  const last = JSON.parse(fs.readFileSync(path.join(snapshotDir, 'epoch-02.json'), 'utf8'));
  assert.deepStrictEqual(last.params, qp.params, 'QAT snapshots hold the quantized params');
  P.createNet(qp.layers, qp.params);
});

// A small duck teacher trained through the CLI, shared by the next three tests.
let kdTeacher = null;
async function kdTeacherFile() {
  if (kdTeacher) return kdTeacher;
  const teacherDir = path.join(tmp, 'kd-teacher');
  const run = cli('bc-train', '--dir', teacherDir, '--demos', await smallDir('duck'), '--hidden', '8,8', '--epochs', '2', '--batch', '64', '--target', '50', '--workers', '1');
  assert.strictEqual(run.status, 0, run.stderr);
  return (kdTeacher = path.join(teacherDir, 'best.json'));
}

test('bc-train --distill-from / --quant-bits: config records the teacher (path, sha256), tau, mode, alpha, quant; bad flags refused', async () => {
  const dir = await smallDir('duck');
  const tf = await kdTeacherFile();
  const teacher = JSON.parse(fs.readFileSync(tf, 'utf8'));
  const out = path.join(tmp, 'kd-student');
  let run = cli('bc-train', '--dir', out, '--demos', dir, '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1',
    '--distill-from', tf, '--tau', '2', '--alpha', '0.25', '--quant-bits', '4');
  assert.strictEqual(run.status, 0, run.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  const c = best.config;
  assert.strictEqual(c.distillFrom.path, path.relative(ROOT, tf));
  assert.strictEqual(c.distillFrom.paramsSha256, bc.paramsSha256(teacher.params));
  assert.deepStrictEqual(c.distillFrom.layers, [P.OBS_DIM, 8, 8, 3]);
  assert.strictEqual(c.distillFrom.teacherName, 'duck');
  assert.strictEqual(c.distillMode, 'soft');
  assert.strictEqual(c.tau, 2);
  assert.strictEqual(c.alpha, 0.25);
  assert.strictEqual(c.teacherName, 'duck');
  assert.deepStrictEqual([c.quant.bits, c.quant.scheme, c.quant.mode, c.quant.biases], [4, 'row', 'qat', 'float']);
  assert.strictEqual(bc.checkQuantized(best.layers, best.params, c.quant), c.quant.levels);
  const rows = fs.readFileSync(path.join(out, 'bc-metrics.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
  assert.deepStrictEqual(Object.keys(rows[0]), ['backend', 'epoch', 'loss', 'recall', 'seconds', 'distillLoss']);
  assert.ok(fs.readdirSync(out).every(f => !f.endsWith('.tmp')), 'no temp files left');
  // hard mode records no tau/alpha; flags without --distill-from, or misused, are refused.
  const hard = path.join(tmp, 'kd-hard');
  run = cli('bc-train', '--dir', hard, '--demos', dir, '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1',
    '--distill-from', tf, '--distill-mode', 'hard');
  assert.strictEqual(run.status, 0, run.stderr);
  const hc = JSON.parse(fs.readFileSync(path.join(hard, 'best.json'), 'utf8')).config;
  assert.strictEqual(hc.distillMode, 'hard');
  assert.ok(!('tau' in hc) && !('alpha' in hc) && !('quant' in hc));
  const bads = [[['--tau', '2'], /need --distill-from/], [['--distill-from', tf, '--distill-mode', 'hard', '--tau', '2'], /soft only/],
    [['--distill-from', tf, '--alpha', '2'], /--alpha must be in/], [['--quant-bits', '9'], /2\.\.8/], [['--quant-scheme', 'row'], /needs --quant-bits/],
    [['--distill-from', path.join(tmp, 'nope.json')], /no model JSON/]];
  for (const [args, re] of bads) {
    const r = cli('bc-train', '--dir', path.join(tmp, 'kd-bad'), '--demos', dir, '--hidden', '4', ...args);
    assert.notStrictEqual(r.status, 0, args.join(' '));
    assert.match(r.stderr, re);
  }
  assert.ok(!fs.existsSync(path.join(tmp, 'kd-bad')));
  // Without the new flags, the config gains none of the new keys.
  for (const k of ['distillFrom', 'distillMode', 'tau', 'alpha', 'quant']) assert.ok(!(k in teacher.config), k);
});

test('python backend via bc-train: distilled QAT run writes distillLoss and the config (skipped without numpy)', async (t) => {
  if (!hasNumpy()) return t.skip('no numpy');
  const tf = await kdTeacherFile();
  const out = path.join(tmp, 'kd-py');
  const run = cli('bc-train', '--dir', out, '--demos', await smallDir('duck'), '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1',
    '--backend', 'python', '--distill-from', tf, '--tau', '4', '--quant-bits', '2', '--quant-scheme', 'layer');
  assert.strictEqual(run.status, 0, run.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  assert.strictEqual(best.config.tau, 4);
  assert.ok(bc.checkQuantized(best.layers, best.params, best.config.quant) <= 3);
  const rows = fs.readFileSync(path.join(out, 'bc-metrics.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
  assert.ok(Number.isFinite(rows[0].distillLoss) && Number.isFinite(rows[0].loss));
  assert.ok(fs.readdirSync(out).every(f => !f.endsWith('.tmp')));
});

test('bc-quantize: post-training quantization into a new run dir, validated, with the source recorded', async () => {
  const src = await kdTeacherFile();
  const srcBest = JSON.parse(fs.readFileSync(src, 'utf8'));
  const out = path.join(tmp, 'ptq-int4');
  const run = cli('bc-quantize', '--from', src, '--dir', out, '--quant-bits', '4', '--target', '50', '--workers', '1');
  assert.strictEqual(run.status, 0, run.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  assert.strictEqual(best.format, P.FORMAT);
  assert.deepStrictEqual(best.layers, srcBest.layers);
  assert.strictEqual(best.config.quant.mode, 'ptq');
  assert.strictEqual(best.config.quantFrom.paramsSha256, bc.paramsSha256(srcBest.params));
  assert.strictEqual(best.config.teacherName, 'duck');
  assert.match(best.config.method, /behaviou?r[ -]clon/, 'the dashboard classes it as a BC run');
  assert.deepStrictEqual(best.params, bc.quantizeParams(srcBest.layers, srcBest.params, { bits: 4, scheme: 'row' }).params);
  assert.strictEqual(typeof best.validation.median, 'number');
  const again = cli('bc-quantize', '--from', src, '--dir', out, '--quant-bits', '4', '--target', '50', '--workers', '1');
  assert.notStrictEqual(again.status, 0);
  assert.match(again.stderr, /exists; pass --overwrite/);
  const none = cli('bc-quantize', '--from', src, '--dir', path.join(tmp, 'ptq-none'));
  assert.match(none.stderr, /needs --quant-bits/);
  assert.strictEqual(best.coreHash, srcBest.coreHash, 'the source model core hash is carried over');
  const stale = path.join(tmp, 'stale-src.json');
  fs.writeFileSync(stale, JSON.stringify({ ...srcBest, coreHash: '0'.repeat(64) }));
  const refused = cli('bc-quantize', '--from', stale, '--dir', path.join(tmp, 'ptq-stale'), '--quant-bits', '8');
  assert.notStrictEqual(refused.status, 0);
  assert.match(refused.stderr, /core hash mismatch/);
});

// ---------------------------------------------------------------------------
// --quant-scheme pow2: ALL params N-bit fixed point, one 2^-k per layer
// ---------------------------------------------------------------------------
const pack = require('../research/scripts/prereg-distill/pack.js');
// Independent oracle: every weight AND bias of layer l is an integer in [-qmax, qmax] times
// 2^-k[l], and k[l] is the largest such exponent for the layer's float max |value|.
function assertPow2Grid(layers, params, k, bits, floatParams) {
  const qmax = 2 ** (bits - 1) - 1;
  bc.layerOffsets(layers).forEach(({ nin, nout, w }, l) => {
    const n = nin * nout + nout;
    for (let i = w; i < w + n; i++) {
      const s = params[i] * 2 ** k[l];
      assert.ok(Number.isInteger(s) && Math.abs(s) <= qmax, `layer ${l} param ${i - w}: ${params[i]} * 2^${k[l]} = ${s}`);
    }
    if (floatParams) {
      let m = 0;
      for (let i = w; i < w + n; i++) m = Math.max(m, Math.abs(floatParams[i]));
      assert.ok(m * 2 ** k[l] <= qmax && m * 2 ** (k[l] + 1) > qmax, `layer ${l}: k ${k[l]} is the largest exponent for max ${m}`);
    }
  });
}

test('pow2 PTQ: every weight and bias is an N-bit integer times 2^-k (N = 8, 4); k largest; idempotent; off-grid params refused', async () => {
  const data = await loadSmall('duck');
  const fl = bc.trainNode(data, { hidden: [8, 8], epochs: 2, batch: 64, lr: 1e-2, seed: 2 });
  for (const bits of [8, 4]) {
    const quant = { bits, scheme: 'pow2' };
    const q = bc.quantizeParams(fl.layers, fl.params, quant);
    assert.ok(!('scales' in q) && q.k.length === 3 && q.k.every(Number.isInteger));
    assertPow2Grid(fl.layers, q.params, q.k, bits, fl.params);
    const rec = bc.quantRecord(fl.layers, q.params, quant, 'ptq', q);
    assert.deepStrictEqual(Object.keys(rec), ['bits', 'scheme', 'mode', 'biases', 'k', 'levels', 'checked']);
    assert.deepStrictEqual([rec.biases, rec.checked], ['int', true]);
    assert.ok(rec.levels <= 2 * bc.quantMax(bits) + 1);
    const biasChanged = bc.layerOffsets(fl.layers).some(({ nout, b }) => { for (let o = 0; o < nout; o++) if (q.params[b + o] !== fl.params[b + o]) return true; return false; });
    assert.ok(biasChanged, 'biases are quantized too');
    assert.deepStrictEqual(bc.quantizeParams(fl.layers, q.params, quant), q, 'quantizing a pow2 grid is the identity (same k)');
    assert.deepStrictEqual(JSON.parse(JSON.stringify(q.params)), q.params, 'JSON round-trips the values');
    P.createNet(fl.layers, q.params);
    const bad = q.params.slice(); bad[bad.length - 1] += 2 ** -(q.k[2] + 1);   // half a step on a bias
    assert.throws(() => bc.checkQuantized(fl.layers, bad, rec), /not an integer in/);
    assert.throws(() => bc.checkQuantized(fl.layers, q.params, { ...rec, k: q.k.map(e => e + 1).slice(0, 2) }), /k must be 3 integers/);
  }
  assert.strictEqual(bc.pow2Exponent(0, 127), 0);
  assert.strictEqual(bc.pow2Exponent(1, 127), 6);       // 64 <= 127 < 128
  assert.strictEqual(bc.pow2Exponent(7, 7), 0);
  assert.strictEqual(bc.pow2Exponent(20, 7), -2);       // 20 / 4 = 5 <= 7 < 10
});

test('pow2 QAT trains on the stub duck demos on both backends; each output is an exact pow2 grid that the other backend\'s rule leaves unchanged', async (t) => {
  const data = await loadSmall('duck');
  const o = { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1 };
  for (const bits of [8, 4]) {
    const quant = { bits, scheme: 'pow2' };
    const qn = bc.trainNode(data, { ...o, quant });
    assert.ok(!('scales' in qn));
    assertPow2Grid(qn.layers, qn.params, qn.k, bits);
    assert.ok(qn.log.every(r => Number.isFinite(r.loss)));
    P.createNet(qn.layers, qn.params);
    if (!hasNumpy()) { t.skip('python pow2 QAT not checked: no numpy'); continue; }
    const snapshotDir = path.join(tmp, `pow2-snaps-${bits}`);
    const qp = await bc.trainPython(path.join(await smallDir('duck'), 'demos.bin'), path.join(tmp, `pow2-qat-${bits}.json`), { ...o, quant, snapshotDir }, ROOT);
    assert.ok(!('scales' in qp) && qp.k.length === 2);
    assertPow2Grid(qp.layers, qp.params, qp.k, bits);
    assert.deepStrictEqual(JSON.parse(fs.readFileSync(path.join(snapshotDir, 'epoch-02.json'), 'utf8')).params, qp.params, 'snapshots hold the pow2 params');
    // The backends do not match each other bit for bit (different RNGs and float32 vs
    // float64 training); what they share is the grid rule: node's quantizer maps the
    // python output to itself with python's k.
    assert.deepStrictEqual(bc.quantizeParams(qp.layers, qp.params, quant), { params: qp.params, k: qp.k });
    assert.ok(qp.log.every(r => Number.isFinite(r.loss)));
  }
});

test('pow2 through the CLI (bc-quantize and bc-train): config.quant records k, biases int, checked; pack.js round trip and storage bits', async () => {
  const src = await kdTeacherFile();
  const srcBest = JSON.parse(fs.readFileSync(src, 'utf8'));
  const out = path.join(tmp, 'ptq-pow2-int4');
  let run = cli('bc-quantize', '--from', src, '--dir', out, '--quant-bits', '4', '--quant-scheme', 'pow2', '--target', '50', '--workers', '1');
  assert.strictEqual(run.status, 0, run.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  const q = best.config.quant;
  assert.deepStrictEqual([q.bits, q.scheme, q.mode, q.biases, q.checked], [4, 'pow2', 'ptq', 'int', true]);
  assert.deepStrictEqual(best.params, bc.quantizeParams(srcBest.layers, srcBest.params, { bits: 4, scheme: 'pow2' }).params);
  assertPow2Grid(best.layers, best.params, q.k, 4, srcBest.params);
  // pack.js: module round trip, then the CLI (writes the .bin, re-reads it, reports bits).
  const p = pack.pack(best);
  const back = pack.unpack(p.buf);
  assert.deepStrictEqual([back.bits, back.layers, back.k], [4, best.layers, q.k]);
  assert.deepStrictEqual(back.params, best.params, 'unpacked params are identical');
  const n = P.numParams(best.layers);
  assert.strictEqual(p.buf.length, p.headerBytes + Math.ceil(n * 4 / 8));
  const binFile = path.join(tmp, 'packed.bin');
  const pk = spawnSync(process.execPath, [path.join(ROOT, 'research', 'scripts', 'prereg-distill', 'pack.js'), '--model', path.join(out, 'best.json'), '--out', binFile], { encoding: 'utf8' });
  assert.strictEqual(pk.status, 0, pk.stderr);
  const rep = JSON.parse(pk.stdout);
  assert.deepStrictEqual([rep.params, rep.paramBits, rep.kBits, rep.totalBits, rep.roundTrip], [n, 4 * n, 24, 4 * n + 24, 'identical']);
  assert.deepStrictEqual(fs.readFileSync(binFile), p.buf);
  assert.deepStrictEqual(pack.storage([44, 8, 8, 3], 8), { params: 459, paramBits: 3672, kBits: 24, totalBits: 3696 });
  assert.deepStrictEqual(pack.storage([44, 8, 8, 3], 4), { params: 459, paramBits: 1836, kBits: 24, totalBits: 1860 });
  // pack.js refuses a model that is not on its claimed grid, and a non-pow2 model.
  const off = path.join(tmp, 'pow2-off.json');
  fs.writeFileSync(off, JSON.stringify({ ...best, params: best.params.map((v, i) => (i === 0 ? v + 1e-3 : v)) }));
  const refused = spawnSync(process.execPath, [path.join(ROOT, 'research', 'scripts', 'prereg-distill', 'pack.js'), '--model', off, '--out', path.join(tmp, 'x.bin')], { encoding: 'utf8' });
  assert.notStrictEqual(refused.status, 0);
  assert.match(refused.stderr, /not an integer in/);
  assert.throws(() => pack.pack(srcBest), /not a pow2-quantized model/);
  // bc-train QAT pow2 (node backend): the same record, mode qat.
  const qat = path.join(tmp, 'qat-pow2');
  run = cli('bc-train', '--dir', qat, '--demos', await smallDir('duck'), '--hidden', '4', '--epochs', '1', '--batch', '64', '--target', '50', '--workers', '1',
    '--quant-bits', '8', '--quant-scheme', 'pow2');
  assert.strictEqual(run.status, 0, run.stderr);
  const qb = JSON.parse(fs.readFileSync(path.join(qat, 'best.json'), 'utf8'));
  assert.deepStrictEqual([qb.config.quant.scheme, qb.config.quant.mode, qb.config.quant.biases, qb.config.quant.checked], ['pow2', 'qat', 'int', true]);
  assertPow2Grid(qb.layers, qb.params, qb.config.quant.k, 8);
  assert.deepStrictEqual(pack.unpack(pack.pack(qb).buf).params, qb.params);
});

// ---------------------------------------------------------------------------
// Release-fix flags (research/prereg-teacher-duck-release-fix.md §4.1)
// ---------------------------------------------------------------------------
const { createDuckSolver } = require('./solver.js');
const NO_NUMPY = 'python backend not tested: no numpy (set PYTHON to an interpreter with numpy, e.g. the repo\'s .venv)';
// The real duck teacher over the opening (pterodactyls from about 450 points), one seed.
const OPENING = { seeds: [3, 4], levels: [0], span0: 1000, span: 5000, keepBoring: 10, gameVersion: 'v2', teacher: 'duck', workers: 1 };
async function genAt(name, o) {
  const dir = path.join(tmp, name);
  const manifest = await bc.generateDemos(dir, { ...o, coreHash: coreHash() });
  fs.writeFileSync(path.join(dir, 'demos.json'), JSON.stringify(manifest));
  return { dir, manifest, bin: fs.readFileSync(path.join(dir, 'demos.bin')) };
}

test('bc-gen scoreWindow: an all-covering window is byte-identical to none; a window keeps a strict subset', async () => {
  const all = await genAt('win-all', { ...SMALL, workers: 1, scoreWindow: [0, 1e9] });
  assert.strictEqual(all.manifest.sha256, SMALL_JUMP_SHA256);
  assert.deepStrictEqual(all.manifest.scoreWindow, [0, 1e9]);
  const part = await genAt('win-part', { ...SMALL, workers: 1, scoreWindow: [100, 200] });
  assert.ok(part.manifest.rows > 0 && part.manifest.rows < all.manifest.rows, `${part.manifest.rows} of ${all.manifest.rows}`);
  assert.ok(!('scoreWindow' in (await smallDemos()).manifest), 'a default manifest has no scoreWindow key');
});

test('DAgger relabel control: the duck teacher as its own student reproduces bc-gen rows byte for byte', async () => {
  for (const [name, extra] of [['full', {}], ['window', { scoreWindow: [400, 800] }]]) {
    const ref = await genAt(`ctl-teacher-${name}`, { ...OPENING, ...extra });
    const dag = await genAt(`ctl-student-${name}`, { ...OPENING, ...extra, createStudent: createDuckSolver, student: { path: 'test/solver.js createDuckSolver', paramsSha256: null } });
    assert.ok(ref.manifest.counts.rows.duck > 0, `${name}: the control must contain duck rows to bite`);
    assert.strictEqual(dag.manifest.sha256, ref.manifest.sha256, `${name}: rows differ`);
    assert.strictEqual(dag.manifest.source, 'dagger');
    assert.strictEqual(dag.manifest.undefinedAir, 0);
    assert.strictEqual(dag.manifest.studentCrashes, 0);
    assert.ok(!('solverCrashes' in dag.manifest) && ref.manifest.solverCrashes === 0);
  }
});

test('DAgger rollouts of a weak student: the student plays, the shadow labels, crashes are the student\'s', async () => {
  // An actor that taps jump every other frame: it takes off whenever it lands, so most of
  // its airborne frames follow takeoffs the shadow did not choose.
  const hopper = () => { let k = 0; return () => ({ jump: k++ % 2 === 0, down: false }); };
  const dag = await genAt('dag-weak', { ...OPENING, createStudent: hopper, student: { path: 'x', paramsSha256: 'y' } });
  const ref = await genAt('dag-weak-ref', OPENING);
  assert.strictEqual(dag.manifest.source, 'dagger');
  assert.deepStrictEqual(dag.manifest.student, { path: 'x', paramsSha256: 'y' });
  assert.notStrictEqual(dag.manifest.sha256, ref.manifest.sha256);
  assert.ok(dag.manifest.undefinedAir > 0, 'airborne frames of takeoffs the shadow did not choose are dropped');
  assert.strictEqual(dag.manifest.episodes.reduce((a, e) => a + e.undefinedAir, 0), dag.manifest.undefinedAir);
  assert.ok(dag.manifest.studentCrashes > 0, 'the hopper crashes');
  // A model student (plain data): the same rows through worker threads.
  const layers = [P.OBS_DIM, 3];
  const params = new Array(P.numParams(layers)).fill(0);
  params[params.length - 2] = 1;   // jump bias: holds jump
  const model = { format: P.FORMAT, layers, params, actionRepeat: 1, gameVersion: 'v2' };
  const two = await genAt('dag-weak-w2', { ...OPENING, seeds: [3, 4], workers: 2, studentModel: model });
  const one = await genAt('dag-weak-w1', { ...OPENING, seeds: [3, 4], workers: 1, studentModel: model });
  assert.strictEqual(two.manifest.sha256, one.manifest.sha256);
});

test('transitionMask: grounded onsets and releases only', () => {
  assert.strictEqual(bc.transitionMask(0, 0, 2), 1);   // onset
  assert.strictEqual(bc.transitionMask(0, 1, 0), 2);   // release
  assert.strictEqual(bc.transitionMask(0, 1, 1), 2);   // release into a jump
  assert.strictEqual(bc.transitionMask(0, 1, 2), 0);   // inside a duck
  assert.strictEqual(bc.transitionMask(0, 0, 0), 0);
  assert.strictEqual(bc.transitionMask(0, 0, 1), 0);   // a jump takeoff is not a duck transition
  assert.strictEqual(bc.transitionMask(1, 0, 2), 0);   // airborne
  assert.strictEqual(bc.transitionMask(1, 1, 0), 0);
});

test('counts.js (C-a1) counts the transitions bc_train.py weights (skipped without numpy for the python half)', async (t) => {
  const { dir } = await duckDemos();
  const r = spawnSync(process.execPath, [path.join(ROOT, 'research', 'scripts', 'prereg-teacher-duck-release-fix', 'counts.js'), '--demos', dir], { encoding: 'utf8' });
  assert.strictEqual(r.status, 0, r.stderr);
  const c = JSON.parse(r.stdout);
  assert.ok(c.total.onset > 0 && c.total.release > 0, JSON.stringify(c.total));
  assert.strictEqual(c.total.release, c.total.releaseToNone + c.total.releaseToJump);
  assert.strictEqual(c.total.rows, (await duckDemos()).manifest.rows);
  if (!hasNumpy()) { t.skip(NO_NUMPY); return; }
  const o = { hidden: [4], epochs: 1, batch: 64, lr: 1e-2, seed: 1, transitionWeight: 4 };
  const p = await bc.trainPython(path.join(dir, 'demos.bin'), path.join(tmp, 'py-tw4.json'), o, ROOT);
  assert.deepStrictEqual(p.transitions, { onset: c.total.onset, release: c.total.release, releaseToJump: c.total.releaseToJump });
});

test('python --transition-weight: 1 is identical to no flag, 4 changes the params (skipped without numpy)', async (t) => {
  if (!hasNumpy()) { t.skip(NO_NUMPY); return; }
  const bin = path.join((await duckDemos()).dir, 'demos.bin');
  const o = { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 1 };
  const plain = await bc.trainPython(bin, path.join(tmp, 'tw-none.json'), o, ROOT);
  const one = await bc.trainPython(bin, path.join(tmp, 'tw-1.json'), { ...o, transitionWeight: 1 }, ROOT);
  const four = await bc.trainPython(bin, path.join(tmp, 'tw-4.json'), { ...o, transitionWeight: 4 }, ROOT);
  assert.deepStrictEqual(one.params, plain.params);
  assert.ok(!('transitions' in plain) && !('transitions' in one));
  assert.notDeepStrictEqual(four.params, plain.params);
});

test('python --extra-demos equals training on the concatenated file (skipped without numpy)', async (t) => {
  if (!hasNumpy()) { t.skip(NO_NUMPY); return; }
  const a = path.join((await smallDemos()).dir, 'demos.bin'), b = path.join((await duckDemos()).dir, 'demos.bin');
  const cat = path.join(tmp, 'cat.bin');
  fs.writeFileSync(cat, Buffer.concat([fs.readFileSync(a), fs.readFileSync(b)]));
  const o = { hidden: [8], epochs: 2, batch: 64, lr: 1e-2, seed: 2 };
  const joined = await bc.trainPython(cat, path.join(tmp, 'x-cat.json'), o, ROOT);
  const extra = await bc.trainPython(a, path.join(tmp, 'x-extra.json'), { ...o, extraDemos: [b] }, ROOT);
  assert.deepStrictEqual(extra.params, joined.params);
});

test('cli: release-fix flags are python-only, validated, and recorded in best.json / the manifest', async (t) => {
  const { dir: jdir } = await smallDemos();
  const { dir: ddir } = await duckDemos();
  for (const flag of [['--extra-demos', jdir], ['--transition-weight', '4']]) {
    const r = cli('bc-train', '--dir', path.join(tmp, 'cli-node' + flag[0]), '--demos', jdir, '--hidden', '4', '--epochs', '1', ...flag);
    assert.notStrictEqual(r.status, 0);
    assert.match(r.stderr + r.stdout, /only supported with --backend python/);
  }
  const bad = cli('bc-gen', '--dir', path.join(tmp, 'cli-badwin'), '--seeds', '1-1', '--levels', '0', '--span0', '300', '--score-window', '800-400');
  assert.notStrictEqual(bad.status, 0);
  assert.match(bad.stderr + bad.stdout, /--score-window must be A-B/);
  // bc-gen --student with a committed duck clone, in a score window.
  const student = path.join(ROOT, 'runs', 'exp-bc-duck', 'v2-bc-duck-d2w8-s1', 'best.json');
  const sd = path.join(tmp, 'cli-dagger');
  const g = cli('bc-gen', '--dir', sd, '--seeds', '3-4', '--levels', '0', '--span0', '800', '--score-window', '400-800', '--teacher', 'duck', '--student', student, '--workers', '1');
  assert.strictEqual(g.status, 0, g.stderr);
  const m = JSON.parse(fs.readFileSync(path.join(sd, 'demos.json'), 'utf8'));
  assert.strictEqual(m.source, 'dagger');
  assert.strictEqual(m.student.path, path.relative(ROOT, student));
  assert.match(m.student.paramsSha256, /^[0-9a-f]{64}$/);
  assert.deepStrictEqual(m.scoreWindow, [400, 800]);
  assert.ok(m.rows > 0 && m.counts.rows.duck > 0);
  if (!hasNumpy()) { t.skip(NO_NUMPY); return; }
  // A teacher mismatch between base and extra demos is refused.
  const mis = cli('bc-train', '--dir', path.join(tmp, 'cli-mis'), '--demos', ddir, '--extra-demos', jdir, '--hidden', '4', '--epochs', '1', '--backend', 'python', '--target', '50', '--workers', '1');
  assert.notStrictEqual(mis.status, 0);
  assert.match(mis.stderr + mis.stdout, /teacher jump, base demos duck/);
  const out = path.join(tmp, 'cli-py');
  const r = cli('bc-train', '--dir', out, '--demos', ddir, '--extra-demos', sd, '--transition-weight', '4', '--hidden', '4', '--epochs', '1', '--batch', '64',
    '--backend', 'python', '--target', '50', '--workers', '1');
  assert.strictEqual(r.status, 0, r.stderr);
  const best = JSON.parse(fs.readFileSync(path.join(out, 'best.json'), 'utf8'));
  assert.strictEqual(best.config.extraDemos.length, 1);
  assert.strictEqual(best.config.extraDemos[0].demosSha256, m.sha256);
  assert.strictEqual(best.config.transitionWeight, 4);
  assert.ok(best.config.transitions.onset > 0);
  // The release-fix flags do not combine with distillation (teacher logits cover the base demos only).
  const dz = cli('bc-train', '--dir', path.join(tmp, 'cli-dz'), '--demos', ddir, '--extra-demos', sd, '--distill-from', student, '--hidden', '4', '--epochs', '1', '--backend', 'python');
  assert.notStrictEqual(dz.status, 0);
  assert.match(dz.stderr + dz.stdout, /do not combine with --distill-from/);
});

test('bc-train-batch: each member equals bc-train --backend python byte for byte; own run dir, train.log, config.batchRun; bad flags refused', async (t) => {
  const { dir } = await smallDemos();
  const bad = [
    [['--dir', path.join(tmp, 'x'), '--shapes', '4', '--seeds', '1', '--demos', dir], /--dir-pattern, not --dir/],
    [['--dir-pattern', path.join(tmp, 'x-{seed}'), '--hidden', '4', '--seeds', '1', '--demos', dir], /not --hidden/],
    [['--dir-pattern', path.join(tmp, 'x-{seed}'), '--shapes', '4/6', '--seeds', '1', '--demos', dir], /same dir/],
    [['--dir-pattern', path.join(tmp, 'x-{width}-{seed}'), '--shapes', '4,6', '--seeds', '1', '--demos', dir], /not uniform/],
    [['--dir-pattern', path.join(tmp, 'x-{hidden}-{seed}'), '--shapes', '4,y', '--seeds', '1', '--demos', dir], /bad hidden list/],
    [['--dir-pattern', path.join(tmp, 'x-{hidden}-{seed}'), '--shapes', '4', '--seeds', '1', '--demos', dir, '--backend', 'node'], /must be python or mlx/],
  ];
  for (const [args, re] of bad) {
    const r = cli('bc-train-batch', ...args);
    assert.notStrictEqual(r.status, 0, args.join(' '));
    assert.match(r.stderr + r.stdout, re);
  }
  if (!hasNumpy()) { t.skip(NO_NUMPY); return; }
  const common = ['--demos', dir, '--epochs', '2', '--batch', '64', '--target', '50', '--workers', '1'];
  const b = cli('bc-train-batch', '--dir-pattern', path.join(tmp, 'batch', 'd{depth}w{width}-s{seed}'), '--shapes', '4x2/6', '--seeds', '2-3', '--jobs', '3', '--snapshots', ...common);
  assert.strictEqual(b.status, 0, b.stderr + b.stdout);
  const runs = ['d2w4-s2', 'd2w4-s3', 'd1w6-s2', 'd1w6-s3'];
  for (const [i, name] of runs.entries()) {
    const d = path.join(tmp, 'batch', name);
    assert.deepStrictEqual(fs.readdirSync(d).sort(), ['bc-metrics.jsonl', 'best.json', 'snapshots', 'train.log']);
    const best = JSON.parse(fs.readFileSync(path.join(d, 'best.json'), 'utf8'));
    assert.strictEqual(best.config.batch, 64);   // the minibatch size, unchanged
    assert.strictEqual(best.config.trainer, 'trainer/cli.js bc-train-batch --backend python');
    assert.deepStrictEqual({ ...best.config.batchRun, id: null }, { id: null, index: i, size: 4, backend: 'python', jobs: 3, blasThreads: 1 });
    const log = fs.readFileSync(path.join(d, 'train.log'), 'utf8');
    assert.match(log.split('\n')[0], new RegExp(`bc-train: dir=\\S+ backend=python layers=${best.layers.join('-')} epochs 2 .* batch=${i + 1}/4 jobs=3$`));
    assert.match(log, /validation: median/);
  }
  // The same model through bc-train: identical params, validation and metrics (seconds aside).
  const s = path.join(tmp, 'batch-single');
  const r = cli('bc-train', '--dir', s, '--hidden', '4,4', '--seed', '3', '--backend', 'python', '--snapshots', ...common);
  assert.strictEqual(r.status, 0, r.stderr);
  const one = (d) => {
    const best = JSON.parse(fs.readFileSync(path.join(d, 'best.json'), 'utf8'));
    return { params: best.params, validation: best.validation, final: best.config.final,
      metrics: fs.readFileSync(path.join(d, 'bc-metrics.jsonl'), 'utf8').replace(/"seconds":[0-9.e-]+/g, ''),
      snap: fs.readFileSync(path.join(d, 'snapshots', 'epoch-02.json'), 'utf8') };
  };
  assert.deepStrictEqual(one(path.join(tmp, 'batch', 'd2w4-s3')), one(s));
  // A dir with best.json stops the whole batch before any training, unless --overwrite.
  const again = cli('bc-train-batch', '--dir-pattern', path.join(tmp, 'batch', 'd{depth}w{width}-s{seed}'), '--shapes', '6', '--seeds', '3-4', ...common);
  assert.notStrictEqual(again.status, 0);
  assert.match(again.stderr + again.stdout, /best\.json exists; pass --overwrite/);
  assert.ok(!fs.existsSync(path.join(tmp, 'batch', 'd1w6-s4')));
});

test('packed tier A: scoreSeedsMany gives each model exactly its own scoreSeeds scores (models of different shapes in one pool run)', async () => {
  const { scoreSeedsMany, scoreSeeds } = require('../trainer/cli.js');
  const { createPool } = require('../trainer/pool.js');
  const { validationSeeds } = require('../trainer/env.js');
  const load = (f) => { const m = JSON.parse(fs.readFileSync(path.join(ROOT, f), 'utf8')); return { layers: m.layers, params: m.params, actionRepeat: 1, gameVersion: 'v2' }; };
  const jump = load('runs/v2-bc-jump-d2w32-s1/best.json');
  // Damaged copies crash at seed-dependent scores, so a model/seed mix-up would show.
  const damage = (m, f) => ({ ...m, params: m.params.map((v, i) => v * (1 + f * Math.sin(i * 12.9898))) });
  const models = [damage(jump, 0.6), load('runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json'), damage(jump, 0.9), load('runs/v2-es-d2w32-s1-t1m/best.json')];
  const seeds = validationSeeds(25), target = 3000;
  const pool = createPool(3);
  try {
    const packed = await scoreSeedsMany(pool, models, seeds, target);
    for (const [k, m] of models.entries()) assert.deepStrictEqual(packed[k], await scoreSeeds(pool, m, seeds, target), `model ${k}`);
    const distinct = new Set(packed.flat());
    assert.ok(distinct.size >= 5, `the test needs varied scores to detect a mix-up, got ${[...distinct].join(',')}`);
    assert.notDeepStrictEqual(packed[0], packed[2]);
  } finally {
    await pool.terminate();
  }
  await assert.rejects(scoreSeedsMany(null, [models[0], { ...models[1], gameVersion: 'v3' }], seeds, target), /differ in actionRepeat or gameVersion/);
});

// The MLX engine (Apple GPU) needs an interpreter with mlx: PYTHON_MLX or <root>/.venv-mlx.
const hasMlx = () => {
  let py;
  try { py = bc.mlxPythonPath(ROOT); } catch (e) { return false; }
  const probe = spawnSync(py, ['-c', 'import mlx.core'], { encoding: 'utf8' });
  return !probe.error && probe.status === 0;
};
const NO_MLX = 'mlx backend not tested: no interpreter with mlx (set PYTHON_MLX, e.g. to the repo\'s .venv-mlx/bin/python)';

// Compares against tmp/batch-single, the numpy run written by the bc-train-batch test above
// (tests in a file run in order), so run both together (e.g. --test-name-pattern 'bc-train-batch|mlx').
test('mlx backend: bc-train and bc-train-batch match the numpy backend to float32 noise, same tier A; engine and batchRun recorded (skipped without mlx)', async (t) => {
  const { dir } = await smallDemos();
  const jm = cli('bc-train-batch', '--dir-pattern', path.join(tmp, 'mx-x-{seed}'), '--shapes', '4', '--seeds', '1', '--demos', dir, '--backend', 'mlx', '--jobs', '2');
  assert.notStrictEqual(jm.status, 0);
  if (!hasMlx() || !hasNumpy()) { t.skip(hasMlx() ? NO_NUMPY : NO_MLX); return; }
  assert.match(jm.stderr + jm.stdout, /--jobs is for --backend python/);
  const common = ['--demos', dir, '--epochs', '2', '--batch', '64', '--target', '50', '--workers', '1'];
  const b = cli('bc-train-batch', '--backend', 'mlx', '--dir-pattern', path.join(tmp, 'mxb', 'd{depth}w{width}-s{seed}'), '--shapes', '4x2/6', '--seeds', '2-3', '--snapshots', ...common);
  assert.strictEqual(b.status, 0, b.stderr + b.stdout);
  const s = path.join(tmp, 'mx-single');
  assert.strictEqual(cli('bc-train', '--dir', s, '--hidden', '4,4', '--seed', '3', '--backend', 'mlx', ...common).status, 0);
  const load = (d) => JSON.parse(fs.readFileSync(path.join(d, 'best.json'), 'utf8'));
  const np = load(path.join(tmp, 'batch-single'));   // the numpy run of the same model (bc-train-batch test above)
  for (const m of [load(path.join(tmp, 'mxb', 'd2w4-s3')), load(s)]) {
    assert.strictEqual(m.config.backend, 'mlx');
    assert.strictEqual(m.config.engine.name, 'mlx');
    assert.strictEqual(m.config.engine.tf32, false);
    assert.strictEqual(m.params.length, np.params.length);
    const d = Math.max(...m.params.map((v, i) => Math.abs(v - np.params[i])));
    assert.ok(d < 1e-4, `max |params - numpy params| ${d}`);
    assert.deepStrictEqual(m.validation, np.validation);
  }
  const g = load(path.join(tmp, 'mxb', 'd1w6-s2')).config;
  assert.deepStrictEqual({ ...g.batchRun, id: null }, { id: null, index: 2, size: 4, backend: 'mlx', group: '6', groupSize: 2, groupIndex: 0 });
  assert.strictEqual(g.batch, 64);
  assert.ok(fs.existsSync(path.join(tmp, 'mxb', 'd1w6-s2', 'snapshots', 'epoch-02.json')));
  assert.match(fs.readFileSync(path.join(tmp, 'mxb', 'd1w6-s2', 'train.log'), 'utf8').split('\n')[0], /backend=mlx layers=44-6-3 .* batch=3\/4$/);
  // Default backend: mlx when an mlx interpreter is configured, python with --jobs or without one
  // (each run below stops at the existing best.json, after logging its choice).
  const pat = ['--dir-pattern', path.join(tmp, 'mxb', 'd{depth}w{width}-s{seed}'), '--shapes', '6', '--seeds', '2', ...common];
  assert.match(cli('bc-train-batch', ...pat).stdout, /--backend mlx \(default: mlx interpreter found\)/);
  assert.match(cli('bc-train-batch', ...pat, '--jobs', '2').stdout, /--backend python \(default: mlx interpreter found, but --jobs given\)/);
  if (!fs.existsSync(path.join(ROOT, '.venv-mlx'))) {
    const env = { ...process.env }; delete env.PYTHON_MLX;
    const r = spawnSync(process.execPath, [path.join(ROOT, 'trainer', 'cli.js'), 'bc-train-batch', ...pat], { encoding: 'utf8', env });
    assert.match(r.stdout, /--backend python \(default: no \.venv-mlx or PYTHON_MLX\)/);
  }
});
