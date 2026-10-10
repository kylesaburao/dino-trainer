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
