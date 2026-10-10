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
