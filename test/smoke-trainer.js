'use strict';
// End-to-end smoke test of the training pipeline with tiny settings (NOT real
// training). Run: `node test/smoke-trainer.js`. Uses runs/smoke (gitignored).
//
// Checks: checkpoints + metrics written; SIGINT mid-generation saves the last
// completed generation and releases the lock; a concurrent trainer is refused;
// a stale lock (dead PID) is replaced; --resume continues the generation count;
// a core-hash mismatch is refused without --force (on a temporary copy of the
// repo, never the real core); eval and export work on the smoke checkpoint.
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'trainer', 'cli.js');
const DIR = path.join(ROOT, 'runs', 'smoke');
const LOCK = path.join(DIR, 'train.lock');
const CKPT = path.join(DIR, 'checkpoint.json');
const TINY = ['--target', '10000', '--pop', '8', '--episodes', '3000', '--workers', '3', '--time-budget', '2',
  '--checkpoint-every', '1', '--eval-every', '1'];

const results = [];
function check(name, fn) {
  try {
    fn();
    results.push(['PASS', name]);
    console.log(`PASS  ${name}`);
  } catch (e) {
    results.push(['FAIL', name, e.message]);
    console.log(`FAIL  ${name}: ${e.message}`);
  }
}
const run = (args, cwd = ROOT, cli = CLI) => spawnSync(process.execPath, [cli, ...args], { cwd, encoding: 'utf8', timeout: 180000 });
const readJson = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const generations = () => fs.readFileSync(path.join(DIR, 'metrics.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l).generation);

function startTrainer(args) {
  const child = spawn(process.execPath, [CLI, 'train', ...args], { cwd: ROOT });
  child.out = '';
  child.stdout.on('data', (d) => { child.out += d; process.stdout.write('    | ' + d); });
  child.stderr.on('data', (d) => { child.out += d; process.stdout.write('    | ' + d); });
  child.done = new Promise((res) => child.on('exit', (code, signal) => res({ code, signal })));
  return child;
}
const waitFor = async (pred, ms = 180000) => {
  const t0 = Date.now();
  while (!pred()) {
    if (Date.now() - t0 > ms) throw new Error('timed out waiting');
    await new Promise(r => setTimeout(r, 50));
  }
};

(async () => {
  // 1. Fresh tiny run, interrupted with SIGINT mid-generation.
  console.log('--- train (fresh, 3 generations max) then SIGINT mid-generation');
  const a = startTrainer([...TINY, '--dir', DIR, '--generations', '3', '--overwrite']);
  const ckptGen = () => { try { return readJson(CKPT).generation; } catch (e) { return -1; } };
  await waitFor(() => /gen 1:/.test(a.out) && ckptGen() === 1);
  check('checkpoint written after generation 1', () => assert.strictEqual(readJson(CKPT).generation, 1));
  check('metrics logged (jsonl + csv)', () => {
    assert.deepStrictEqual(generations(), [1]);
    assert.strictEqual(fs.readFileSync(path.join(DIR, 'metrics.csv'), 'utf8').trim().split('\n').length, 2);
  });
  check('lockfile holds the trainer PID while running', () => assert.strictEqual(readJson(LOCK).pid, a.pid));

  console.log('--- second concurrent trainer on the same directory');
  const b = run(['train', '--resume', '--dir', DIR, '--workers', '1']);
  check('second concurrent train is refused by the lockfile', () => {
    assert.strictEqual(b.status, 1, b.stdout + b.stderr);
    assert.match(b.stderr, /held by running trainer PID/);
  });

  await new Promise(r => setTimeout(r, 300));   // let generation 2 get under way
  a.kill('SIGINT');
  const ea = await a.done;
  const savedGen = readJson(CKPT).generation;
  check('SIGINT saves a checkpoint and exits 130', () => {
    assert.strictEqual(ea.code, 130);
    assert.match(a.out, /checkpoint saved: generation \d+ \(SIGINT\)/);
    assert.strictEqual(readJson(CKPT).reason, 'SIGINT');
  });
  check(`SIGINT checkpoint is the last completed generation (${savedGen})`, () => {
    assert.ok(savedGen >= 1 && savedGen < 3);
    assert.strictEqual(Math.max(...generations()), savedGen);
  });
  check('lockfile released after SIGINT', () => assert.ok(!fs.existsSync(LOCK)));

  // 2. Resume continues from the saved generation.
  console.log('--- train --resume');
  const r = run(['train', '--resume', '--dir', DIR, '--workers', '3', '--time-budget', '2']);
  process.stdout.write(r.stdout.replace(/^/gm, '    | '));
  check('resume continues from the saved generation to 3', () => {
    assert.strictEqual(r.status, 0, r.stderr);
    assert.match(r.stdout, new RegExp(`resuming .* at generation ${savedGen}`));
    assert.deepStrictEqual(generations(), [1, 2, 3]);
    assert.strictEqual(readJson(CKPT).generation, 3);
    assert.ok(!fs.existsSync(LOCK));
  });

  // 3. Stale lock from a dead process is replaced.
  fs.writeFileSync(LOCK, JSON.stringify({ pid: 999999, startedAt: 'stale' }));
  const s = run(['train', '--resume', '--dir', DIR, '--workers', '1']);
  check('stale lock (dead PID) is detected and replaced', () => {
    assert.strictEqual(s.status, 0, s.stderr);
    assert.match(s.stdout, /removing stale lock/);
    assert.ok(!fs.existsSync(LOCK));
  });

  // 4. Fresh train refuses to clobber an existing checkpoint.
  const c = run(['train', '--dir', DIR, '--workers', '1']);
  check('fresh train refuses to overwrite an existing checkpoint', () => {
    assert.strictEqual(c.status, 1);
    assert.match(c.stderr, /use --resume/);
  });

  // 5. Core-hash mismatch, on a temporary copy of the repo.
  console.log('--- core hash mismatch on a temporary copy');
  const { coreHash } = require('../trainer/store.js');
  const realHash = coreHash();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-hash-'));
  for (const f of ['core.js', 'policy.js', 'trainer', 'test']) fs.cpSync(path.join(ROOT, f), path.join(tmp, f), { recursive: true });
  fs.mkdirSync(path.join(tmp, 'runs', 'smoke'), { recursive: true });
  fs.copyFileSync(CKPT, path.join(tmp, 'runs', 'smoke', 'checkpoint.json'));
  fs.appendFileSync(path.join(tmp, 'core.js'), '\n// deliberate edit to change the core hash\n');
  const tmpCli = path.join(tmp, 'trainer', 'cli.js');
  const m1 = run(['train', '--resume', '--dir', path.join(tmp, 'runs', 'smoke'), '--workers', '1'], tmp, tmpCli);
  check('resume refuses a core-hash mismatch', () => {
    assert.strictEqual(m1.status, 1);
    assert.match(m1.stderr, /core hash mismatch/);
  });
  const m2 = run(['train', '--resume', '--force', '--dir', path.join(tmp, 'runs', 'smoke'), '--workers', '1'], tmp, tmpCli);
  check('resume with --force accepts the mismatch', () => {
    assert.strictEqual(m2.status, 0, m2.stderr);
    assert.match(m2.stdout, /WARNING: core hash mismatch/);
  });
  fs.rmSync(tmp, { recursive: true, force: true });
  check('real core.js untouched by the mismatch test', () => assert.strictEqual(coreHash(), realHash));

  // 6. eval + export.
  console.log('--- eval and export');
  const e = run(['eval', '--dir', DIR, '--which', 'best', '--workers', '3']);
  process.stdout.write(e.stdout.replace(/^/gm, '    | '));
  check('eval runs the 25 held-out seeds', () => {
    assert.strictEqual(e.status, 0, e.stderr);
    assert.match(e.stdout, /held-out eval \(25 seeds, cap 10000\)/);
  });
  const x = run(['export', '--dir', DIR, '--which', 'best', '--out', path.join(DIR, 'model.js')]);
  process.stdout.write(x.stdout.replace(/^/gm, '    | '));
  check('export writes model.js usable via require()', () => {
    assert.strictEqual(x.status, 0, x.stderr);
    const file = path.join(DIR, 'model.js');
    delete require.cache[file];
    const model = require(file);
    const { createAgent } = require('../policy.js');
    assert.ok(createAgent(model));
    assert.strictEqual(model.params.length, 771);
  });
  const e2 = run(['eval', '--model', path.join(DIR, 'model.js'), '--workers', '1']);
  check('eval of exported model.js matches eval of best.json', () => {
    const med = (s) => s.match(/median (\S+)\s+min (\S+)\s+max (\S+)/).slice(1).join(' ');
    assert.strictEqual(med(e2.stdout), med(e.stdout));
  });

  const failed = results.filter(r => r[0] === 'FAIL');
  console.log(`\n${results.length - failed.length}/${results.length} smoke checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
