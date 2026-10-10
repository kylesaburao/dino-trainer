'use strict';
// Robustness scan runner (research/prereg-bc-robustness.md §4).
// Plays full v2 games from 0 to --target on seeds BASE..BASE+N-1 with one model,
// on the trainer's worker pool (the same job path as `trainer/cli.js eval`:
// setup sigma 0, noiseSeed 0, sign 0, one seed per job), and appends one JSON
// line per seed to --out. The first line is a header (model, hash, flags).
// Usage: node scan.js --model <best.json> --out <arm.jsonl> [--base 30000000] [--n 2000]
//          [--target 1000000] [--workers 18] [--chunk 180] [--wasm]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const { createPool, defaultWorkers } = require(path.join(ROOT, 'trainer', 'pool.js'));
const { coreHash } = require(path.join(ROOT, 'trainer', 'store.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const modelFile = path.resolve(opt('--model'));
const out = path.resolve(opt('--out'));
const BASE = Number(opt('--base', 30000000)), N = Number(opt('--n', 2000));
const TARGET = Number(opt('--target', 1000000));
const WORKERS = Number(opt('--workers', defaultWorkers()));
const CHUNK = Number(opt('--chunk', 180));
const wasm = args.includes('--wasm');
const EXPECT_HASH = '0f329d4e';

async function main() {
  const m = JSON.parse(fs.readFileSync(modelFile, 'utf8'));
  const hash = coreHash();
  if (!hash.startsWith(EXPECT_HASH)) throw new Error(`core hash ${hash} != ${EXPECT_HASH}…`);
  if (m.gameVersion !== 'v2') throw new Error(`model gameVersion ${m.gameVersion}`);
  if (wasm) require(path.join(ROOT, 'trainer', 'wasm-net.js')).assertAvailable();
  const seeds = Array.from({ length: N }, (_, i) => BASE + i);
  // Resume: skip seeds already in --out.
  const done = new Set();
  if (fs.existsSync(out)) for (const l of fs.readFileSync(out, 'utf8').split('\n')) {
    if (!l) continue; const r = JSON.parse(l); if (r.seed !== undefined) done.add(r.seed);
  } else {
    fs.writeFileSync(out, JSON.stringify({ header: true, model: modelFile, layers: m.layers, gameVersion: m.gameVersion, actionRepeat: m.actionRepeat,
      generation: m.generation ?? null, coreHash: hash, base: BASE, n: N, target: TARGET, wasm, workers: WORKERS, start: new Date().toISOString() }) + '\n');
  }
  const todo = seeds.filter(s => !done.has(s));
  const pool = createPool(WORKERS);
  const setup = { theta: m.params, layers: m.layers, actionRepeat: m.actionRepeat, gameVersion: m.gameVersion, sigma: 0, target: TARGET, wasm };
  const t0 = Date.now();
  let fails = 0, n = 0;
  for (let i = 0; i < todo.length; i += CHUNK) {
    const chunk = todo.slice(i, i + CHUNK);
    const res = await pool.run(setup, chunk.map(s => ({ noiseSeed: 0, sign: 0, seeds: [s] })));
    const lines = res.map((r, j) => {
      const score = r.scores[0], pass = score >= TARGET;
      if (!pass) fails++;
      return JSON.stringify({ seed: chunk[j], score, pass, frames: r.frames });
    });
    fs.appendFileSync(out, lines.join('\n') + '\n');
    n += chunk.length;
    console.log(`[${new Date().toISOString()}] ${n}/${todo.length} seeds, ${fails} failures, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await pool.terminate();
  fs.appendFileSync(out, JSON.stringify({ footer: true, seconds: (Date.now() - t0) / 1000, end: new Date().toISOString() }) + '\n');
  console.log(`done: ${fails} failures in ${n} games, ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
main().catch((e) => { console.error(e); process.exit(1); });
