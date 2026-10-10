'use strict';
// Robustness scan runner (research/prereg-bc-robustness.md §4).
// Plays full games (the model's rules) from 0 (or --start) to --target on seeds BASE..BASE+N-1 with one model,
// on the trainer's worker pool (the same job path as `trainer/cli.js eval`:
// setup sigma 0, noiseSeed 0, sign 0, one seed per job), and appends one JSON
// line per seed to --out. The first line is a header (model, hash, flags, and the
// runner's pid and host, so trainer/dashboard.js can tell a live scan from a dead one);
// a resumed scan appends a {resume, pid, host, start} line before its new seeds, and every
// completed pass ends with a footer line. Readers select seed lines with `seed !== undefined`.
// Usage: node scan.js --model <best.json> --out <arm.jsonl> [--base 30000000] [--n 2000]
//          [--target 1000000] [--workers 18] [--chunk 180] [--wasm]
//          [--game v2|v3] [--expect-hash <prefix>] [--start S]
// (research/prereg-v3-clones.md §4.0; every option below leaves the default scan unchanged)
//   Rules: games are played on the model's own gameVersion (v2 or v3). --game <v> plays other
//     rules (a cross-rules scan, as `assess --game`): warned, and the header records
//     gameVersion = game = the rules played and modelGameVersion = the model's.
//   Core hash: the current core.js/policy.js hash must start with --expect-hash, which
//     defaults to the model's own best.json coreHash; a scan on another core is refused unless
//     the flag is passed (e.g. --expect-hash 65f3546c for a pre-v3 model on rules v2, which
//     AGENTS.md "Core hash" justifies). The header records expectHash.
//   --start S: every episode starts at score S (createGame startScore) and ends at --target
//     (absolute), through the worker's per-job starts/caps. The header records startScore
//     (default 0; the header's `start` stays the ISO start time, which trainer/scans.js reads).
const fs = require('fs');
const os = require('os');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const { createPool, defaultWorkers } = require(path.join(ROOT, 'trainer', 'pool.js'));
const { coreHash } = require(path.join(ROOT, 'trainer', 'store.js'));
const { GAME_VERSIONS } = require(path.join(ROOT, 'core.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const modelFile = path.resolve(opt('--model'));
const out = path.resolve(opt('--out'));
const BASE = Number(opt('--base', 30000000)), N = Number(opt('--n', 2000));
const TARGET = Number(opt('--target', 1000000));
const WORKERS = Number(opt('--workers', defaultWorkers()));
const CHUNK = Number(opt('--chunk', 180));
const wasm = args.includes('--wasm');
const GAME = opt('--game', null);
const START = Number(opt('--start', 0));
const PLAYABLE = ['v2', 'v3'];

async function main() {
  const m = JSON.parse(fs.readFileSync(modelFile, 'utf8'));
  const hash = coreHash();
  const expectHash = opt('--expect-hash', m.coreHash || null);
  if (!expectHash) throw new Error(`${modelFile} records no coreHash: pass --expect-hash <prefix>`);
  if (!hash.startsWith(expectHash)) {
    throw new Error(`core hash ${hash} != ${expectHash}…${opt('--expect-hash') ? '' : ` (the model's coreHash); pass --expect-hash ${hash.slice(0, 8)} to scan it on this core`}`);
  }
  const modelGame = m.gameVersion;
  if (!PLAYABLE.includes(modelGame)) throw new Error(`model gameVersion ${modelGame}`);
  if (GAME !== null && !(GAME_VERSIONS.includes(GAME) && PLAYABLE.includes(GAME))) throw new Error(`--game must be one of ${PLAYABLE.join(', ')}`);
  const game = GAME || modelGame;
  if (game !== modelGame) console.warn(`WARNING: cross-rules scan: a ${modelGame} model on rules ${game}`);
  if (!(Number.isFinite(START) && START >= 0 && START < TARGET)) throw new Error(`--start must be in [0, --target), got ${START}`);
  if (wasm) require(path.join(ROOT, 'trainer', 'wasm-net.js')).assertAvailable();
  const seeds = Array.from({ length: N }, (_, i) => BASE + i);
  // Resume: skip seeds already in --out.
  const done = new Set();
  if (fs.existsSync(out)) {
    for (const l of fs.readFileSync(out, 'utf8').split('\n')) {
      if (!l) continue; const r = JSON.parse(l); if (r.seed !== undefined) done.add(r.seed);
    }
    fs.appendFileSync(out, JSON.stringify({ resume: true, pid: process.pid, host: os.hostname(), start: new Date().toISOString() }) + '\n');
  } else {
    fs.writeFileSync(out, JSON.stringify({ header: true, model: modelFile, layers: m.layers, gameVersion: game, actionRepeat: m.actionRepeat,
      generation: m.generation ?? null, coreHash: hash, base: BASE, n: N, target: TARGET, wasm, workers: WORKERS, start: new Date().toISOString(),
      pid: process.pid, host: os.hostname(), game, modelGameVersion: modelGame, startScore: START, expectHash }) + '\n');
  }
  const todo = seeds.filter(s => !done.has(s));
  const pool = createPool(WORKERS);
  const setup = { theta: m.params, layers: m.layers, actionRepeat: m.actionRepeat, gameVersion: game, sigma: 0, target: TARGET, wasm };
  // Without --start the job is exactly the old one (no starts/caps: the worker's full-game path).
  const job = START ? (s) => ({ noiseSeed: 0, sign: 0, seeds: [s], starts: [START], caps: [TARGET] })
    : (s) => ({ noiseSeed: 0, sign: 0, seeds: [s] });
  const t0 = Date.now();
  let fails = 0, n = 0;
  for (let i = 0; i < todo.length; i += CHUNK) {
    const chunk = todo.slice(i, i + CHUNK);
    const res = await pool.run(setup, chunk.map(job));
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
