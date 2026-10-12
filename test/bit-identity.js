#!/usr/bin/env node
'use strict';
// Bit-identity harness: proves that a trainer change leaves existing behaviour unchanged.
//
//   node test/bit-identity.js <oldTree> <newTree> [--workers N] [--keep] [--only <scenario>[,<scenario>...]] [--allow-busy]
//                             [--core-changed]
// --core-changed: core.js/policy.js were edited on purpose (e.g. a new game version), so the
// coreHash fields are expected to differ; they are reported as a note, not compared.
// Everything else must still be identical.
//
// For each scenario, runs `node trainer/cli.js train ...` with identical arguments in
// both checkouts (old = reference, new = candidate), each in its own scratch run dir
// <tree>/runs/exp-bitid-<scenario> (gitignored), and diffs the outputs after every phase:
//   - metrics.jsonl: equal row count; per row every field present in both rows must be
//     identical, except the timing fields (time, genMs, fps, cumulativeMs). Fields only in
//     the new rows are allowed (new metric columns); fields only in the old rows are a difference.
//   - checkpoint.json: theta, adam, rngState, best, heldOut, lastValidated, generation,
//     reason, sigma, stallMult, annealedAt, lastValMin, window, layers, coreHash; config on
//     the keys present in both (new config keys are listed, not compared; the runtime key
//     wasm, the network kernel, is skipped: RUNTIME_CONFIG).
//   - best.json: everything except config, savedAt and timestamps; coreHash on its own line.
// Every compared scalar is counted. Prints per scenario `IDENTICAL (n values)` or the first
// difference, then an overall verdict; exits 0 only if everything is identical.
// The scratch run dirs are deleted afterwards unless --keep.
//
// Not a node:test file (node --test test/ only picks up *.test.js). It runs real, small
// trainings, so it refuses to start while another trainer/cli.js process is running
// (`pgrep -fl trainer/cli.js`) unless --allow-busy.
const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const SMALL = ['--hidden', '8,8'];
// Each scenario: phases of train arguments (without --dir/--workers), plus optional checks.
const SCENARIOS = [
  { name: 'plain', phases: [[...SMALL, '--target', '2000', '--generations', '20'], ['--resume', '--generations', '40']] },
  { name: 'plain-2x', phases: [['--hidden', '16,16,16', '--target', '2000', '--generations', '20'], ['--resume', '--generations', '40']] },
  // checkCurriculum: --start-min >= 2000 and (log) start-min + window <= target, so target 3000.
  { name: 'curric-log', phases: [[...SMALL, '--window', '500', '--start-min', '2000', '--target', '3000', '--generations', '20'], ['--resume', '--generations', '40']] },
  { name: 'curric-points', phases: [[...SMALL, '--window', '500', '--start-min', '2000', '--start-dist', 'points', '--target', '3000', '--generations', '20'], ['--resume', '--generations', '40']] },
  // An 8,8 net sits on the ~54-point plateau (validation median <= 54) for hundreds of
  // generations, so --target 300 never stops through the held-out path. Seed 2 at target
  // 52 passes its first held-out check at generation 81 (measured at 2c1e660).
  {
    name: 'heldout', phases: [[...SMALL, '--seed', '2', '--target', '52', '--generations', '300', '--eval-every', '1']],
    expectReason: /^held-out /,
  },
  // The allpass stop path: seed 3 at target 48 (measured at 322fcaf, 4 workers) aborts
  // held-out attempt 1 (best of gen 18) after its first round of --workers seeds (1/4
  // passed); phase 1 ends at generation 19 so its checkpoint records that attempt. The
  // resume stops at generation 20, when attempt 2 passes all 8 seeds.
  {
    name: 'allpass', phases: [[...SMALL, '--seed', '3', '--target', '48', '--generations', '19', '--eval-every', '1',
      '--select', 'count', '--stop-rule', 'allpass:8', '--val-seeds', '6'], ['--resume', '--generations', '150']],
    expectReason: /^held-out 8\/8 seeds reached target 48$/,
  },
  {
    name: 'hygiene', phases: [[...SMALL, '--target', '2000', '--generations', '20', '--rank-ties', 'mid', '--decay', 'decoupled',
      '--weight-decay', '0.01', '--select', 'count', '--stop-rule', 'allpass:5', '--val-seeds', '6', '--fold-validation', '--stall-rule'],
    ['--resume', '--generations', '40']],
  },
];

let CORE_CHANGED = false;    // --core-changed
const METRIC_SKIP = new Set(['time', 'genMs', 'fps', 'cumulativeMs']);
const CKPT_FIELDS = ['theta', 'adam', 'rngState', 'best', 'heldOut', 'lastValidated', 'generation', 'reason', 'sigma',
  'stallMult', 'annealedAt', 'lastValMin', 'window', 'layers', 'coreHash'];
const BEST_SKIP = new Set(['config', 'savedAt', 'timestamps', 'coreHash']);
// Checkpoint config keys that are runtime choices, not training settings: neither compared
// nor required in both. wasm is the network kernel (wasm-net.js vs policy.js), whose decisions
// are bit-identical (test/wasm-net.test.js); trees before 2026-10-12 record wasm: false, later
// ones run the kernel by default and do not record it.
const RUNTIME_CONFIG = new Set(['wasm']);

function parseArgs(argv) {
  const out = { trees: [], workers: 4, keep: false, only: null, allowBusy: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--workers') out.workers = Number(argv[++i]);
    else if (a === '--keep') out.keep = true;
    else if (a === '--allow-busy') out.allowBusy = true;
    else if (a === '--core-changed') out.coreChanged = true;
    else if (a === '--only') out.only = String(argv[++i]).split(',');
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else out.trees.push(path.resolve(a));
  }
  if (out.trees.length !== 2) throw new Error('usage: node test/bit-identity.js <oldTree> <newTree> [--workers N] [--keep] [--only <scenario>]');
  if (!(Number.isInteger(out.workers) && out.workers > 0)) throw new Error('--workers must be a positive integer');
  for (const t of out.trees) if (!fs.existsSync(path.join(t, 'trainer', 'cli.js'))) throw new Error(`${t}: no trainer/cli.js`);
  if (out.trees[0] === out.trees[1]) throw new Error('old and new tree must differ (the run dirs would collide)');
  if (out.only) {
    const names = SCENARIOS.map(s => s.name);
    for (const n of out.only) if (!names.includes(n)) throw new Error(`unknown scenario ${n} (one of ${names.join(', ')})`);
  }
  return out;
}

// The scratch run dir of a scenario in a tree; deleted only by this exact path.
function runDir(tree, name) {
  const dir = path.join(tree, 'runs', `exp-bitid-${name}`);
  if (path.dirname(dir) !== path.join(tree, 'runs') || !path.basename(dir).startsWith('exp-bitid-')) throw new Error(`bad run dir ${dir}`);
  return dir;
}

// Run one train phase in a tree; resolves to { code, ms }. Output goes to <dir>/bitid.log.
function train(tree, dir, args, workers) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(dir, { recursive: true });
    const logFd = fs.openSync(path.join(dir, 'bitid.log'), 'a');
    const argv = [path.join(tree, 'trainer', 'cli.js'), 'train', ...args, '--dir', dir, '--workers', String(workers)];
    fs.writeSync(logFd, `$ node ${argv.join(' ')}\n`);
    const t0 = Date.now();
    const child = spawn(process.execPath, argv, { cwd: tree, stdio: ['ignore', logFd, logFd] });
    child.on('error', reject);
    child.on('exit', (code, signal) => { fs.closeSync(logFd); resolve({ code: code ?? signal, ms: Date.now() - t0 }); });
  });
}

const logTail = (dir, n = 15) => {
  try { return fs.readFileSync(path.join(dir, 'bitid.log'), 'utf8').trimEnd().split('\n').slice(-n).join('\n'); } catch (e) { return '(no log)'; }
};

class Diff extends Error {}

// Deep comparison that counts compared scalars and throws Diff at the first difference.
function makeComparer() {
  const c = { count: 0 };
  c.eq = (where, a, b) => {
    if (Array.isArray(a) || Array.isArray(b)) {
      if (!Array.isArray(a) || !Array.isArray(b)) throw new Diff(`${where}: old=${show(a)} new=${show(b)}`);
      if (a.length !== b.length) throw new Diff(`${where}: length old=${a.length} new=${b.length}`);
      for (let i = 0; i < a.length; i++) c.eq(`${where}[${i}]`, a[i], b[i]);
      return;
    }
    if (a && b && typeof a === 'object' && typeof b === 'object') {
      const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
      for (const k of keys) {
        if (!(k in a) || !(k in b)) throw new Diff(`${where}.${k}: old=${k in a ? show(a[k]) : 'missing'} new=${k in b ? show(b[k]) : 'missing'}`);
        c.eq(`${where}.${k}`, a[k], b[k]);
      }
      return;
    }
    c.count++;
    if (!Object.is(a, b) && a !== b) throw new Diff(`${where}: old=${show(a)} new=${show(b)}`);
  };
  return c;
}
const show = (v) => { const s = JSON.stringify(v); return s && s.length > 80 ? s.slice(0, 77) + '...' : String(s); };

function readJson(file, label) {
  if (!fs.existsSync(file)) throw new Diff(`${label}: missing ${file}`);
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

// Compare the outputs of one phase; returns { count, notes }.
function compareRun(oldDir, newDir, phaseLabel) {
  const c = makeComparer();
  const notes = [];

  // metrics.jsonl
  const rows = (d) => fs.readFileSync(path.join(d, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const [ro, rn] = [rows(oldDir), rows(newDir)];
  if (ro.length !== rn.length) throw new Diff(`metrics.jsonl: row count old=${ro.length} new=${rn.length}`);
  const newCols = new Set();
  ro.forEach((a, i) => {
    const b = rn[i];
    const where = `metrics.jsonl row ${i + 1} (generation ${a.generation})`;
    for (const k of Object.keys(a)) {
      if (METRIC_SKIP.has(k)) continue;
      if (!(k in b)) throw new Diff(`${where} field ${k}: old=${show(a[k])} new=missing`);
      c.eq(`${where} field ${k}`, a[k], b[k]);
    }
    for (const k of Object.keys(b)) if (!(k in a)) newCols.add(k);
  });
  if (newCols.size) notes.push(`new metric fields (ignored): ${[...newCols].join(', ')}`);

  // checkpoint.json
  const co = readJson(path.join(oldDir, 'checkpoint.json'), 'checkpoint.json');
  const cn = readJson(path.join(newDir, 'checkpoint.json'), 'checkpoint.json');
  const where = `checkpoint.json (generation ${co.generation})`;
  for (const k of CKPT_FIELDS) {
    if (k === 'coreHash' && CORE_CHANGED) { if (co[k] !== cn[k]) notes.push(`coreHash differs (--core-changed): ${String(co[k]).slice(0, 12)} -> ${String(cn[k]).slice(0, 12)}`); continue; }
    if ((k in co) !== (k in cn)) throw new Diff(`${where} field ${k}: old=${k in co ? show(co[k]) : 'missing'} new=${k in cn ? show(cn[k]) : 'missing'}`);
    if (k in co) c.eq(`${where} field ${k}`, co[k], cn[k]);
  }
  const newConfig = [];
  for (const k of Object.keys(cn.config)) {
    if (RUNTIME_CONFIG.has(k)) continue;
    if (k in co.config) c.eq(`${where} config.${k}`, co.config[k], cn.config[k]);
    else newConfig.push(`${k}=${show(cn.config[k])}`);
  }
  const runtime = [...RUNTIME_CONFIG].filter(k => co.config[k] !== cn.config[k]).map(k => `${k} ${show(co.config[k])} -> ${show(cn.config[k])}`);
  if (runtime.length) notes.push(`runtime config (not compared): ${runtime.join(', ')}`);
  const goneConfig = Object.keys(co.config).filter(k => !(k in cn.config) && !RUNTIME_CONFIG.has(k));
  if (goneConfig.length) throw new Diff(`${where} config: keys missing in new: ${goneConfig.join(', ')}`);
  if (newConfig.length) notes.push(`new config keys (ignored): ${newConfig.join(', ')}`);

  // best.json
  const bo = readJson(path.join(oldDir, 'best.json'), 'best.json');
  const bn = readJson(path.join(newDir, 'best.json'), 'best.json');
  for (const k of new Set([...Object.keys(bo), ...Object.keys(bn)])) {
    if (BEST_SKIP.has(k)) continue;
    if (!(k in bo)) { notes.push(`new best.json field (ignored): ${k}`); continue; }
    if (!(k in bn)) throw new Diff(`best.json field ${k}: old=${show(bo[k])} new=missing`);
    c.eq(`best.json (generation ${bo.generation}) field ${k}`, bo[k], bn[k]);
  }
  if (!CORE_CHANGED && ('coreHash' in bo || 'coreHash' in bn)) c.eq('best.json coreHash', bo.coreHash, bn.coreHash);

  return { count: c.count, notes: notes.map(n => `${phaseLabel}: ${n}`), reason: [co.reason, cn.reason] };
}

async function runScenario(sc, trees, workers) {
  const dirs = trees.map(t => runDir(t, sc.name));
  for (const d of dirs) fs.rmSync(d, { recursive: true, force: true });
  const t0 = Date.now();
  let count = 0;
  const notes = [];
  for (let p = 0; p < sc.phases.length; p++) {
    const label = `phase ${p + 1}`;
    // Old and new run concurrently: results are indexed by job id, so only the excluded timing fields depend on load.
    const res = await Promise.all(trees.map((t, i) => train(t, dirs[i], sc.phases[p], workers)));
    for (let i = 0; i < 2; i++) {
      if (res[i].code !== 0) {
        return { ok: false, ms: Date.now() - t0, msg: `${label}: ${i ? 'new' : 'old'} tree train exited ${res[i].code}\n${logTail(dirs[i])}` };
      }
    }
    try {
      const r = compareRun(dirs[0], dirs[1], label);
      count += r.count;
      notes.push(...r.notes);
      if (p === sc.phases.length - 1 && sc.expectReason) {
        for (let i = 0; i < 2; i++) {
          if (!sc.expectReason.test(r.reason[i] || '')) {
            return { ok: false, ms: Date.now() - t0, msg: `${label}: ${i ? 'new' : 'old'} tree stop reason ${show(r.reason[i])} does not match ${sc.expectReason}` };
          }
        }
        notes.push(`stop reason: ${r.reason[0]}`);
      }
    } catch (e) {
      if (!(e instanceof Diff)) throw e;
      return { ok: false, ms: Date.now() - t0, msg: `first difference (${label}): ${e.message}` };
    }
  }
  return { ok: true, ms: Date.now() - t0, count, notes };
}

async function main() {
  let opts;
  try { opts = parseArgs(process.argv.slice(2)); } catch (e) { console.error(e.message); process.exit(2); }
  CORE_CHANGED = !!opts.coreChanged;
  const [oldTree, newTree] = opts.trees;
  // Only node processes running a trainer CLI (not shells whose command line merely mentions it).
  const busy = (spawnSync('pgrep', ['-fl', 'trainer/cli.js'], { encoding: 'utf8' }).stdout || '')
    .split('\n').filter(l => /^\d+ \S*node\S* .*trainer\/cli\.js/.test(l)).join('\n');
  if (busy && !opts.allowBusy) {
    console.error(`another trainer is running; not starting (pass --allow-busy to run anyway):\n${busy}`);
    process.exit(2);
  }
  console.log(`bit-identity: old=${oldTree}\n              new=${newTree}\n              workers=${opts.workers}`);
  const scenarios = SCENARIOS.filter(s => !opts.only || opts.only.includes(s.name));
  const t0 = Date.now();
  let allOk = true, total = 0;
  for (const sc of scenarios) {
    let r;
    try {
      r = await runScenario(sc, opts.trees, opts.workers);
    } catch (e) {
      r = { ok: false, ms: 0, msg: `harness error: ${e.stack || e.message}` };
    }
    const secs = (r.ms / 1000).toFixed(1);
    if (r.ok) {
      total += r.count;
      console.log(`${sc.name.padEnd(14)} IDENTICAL (${r.count.toLocaleString('en-US')} values)  ${secs}s`);
      for (const n of r.notes) console.log(`${''.padEnd(14)}   ${n}`);
    } else {
      allOk = false;
      const verdict = r.msg.startsWith('first difference') ? 'DIFFERENT' : 'FAILED';
      console.log(`${sc.name.padEnd(14)} ${verdict}  ${secs}s\n${''.padEnd(14)}   ${r.msg.split('\n').join(`\n${''.padEnd(14)}   `)}`);
    }
    if (!opts.keep) for (const t of opts.trees) fs.rmSync(runDir(t, sc.name), { recursive: true, force: true });
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(allOk
    ? `VERDICT: IDENTICAL - ${scenarios.length} scenario(s), ${total.toLocaleString('en-US')} values compared, ${secs}s`
    : `VERDICT: NOT IDENTICAL - see above (${secs}s)`);
  if (opts.keep) console.log(`scratch run dirs kept: <tree>/runs/exp-bitid-<scenario>`);
  process.exit(allOk ? 0 : 1);
}

main();
