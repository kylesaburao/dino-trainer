#!/usr/bin/env node
'use strict';
// Compare two probe.js outputs (research/prereg-x86-determinism.md):
//   node research/scripts/x86-determinism/compare.js <reference.json> <candidate.json>
// Prints the environment and its confounds, a per-stage verdict (IDENTICAL / DIFFERS, with
// the functions, buckets, columns, models and seeds that differ), the pre-registered
// hypothesis the result supports (H0 / H1 / H2) and the job classes (a)-(d) it clears for
// the candidate's machine (prereg §7). `timings` are ignored. Exit code 0 when every compared
// stage is IDENTICAL, 1 when something differs or was refused, 2 on a usage or format error.
const fs = require('fs');
const path = require('path');
const probe = require('./probe.js');

const PACK = 8;
const unpack = (s) => (typeof s === 'string' ? s.match(new RegExp(`.{1,${PACK}}`, 'g')) || [] : []);
const PROBE_FILE = 'research/scripts/x86-determinism/probe.js';
// Files each stage's content depends on (S0 `files`); a mismatch refuses the stage.
const MODELS = Object.values(probe.S5_MODELS);
const DEPS = {
  S1: ['core.js'],
  S2: ['core.js', 'policy.js', 'test/solver.js', 'trainer/bc.js', 'trainer/es.js', 'trainer/env.js', 'model.js', probe.S5_MODELS['v3-bc-duck-d2w32-s1']],
  S3: ['core.js', 'test/solver.js', 'test/golden-format.js'],
  S4: ['core.js', 'policy.js', 'test/solver.js', 'trainer/bc.js'],
  S5: ['core.js', 'policy.js', 'trainer/env.js', 'trainer/wasm-net.js', ...MODELS],
  S6: ['core.js', 'policy.js', 'test/solver.js', 'trainer/bc.js', 'trainer/es.js'],
  S7: ['core.js', 'policy.js', 'test/solver.js', 'trainer/bc.js', 'trainer/bc_train.py', 'trainer/bc_common.py'],
};
const STAGES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'];

// Leaf-level diff: [{ path: [...keys], a, b }]. Arrays and objects are walked; packed hash
// strings are compared as wholes here and unpacked by the per-stage readers.
function diff(a, b, at = [], out = []) {
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) diff(a[k], b[k], [...at, k], out);
  } else if (!(a === b || (Number.isNaN(a) && Number.isNaN(b)))) out.push({ path: at, a, b });
  return out;
}
const isDiscrete = (d) => d.path.includes('discrete');
const packedDiff = (a, b) => { const x = unpack(a), y = unpack(b); return x.map((h, i) => (h !== y[i] ? i : -1)).filter(i => i >= 0); };
const fmtList = (xs, n = 12) => (xs.length > n ? `${xs.slice(0, n).join(', ')}, … (${xs.length})` : xs.join(', '));

// Per-stage readers: lines describing where a stage differs (ds = the stage's diff list).
const READERS = {
  S1(ref, cand, ds) {
    const lines = [];
    const fns = [...new Set(ds.filter(d => d.path[0] === 'fns').map(d => d.path[1]))];
    for (const fn of fns) {
      const spec = probe.S1_FNS[fn];
      const labels = spec ? spec.ranges.flatMap(([label, nb]) => Array(nb).fill(label)) : [];
      const a = ref.fns[fn] && ref.fns[fn].bits, b = cand.fns[fn] && cand.fns[fn].bits;
      if (!a || !b) { lines.push(`${fn}: missing on one side`); continue; }
      const bk = packedDiff(a.buckets, b.buckets);
      const ranges = [...new Set(bk.map(i => labels[i]))];
      lines.push(`${fn}${spec && spec.control ? ' (NEGATIVE CONTROL)' : ''}: ${bk.length}/${unpack(a.buckets).length} buckets differ [${fmtList(bk.map(String))}]` +
        ` in ranges: ${ranges.join('; ') || '-'}${a.edges !== b.edges ? '; edge values differ' : ''}${a.inputs !== b.inputs ? '; INPUTS differ (probe broken)' : ''}`);
    }
    for (const dom of [...new Set(ds.filter(d => d.path[0] === 'domains').map(d => d.path[1]))]) {
      const bk = packedDiff(ref.domains[dom].bits.buckets, cand.domains[dom].bits.buckets);
      const span = dom === 'adam-pow' ? (i) => `t ${i * 10000 + 1}-${(i + 1) * 10000}` : (i) => `score ${i * 10000}-${Math.min(1000000, i * 10000 + 9999)}`;
      lines.push(`${dom}: ${bk.length} buckets differ: ${fmtList(bk.map(span), 8)}`);
    }
    if (ds.some(d => d.path[0] === 'selfTest')) lines.push('selfTest differs');
    return lines;
  },
  S2(ref, cand, ds) {
    const by = {};
    for (const d of ds) { const k = `${d.path[1]}`; (by[k] = by[k] || new Set()).add(`${d.path[2]}${isDiscrete(d) ? '(calls)' : ''}`); }
    return Object.entries(by).map(([act, fns]) => `[${act}] ${[...fns].join(', ')}`);
  },
  S3(ref, cand, ds) {
    const arms = [...new Set(ds.map(d => d.path[1]))];
    return arms.map(arm => {
      const a = ref.arms[arm], b = cand.arms[arm];
      if (!a || !b) return `${arm}: missing on one side`;
      const disc = ds.some(d => d.path[1] === arm && isDiscrete(d));
      const ck = packedDiff(a.bits.checkpoints, b.bits.checkpoints);
      const first = ck.length ? `first differing checkpoint ${ck[0]} (frames ${ck[0] * ref.checkpointEvery + 1}-${(ck[0] + 1) * ref.checkpointEvery})` : 'checkpoints equal (difference after the last checkpoint)';
      return `${arm}: ${disc ? 'DISCRETE (frames/score/crash/rng/obstacles) and bits' : 'bits only'}; ${first}`;
    });
  },
  S4(ref, cand, ds) {
    const lines = [];
    for (const set of [...new Set(ds.map(d => d.path[1]))]) {
      const a = ref.sets[set], b = cand.sets[set];
      if (!a || !b) { lines.push(`${set}: missing on one side`); continue; }
      const disc = ds.filter(d => d.path[1] === set && isDiscrete(d)).map(d => d.path.slice(3).join('.'));
      const cols = packedDiff(a.bits.features, b.bits.features).map(i => probe.COLUMN_NAMES[i]);
      const levels = packedDiff(a.bits.levels, b.bits.levels).map(i => (probe.DEMO_SETS[set] ? probe.DEMO_SETS[set].levels[i] : i));
      lines.push(`${set}: sha256 ${a.bits.sha256.slice(0, 8)} -> ${b.bits.sha256.slice(0, 8)} (candidate ${b.matchesManifest ? 'matches' : 'does not match'} the committed manifest);` +
        ` discrete ${disc.length ? `DIFFERS: ${fmtList(disc)}` : 'identical (rows, labels, weights, episodes)'}; feature columns differing: ${fmtList(cols) || 'none'}; start levels differing: ${fmtList(levels.map(String)) || 'none'}`);
    }
    return lines;
  },
  S5(ref, cand, ds) {
    const lines = [];
    for (const m of [...new Set(ds.map(d => d.path[1]))]) {
      for (const arm of [...new Set(ds.filter(d => d.path[1] === m).map(d => d.path[2]))]) {
        const sub = ds.filter(d => d.path[1] === m && d.path[2] === arm);
        const seeds = sub.filter(d => d.path[4] === 'scores').map(d => `#${d.path[5]} ${d.a}->${d.b}`);
        const what = sub.map(d => d.path.slice(3).filter(k => !/^\d+$/.test(k)).join('.'));
        lines.push(`${m} ${arm}: ${fmtList([...new Set(what)])}${seeds.length ? `; scores differ at seed index ${fmtList(seeds)}` : ''}`);
      }
    }
    return lines;
  },
  S6(ref, cand, ds) { return [fmtList([...new Set(ds.map(d => d.path.filter(k => !/^\d+$/.test(k)).join('.')))], 20)]; },
  S7(ref, cand, ds) { return [fmtList([...new Set(ds.map(d => d.path.join('.')))], 20)]; },
};

// Keys that are derived from other compared values or describe the environment, not results.
const INFO = (stage, d) => d.path.some(k => k === 'matchesManifest' || k === 'matches' || k === 'jsWasmAgree') || (stage === 'S7' && d.path[0] === 'env');

function main() {
  const [refFile, candFile] = process.argv.slice(2);
  if (!refFile || !candFile) { console.error('usage: node compare.js <reference.json> <candidate.json>'); process.exit(2); }
  const load = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
  const R = load(refFile), K = load(candFile);
  if (R.format !== probe.FORMAT || K.format !== probe.FORMAT) { console.error(`format must be ${probe.FORMAT}: ${R.format} / ${K.format}`); process.exit(2); }
  const e0 = R.stages.S0 || {}, e1 = K.stages.S0 || {};
  console.log(`reference: ${path.basename(refFile)}  ${e0.platform} ${e0.arch} node ${e0.node} V8 ${e0.v8} clang=${e0.build && e0.build.clang} ${e0.cpu} head ${String(e0.gitHead).slice(0, 10)}${e0.gitDirty ? ' (dirty)' : ''}`);
  console.log(`candidate: ${path.basename(candFile)}  ${e1.platform} ${e1.arch} node ${e1.node} V8 ${e1.v8} clang=${e1.build && e1.build.clang} ${e1.cpu} head ${String(e1.gitHead).slice(0, 10)}${e1.gitDirty ? ' (dirty)' : ''}`);
  const confound = e0.node !== e1.node || e0.v8 !== e1.v8;
  const sameArch = e0.arch === e1.arch && e0.platform === e1.platform;
  if (confound) console.log(`CONFOUND: node/V8 differ (${e0.node}/${e0.v8} vs ${e1.node}/${e1.v8}); a difference cannot be attributed to the architecture.`);
  if (sameArch) console.log('note: same platform and arch on both sides (a run-to-run or same-machine control).');
  const badFiles = Object.keys({ ...(e0.files || {}), ...(e1.files || {}) }).filter(f => (e0.files || {})[f] !== (e1.files || {})[f]);
  if (e0.coreHash !== e1.coreHash) console.log(`coreHash differs: ${String(e0.coreHash).slice(0, 12)} vs ${String(e1.coreHash).slice(0, 12)}`);
  if (badFiles.length) console.log(`files differ: ${badFiles.join(', ')}`);
  const probeDiffers = badFiles.includes(PROBE_FILE);

  const verdict = {};
  console.log('');
  for (const id of STAGES) {
    const a = R.stages[id], b = K.stages[id];
    const refused = probeDiffers ? [PROBE_FILE] : DEPS[id].filter(f => badFiles.includes(f));
    if (!a || !b) { verdict[id] = 'ABSENT'; console.log(`${id}: ABSENT on ${!a ? 'reference' : 'candidate'}`); continue; }
    if (a.error || b.error) { verdict[id] = 'ERROR'; console.log(`${id}: ERROR ${a.error ? `reference: ${a.error}` : ''} ${b.error ? `candidate: ${b.error}` : ''}`); continue; }
    if (a.skipped || b.skipped) { verdict[id] = 'SKIPPED'; console.log(`${id}: SKIPPED (${a.skipped ? `reference: ${a.skipped}` : ''}${a.skipped && b.skipped ? '; ' : ''}${b.skipped ? `candidate: ${b.skipped}` : ''})`); continue; }
    if (refused.length) { verdict[id] = 'REFUSED'; console.log(`${id}: REFUSED, input files differ: ${refused.join(', ')}`); continue; }
    const ds = diff(a, b).filter(d => !INFO(id, d));
    const disc = ds.filter(isDiscrete);
    verdict[id] = !ds.length ? 'IDENTICAL' : disc.length ? 'DIFFERS (discrete)' : 'DIFFERS (bits only)';
    verdict[`${id}.diffs`] = ds;
    console.log(`${id}: ${verdict[id]}`);
    if (ds.length) for (const line of READERS[id](a, b, ds)) console.log(`    ${line}`);
  }
  // Controls.
  const s1a = R.stages.S1, s1b = K.stages.S1;
  const controlsBad = verdict.S1 && verdict.S1.startsWith('DIFFERS') && (verdict['S1.diffs'] || []).some(d => d.path[0] === 'fns' && probe.S1_FNS[d.path[1]] && probe.S1_FNS[d.path[1]].control);
  const selfTestOk = s1a && s1b && s1a.selfTest && s1b.selfTest && s1a.selfTest.pass && s1b.selfTest.pass;
  console.log('');
  console.log(`controls: negative (sqrt, fround, arithmetic) ${controlsBad ? 'FAILED' : verdict.S1 && !['ERROR', 'ABSENT', 'REFUSED'].includes(verdict.S1) ? 'pass' : 'not read'}; 1-ulp self-test ${selfTestOk ? 'pass' : 'FAILED or missing'} on both`);
  if (controlsBad || !selfTestOk) {
    console.log('HYPOTHESIS: none read: a control failed, so the probe or the harness is broken (prereg §5).');
    process.exit(1);
  }

  // Hypothesis (prereg §4) and job classes (prereg §7).
  const is = (id, ...vs) => vs.includes(verdict[id]);
  const decisive = ['S3', 'S4', 'S5'];
  const read = STAGES.filter(id => verdict[id] && verdict[id] !== 'SKIPPED');
  const unread = STAGES.filter(id => id !== 'S7' && !['IDENTICAL', 'DIFFERS (discrete)', 'DIFFERS (bits only)'].includes(verdict[id]));
  const anyDiff = read.some(id => verdict[id].startsWith('DIFFERS'));
  const discrete = decisive.some(id => is(id, 'DIFFERS (discrete)'));
  const compared = read.filter(id => verdict[id] === 'IDENTICAL' || verdict[id].startsWith('DIFFERS'));
  const h = !compared.length ? 'not read (no stage could be compared)'
    : !anyDiff ? 'H0 (bit-identical in every compared stage)'
    : discrete ? `H2 (differences reach the game or decisions: ${decisive.filter(id => is(id, 'DIFFERS (discrete)')).join(', ')})`
      : 'H1 (libm last bits only: no rows, labels, scores, actions or trajectories differ)';
  console.log(`HYPOTHESIS: ${h}${unread.length ? `; incomplete, not read: ${unread.join(', ')}` : ''}${confound && anyDiff ? '; UNATTRIBUTED to the architecture (node/V8 differ)' : ''}`);
  if (sameArch) console.log('  (same arch on both sides: this is the run-to-run / same-machine reading, not a cross-architecture one)');
  // Anomalies: results the premises say cannot happen (prereg §3); each needs a follow-up.
  const s1d = verdict['S1.diffs'] || [];
  const domainsDiffer = s1d.some(d => d.path[0] === 'domains' && d.path[1] !== 'adam-pow');
  const tanhDiffers = s1d.some(d => d.path[0] === 'fns' && d.path[1] === 'tanh');
  if (is('S1', 'IDENTICAL', 'DIFFERS (bits only)')) {
    for (const id of ['S3', 'S4']) if (verdict[id] && verdict[id].startsWith('DIFFERS') && !domainsDiffer) console.log(`ANOMALY: ${id} differs while the exhaustive rules tables are identical (contradicts premise P1: the game's only libm use is the rules table)`);
    if (verdict.S5 && verdict.S5.startsWith('DIFFERS') && !domainsDiffer && !tanhDiffers && is('S3', 'IDENTICAL')) console.log('ANOMALY: S5 differs while the rules tables, tanh and S3 are identical (look at the JIT dot product; compare JS and wasm)');
  }
  const S5c = K.stages.S5;
  if (S5c && S5c.models) for (const [m, r] of Object.entries(S5c.models)) if (r.jsWasmAgree === false) console.log(`ANOMALY: candidate ${m}: the JS net and the wasm kernel chose different actions (wasm-net.js claims bit-identity)`);

  const ok = (...ids) => ids.every(id => is(id, 'IDENTICAL'));
  const discreteOk = (...ids) => ids.every(id => is(id, 'IDENTICAL', 'DIFFERS (bits only)'));
  const notRead = (...ids) => ids.some(id => !is(id, 'IDENTICAL', 'DIFFERS (bits only)', 'DIFFERS (discrete)'));
  const s6bc = () => { const ds = verdict['S6.diffs'] || []; return is('S6', 'IDENTICAL') || !ds.some(d => d.path[0] === 'bc'); };
  const s6es = () => { const ds = verdict['S6.diffs'] || []; return is('S6', 'IDENTICAL') || !ds.some(d => d.path[0] === 'es'); };
  // The S1 functions and tables each job class runs (prereg §3 P1, P4): a difference there
  // means the job's arithmetic differs on some inputs even if the probed job outputs matched.
  const S1_USES = {
    a: ['rules-v2', 'rules-v3'],
    b: ['rules-v2', 'rules-v3', 'tanh'],
    c: ['tanh', 'exp', 'log', 'sin', 'cos', 'sqrt', 'pow', 'adam-pow'],
    d: ['rules-v2', 'rules-v3', 'tanh', 'log', 'sin', 'cos', 'sqrt', 'pow', 'adam-pow'],
  };
  const s1Hits = (k) => [...new Set((verdict['S1.diffs'] || []).filter(d => (d.path[0] === 'fns' || d.path[0] === 'domains') && S1_USES[k].includes(d.path[1])).map(d => d.path[1]))];
  const s1Note = (k) => { const h = s1Hits(k); return h.length ? `; S1 differs in ${h.join(', ')}, which this job runs` : ''; };
  const tag = (v) => (confound && !v.startsWith('CLEARED') && !v.startsWith('NOT ASSESSED') ? `${v} [unattributed: node/V8 differ; rerun with the reference's node]` : v);
  const COND = {
    a: 'CONDITIONAL: valid demos, not byte-identical; never under a Mac manifest sha256; record the machine',
    b: 'CONDITIONAL: same scores in the probed horizon; keep every arm of a paired comparison on one architecture; final certification on the Mac',
    c: 'CONDITIONAL: valid replicates, not reproductions; the repo node pins may fail there; record the machine',
    d: 'CONDITIONAL: valid independent runs; no cross-architecture resume; bit-identity.js on one machine only',
  };
  const classes = {
    '(a) bc-gen': notRead('S1', 'S4') ? 'NOT ASSESSED' : ok('S4') && !s1Hits('a').length ? 'CLEARED, bitwise'
      : discreteOk('S4') ? COND.a + s1Note('a') : 'NOT CLEARED',
    '(b) assess / scans': notRead('S1', 'S3', 'S5') ? 'NOT ASSESSED' : ok('S3', 'S5') && !s1Hits('b').length ? 'CLEARED, bitwise within the probed horizons (see prereg §6)'
      : discreteOk('S3', 'S5') ? COND.b + s1Note('b') : 'NOT CLEARED',
    '(c) BC node training': notRead('S1', 'S6') ? 'NOT ASSESSED' : s6bc() && !s1Hits('c').length ? 'CLEARED, bitwise (node pins reproduce)'
      : COND.c + s1Note('c') + (is('S4', 'DIFFERS (discrete)') ? '; train only on Mac-made demos' : ''),
    '(d) ES training': notRead('S1', 'S3', 'S5', 'S6') ? 'NOT ASSESSED' : s6es() && ok('S3', 'S5') && !s1Hits('d').length ? 'CLEARED, bitwise (a Mac checkpoint may resume there)'
      : discreteOk('S3', 'S5') ? COND.d + s1Note('d') : 'TRAIN-ONLY: assess and certify candidates on the Mac',
  };
  console.log('JOB CLASSES for the candidate machine (prereg §7):');
  for (const [k, v] of Object.entries(classes)) console.log(`  ${k}: ${tag(v)}`);
  if (is('S7', 'IDENTICAL', 'DIFFERS (bits only)', 'DIFFERS (discrete)')) console.log(`  (python backend, reported only, not cleared: S7 ${verdict.S7})`);
  process.exit(anyDiff || unread.length ? 1 : 0);
}

main();
