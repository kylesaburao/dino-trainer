#!/usr/bin/env node
'use strict';
// Mutation control for compare.js (research/prereg-x86-determinism.md §5):
//   node research/scripts/x86-determinism/compare-selftest.js [reference.json]
// Copies the reference (default mac-reference.json), applies one known mutation per case in
// a scratch dir under os.tmpdir(), runs compare.js on (reference, mutant) and checks that
// the expected verdict lines appear. Prints PASS/FAIL per case; exit 1 on any FAIL.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const refFile = path.resolve(process.argv[2] || path.join(__dirname, 'mac-reference.json'));
const ref = JSON.parse(fs.readFileSync(refFile, 'utf8'));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-x86-compare-'));
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const flip = (s, i) => s.slice(0, i) + (s[i] === '0' ? '1' : '0') + s.slice(i + 1);

// [name, mutate(result), lines that must appear in compare.js's output]
const CASES = [
  ['unmodified copy', () => {}, ['HYPOTHESIS: H0', '(a) bc-gen: CLEARED', '(d) ES training: CLEARED']],
  ['tanh bucket 7', (r) => { const b = r.stages.S1.fns.tanh.bits; b.buckets = flip(b.buckets, 7 * 8); b.all = flip(b.all, 0); },
    ['S1: DIFFERS (bits only)', 'tanh: 1/32 buckets differ [7] in ranges: pre-activation u[-30,30]', 'HYPOTHESIS: H1', '(a) bc-gen: CLEARED', '(c) BC node training: CONDITIONAL']],
  ['rules-v2 bucket 30', (r) => { const b = r.stages.S1.domains['rules-v2'].bits; b.buckets = flip(b.buckets, 30 * 8); b.all = flip(b.all, 0); },
    ['rules-v2: 1 buckets differ: score 300000-309999', '(a) bc-gen: CONDITIONAL']],
  ['v3 demos speed column', (r) => { const s = r.stages.S4.sets['v3-demos-duck-s1-40']; s.bits.features = flip(s.bits.features, 4 * 8); s.bits.sha256 = flip(s.bits.sha256, 0); s.matchesManifest = false; },
    ['S4: DIFFERS (bits only)', 'feature columns differing: speed', 'ANOMALY: S4 differs', '(a) bc-gen: CONDITIONAL']],
  ['v2 jump demos label column', (r) => { const s = r.stages.S4.sets['v2-demos-jump-s1-40']; s.discrete.label = flip(s.discrete.label, 0); },
    ['S4: DIFFERS (discrete)', 'HYPOTHESIS: H2', '(a) bc-gen: NOT CLEARED']],
  ['root model score, seed 3', (r) => { r.stages.S5.models['root-model.js'].js.discrete.scores[3] = 12345; },
    ['S5: DIFFERS (discrete)', 'scores differ at seed index #3 20000->12345', 'HYPOTHESIS: H2', '(b) assess / scans: NOT CLEARED', '(d) ES training: TRAIN-ONLY']],
  ['S3 checkpoint 3', (r) => { const b = r.stages.S3.arms['v2-duck-148000-s1'].bits; b.checkpoints = flip(b.checkpoints, 3 * 8); b.hash = flip(b.hash, 0); },
    ['v2-duck-148000-s1: bits only; first differing checkpoint 3 (frames 3001-4000)', '(b) assess / scans: CONDITIONAL']],
  ['node pin', (r) => { r.stages.S6.bc.jump.bits.pin = flip(r.stages.S6.bc.jump.bits.pin, 0); },
    ['S6: DIFFERS (bits only)', '(c) BC node training: CONDITIONAL', '(d) ES training: CLEARED']],
  ['node version + tanh', (r) => { r.stages.S0.node = 'v0.0.0'; r.stages.S1.fns.tanh.bits.all = flip(r.stages.S1.fns.tanh.bits.all, 0); },
    ['CONFOUND: node/V8 differ', 'UNATTRIBUTED', '[unattributed: node/V8 differ']],
  ['sqrt (negative control)', (r) => { r.stages.S1.fns.sqrt.bits.all = flip(r.stages.S1.fns.sqrt.bits.all, 0); },
    ['negative (sqrt, fround, arithmetic) FAILED', 'HYPOTHESIS: none read']],
  ['core.js hash', (r) => { r.stages.S0.files['core.js'] = flip(r.stages.S0.files['core.js'], 0); },
    ['S4: REFUSED, input files differ: core.js', 'HYPOTHESIS: not read', '(a) bc-gen: NOT ASSESSED']],
];

let fails = 0;
for (const [name, mutate, want] of CASES) {
  const r = JSON.parse(JSON.stringify(ref));
  mutate(r);
  const f = path.join(tmp, 'mutant.json');
  fs.writeFileSync(f, JSON.stringify(r));
  let out;
  try { out = execFileSync(process.execPath, [path.join(__dirname, 'compare.js'), refFile, f], { encoding: 'utf8' }); } catch (e) { out = e.stdout || String(e); }
  const missing = want.filter(w => !out.includes(w));
  if (missing.length) fails++;
  console.log(`${missing.length ? 'FAIL' : 'PASS'}  ${name}${missing.length ? `\n      missing: ${missing.join(' | ')}` : ''}`);
}
console.log(fails ? `${fails} case(s) failed` : `all ${CASES.length} cases pass`);
process.exit(fails ? 1 : 0);
