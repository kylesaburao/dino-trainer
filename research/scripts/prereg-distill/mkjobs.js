'use strict';
// Builds runjobs.js job lists for research/prereg-distill.md §11b (paths of ruling (c)).
// Usage (from the checkout root):
//   node mkjobs.js ctrl   <out.json>                          C-direct-params, C-kd-control (+ opening)
//   node mkjobs.js stage0 <out.json> <shape,shape..> [seeds]    direct clones: train, opening, duck share
//   node mkjobs.js d1     <out.json> <shape,shape..> [seeds] [arms]  distilled arms: train, opening, duck share
//   node mkjobs.js tierb  <out.json> <name> [<name> ...]       tier-B scans + margin of the named runs
// shape = DxW (e.g. 1x6 = --hidden 6, 2x4 = --hidden 4,4); seeds default 1-10;
// arms default tau1,tau2,tau4,hard,tau2-a05.
const fs = require('fs');
const G = 'runs/exp-distill';
const DEMOS = `${G}/v2-demos-duck-s1-40`;
const TEACHER = 'runs/exp-bc-duck/v2-bc-duck-d2w32-s1/best.json';
const ARMS = {
  tau1: ['--distill-from', TEACHER, '--tau', '1'],
  tau2: ['--distill-from', TEACHER, '--tau', '2'],
  tau4: ['--distill-from', TEACHER, '--tau', '4'],
  hard: ['--distill-from', TEACHER, '--distill-mode', 'hard'],
  'tau2-a05': ['--distill-from', TEACHER, '--tau', '2', '--alpha', '0.5'],
};
const [cmd, out, ...rest] = process.argv.slice(2);
const seedList = (s) => { const [a, b] = (s || '1-10').split('-').map(Number); return Array.from({ length: (b || a) - a + 1 }, (_, i) => a + i); };
const hidden = (shape) => { const [d, w] = shape.split('x').map(Number); return Array(d).fill(w).join(','); };
const tag = (shape) => { const [d, w] = shape.split('x'); return `d${d}w${w}`; };

function train(name, shape, seed, extra) {
  const dir = `${G}/${name}`;
  fs.mkdirSync(dir, { recursive: true });
  return { name: `train ${name}`, log: `${dir}/train.log`, done: `${dir}/best.json`,
    argv: ['node', 'trainer/cli.js', 'bc-train', '--backend', 'python', '--hidden', hidden(shape), '--seed', String(seed),
      '--demos', DEMOS, '--snapshots', '--wasm', '--dir', dir, ...extra] };
}
const opening = (name) => ({ name: `opening ${name}`, log: `${G}/opening/${name}.log`, done: `${G}/opening/${name}.json`,
  argv: ['node', 'research/scripts/prereg-teacher-duck-stuck/opening.js', '--model', `${G}/${name}/best.json`, '--out', `${G}/opening/${name}.json`] });
const duck = (name) => ({ name: `duckshare ${name}`, log: `${G}/duckshare/${name}.log`, done: `${G}/duckshare/${name}.json`,
  cond: { tierAMin: 20, best: `${G}/${name}/best.json` },
  argv: ['node', 'research/scripts/prereg-teacher-duck-step2/duckshare.js', '--model', `${G}/${name}/best.json`, '--base', '30003000', '--n', '10', '--out', `${G}/duckshare/${name}.json`] });
const per = (name, shape, seed, extra) => [train(name, shape, seed, extra), opening(name), duck(name)];

const jobs = [];
if (cmd === 'ctrl') {
  jobs.push(train('v2-bc-duck-d2w8-s1-ctrl', '2x8', 1, []));
  jobs.push(train('v2-distill-duck-d2w8-s1-tau2', '2x8', 1, ARMS.tau2), opening('v2-distill-duck-d2w8-s1-tau2'));
} else if (cmd === 'stage0') {
  for (const shape of rest[0].split(',')) for (const s of seedList(rest[1])) jobs.push(...per(`v2-bc-duck-${tag(shape)}-s${s}`, shape, s, []));
} else if (cmd === 'd1') {
  const arms = (rest[2] || Object.keys(ARMS).join(',')).split(',');
  for (const shape of rest[0].split(',')) for (const a of arms) for (const s of seedList(rest[1])) {
    if (!ARMS[a]) throw new Error(`unknown arm ${a}`);
    jobs.push(...per(`v2-distill-duck-${tag(shape)}-s${s}-${a}`, shape, s, ARMS[a]));
  }
} else if (cmd === 'tierb') {
  for (const name of rest) {
    jobs.push({ name: `tierB ${name}`, log: `${G}/scans/${name}.log`, done: null,
      argv: ['node', 'research/scripts/prereg-bc-robustness/scan.js', '--model', `${G}/${name}/best.json`, '--out', `${G}/scans/${name}.jsonl`,
        '--base', '30003000', '--n', '1000', '--target', '1000000', '--wasm'] });
    jobs.push({ name: `margin ${name}`, log: `${G}/margin/${name}.log`, done: `${G}/margin/${name}.json`,
      argv: ['node', 'research/scripts/prereg-bc-sweep/margin.js', '--model', `${G}/${name}/best.json`, '--out', `${G}/margin/${name}.json`] });
  }
} else throw new Error(`unknown command ${cmd}`);
for (const d of ['opening', 'duckshare', 'scans', 'margin', 'agree']) fs.mkdirSync(`${G}/${d}`, { recursive: true });
fs.writeFileSync(out, JSON.stringify(jobs, null, 1));
console.log(`${jobs.length} jobs -> ${out}`);
