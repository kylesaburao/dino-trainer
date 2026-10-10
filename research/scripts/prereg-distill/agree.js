'use strict';
// Held-out argmax agreement of trained models with the SOLVER label and with a TEACHER model's
// argmax (research/prereg-distill.md §5.3 item 1, §11b (b) C-kd-control).
// Loads the held-out demos (bc-gen seeds 41-60), computes each model's logits with
// trainer/bc.js teacherLogits (policy.createNet's summation order; argmax first index wins
// ties, as policy.js) and reports, per model:
//   solver:  sum(w * [argmax == label]) / sum(w)  (weighted, as agreement.py), and unweighted;
//   teacher: sum(w * [argmax == argmax(teacher)]) / sum(w), and unweighted;
//   recallDuck (weighted, vs the solver).
// Usage: node agree.js --demos <dir> --teacher <best.json> --out <f.json> <model best.json> [...]
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '../../..');
const bc = require(path.join(ROOT, 'trainer', 'bc.js'));

const args = process.argv.slice(2);
const take = (name) => { const i = args.indexOf(name); if (i < 0) return null; const v = args[i + 1]; args.splice(i, 2); return v; };
const demosDir = take('--demos'), teacherFile = take('--teacher'), outFile = take('--out');
if (!demosDir || !teacherFile || !args.length) throw new Error('usage: agree.js --demos <dir> --teacher <best.json> [--out f.json] <model> [...]');
const { binFile, manifest } = bc.resolveDemos(demosDir);
const data = bc.loadDemos(binFile);
const { y, sw, n } = data;
const load = f => JSON.parse(fs.readFileSync(path.resolve(f), 'utf8'));
const tLab = bc.argmaxLabels(bc.teacherLogits(load(teacherFile), data));
let W = 0; for (let i = 0; i < n; i++) W += sw[i];
function score(lab) {
  let ws = 0, us = 0, wt = 0, ut = 0, wd = 0, wdh = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] === y[i]) { ws += sw[i]; us++; }
    if (lab[i] === tLab[i]) { wt += sw[i]; ut++; }
    if (y[i] === 2) { wd += sw[i]; if (lab[i] === 2) wdh += sw[i]; }
  }
  return { solver: ws / W, solverUnweighted: us / n, teacher: wt / W, teacherUnweighted: ut / n, recallDuck: wd ? wdh / wd : null };
}
const out = { demos: demosDir, demosSha256: manifest.sha256, rows: n, teacher: teacherFile, teacherVsSolver: score(tLab), models: [] };
for (const f of args) {
  const r = { model: f, ...score(bc.argmaxLabels(bc.teacherLogits(load(f), data))) };
  out.models.push(r);
  console.log(`${f} solver ${r.solver.toFixed(5)} teacher ${r.teacher.toFixed(5)} recallDuck ${r.recallDuck.toFixed(5)}`);
}
console.log(`teacher vs solver ${out.teacherVsSolver.solver.toFixed(5)} (unweighted ${out.teacherVsSolver.solverUnweighted.toFixed(5)})`);
if (outFile) fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
