'use strict';
// Serial job runner for research/prereg-distill.md (§11b "Execution"): one heavy job at a time.
// Usage: node runjobs.js <jobs.json> <status.log>
// jobs.json: [{ name, argv: ["node", ...] | ["python", ...], log, done, cond? }]
//   argv[0] "node" runs this node binary; "python" runs $PYTHON.
//   done: a file whose existence means the job already finished (skipped; makes a rerun resume).
//   cond (optional, checked just before the job):
//     { tierAMin: N, best: <best.json> }        run only if validation.count >= N;
//     { openingPassed: N, file: <opening.json> } run only if passed === N.
// Stops cleanly (after the current job) if the file PAUSE exists next to jobs.json.
// Each job's stdout+stderr go to its log; status.log gets one line per job (start, end, exit).
// The environment is inherited, plus OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 (prereg §5).
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const [jobsFile, statusFile] = process.argv.slice(2);
const jobs = JSON.parse(fs.readFileSync(jobsFile, 'utf8'));
const pauseFile = path.join(path.dirname(path.resolve(jobsFile)), 'PAUSE');
const env = { ...process.env, OMP_NUM_THREADS: '2', VECLIB_MAXIMUM_THREADS: '2' };
const status = (s) => fs.appendFileSync(statusFile, `[${new Date().toISOString()}] ${s}\n`);
status(`runner pid ${process.pid}: ${jobs.length} jobs from ${jobsFile}`);
for (const j of jobs) {
  if (fs.existsSync(pauseFile)) { status(`PAUSE file present; stopping before ${j.name}`); process.exit(0); }
  if (j.done && fs.existsSync(j.done)) { status(`skip ${j.name} (done)`); continue; }
  if (j.cond) {
    if (j.cond.tierAMin !== undefined) {
      const c = JSON.parse(fs.readFileSync(j.cond.best, 'utf8')).validation.count;
      if (c < j.cond.tierAMin) { status(`skip ${j.name} (tier A ${c} < ${j.cond.tierAMin})`); continue; }
    }
    if (j.cond.openingPassed !== undefined) {
      const p = JSON.parse(fs.readFileSync(j.cond.file, 'utf8')).passed;
      if (p !== j.cond.openingPassed) { status(`skip ${j.name} (opening ${p})`); continue; }
    }
  }
  const bin = j.argv[0] === 'node' ? process.execPath : j.argv[0] === 'python' ? env.PYTHON : j.argv[0];
  fs.mkdirSync(path.dirname(j.log), { recursive: true });
  const fd = fs.openSync(j.log, 'a');
  const t0 = Date.now();
  status(`start ${j.name}`);
  const r = spawnSync(bin, j.argv.slice(1), { stdio: ['ignore', fd, fd], env });
  fs.closeSync(fd);
  status(`end ${j.name} exit ${r.status} ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
status('runner done');
