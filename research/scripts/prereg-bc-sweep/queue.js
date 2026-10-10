'use strict';
// Run sweep training jobs (train-one.sh) from a list, at most --parallel at a time, each
// with its log in its run dir; waits on the child PIDs it spawned. Skips a job whose
// best.json already exists. Usage:
//   node queue.js --parallel 9 --workers 4 d1-w8-s1 d2-w8-s1 ...
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const ROOT = '/Users/kyle/Desktop/dino-game';
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args.splice(i, 2)[1] : def; };
const PARALLEL = Number(opt('--parallel', 1));
const WORKERS = Number(opt('--workers', 4));
const jobs = args.filter(a => /^d\d+-w\d+-s\d+$/.test(a));
const ts = () => new Date().toISOString();

let running = 0, next = 0, failed = 0;
function launch() {
  while (running < PARALLEL && next < jobs.length) {
    const job = jobs[next++];
    const [, d, w, s] = job.match(/^d(\d+)-w(\d+)-s(\d+)$/);
    const dir = path.join(ROOT, 'runs', 'exp-bc-sweep', job);
    if (fs.existsSync(path.join(dir, 'best.json'))) { console.log(`${ts()} skip ${job} (best.json exists)`); continue; }
    fs.mkdirSync(dir, { recursive: true });
    const log = fs.openSync(path.join(dir, 'train.log'), 'a');
    const child = spawn(path.join(__dirname, 'train-one.sh'), [d, w, s, String(WORKERS)], { stdio: ['ignore', log, log] });
    running++;
    console.log(`${ts()} start ${job} pid ${child.pid}`);
    child.on('exit', (code) => {
      running--;
      if (code !== 0) failed++;
      console.log(`${ts()} end ${job} code ${code}`);
      launch();
      if (running === 0 && next >= jobs.length) console.log(`${ts()} queue done, ${failed} failed`);
    });
  }
}
launch();
