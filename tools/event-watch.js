'use strict';
// Stop a training run at the validation event: two consecutive validations with
// valMedian >= threshold. Polls <dir>/metrics.jsonl and sends SIGINT to the PID in
// <dir>/train.lock (the trainer saves a checkpoint and releases the lock on SIGINT).
// Exits when the event fires or when the trainer is gone.
// Usage: node tools/event-watch.js <run dir> [--threshold 2000] [--after GEN] [--poll-ms 1000]
//   --after GEN  only count validations at generations > GEN (the resume point)
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? Number(args[i + 1]) : def; };
const dir = args[0];
const THRESHOLD = opt('--threshold', 2000), AFTER = opt('--after', -Infinity), POLL = opt('--poll-ms', 1000);
const log = (m) => console.log(`[${new Date().toISOString()}] event-watch ${path.basename(dir)}: ${m}`);

const pidOf = () => {
  try { return Number(JSON.parse(fs.readFileSync(path.join(dir, 'train.lock'), 'utf8')).pid); } catch (e) { /* not JSON */ }
  try { return Number(fs.readFileSync(path.join(dir, 'train.lock'), 'utf8').trim()); } catch (e) { return null; }
};
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch (e) { return false; } };

let seenPid = null;
const timer = setInterval(() => {
  const pid = pidOf();
  if (pid) seenPid = pid;
  let rows = [];
  try {
    rows = fs.readFileSync(path.join(dir, 'metrics.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  } catch (e) { /* not written yet */ }
  const vals = rows.filter(r => r.valMedian != null && r.generation > AFTER);
  for (let i = 1; i < vals.length; i++) {
    if (vals[i - 1].valMedian >= THRESHOLD && vals[i].valMedian >= THRESHOLD) {
      log(`event at generation ${vals[i - 1].generation} (confirmed at ${vals[i].generation}: ${vals[i - 1].valMedian}, ${vals[i].valMedian})`);
      if (seenPid && alive(seenPid)) { process.kill(seenPid, 'SIGINT'); log(`SIGINT sent to ${seenPid}`); }
      clearInterval(timer);
      return;
    }
  }
  if (seenPid && !alive(seenPid)) { log('trainer exited without the event'); clearInterval(timer); }
}, POLL);
