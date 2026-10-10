'use strict';
// A fixed pool of worker_threads with a dynamic job queue.
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');

const defaultWorkers = () => (typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length);

function createPool(size) {
  const workers = Array.from({ length: size }, () => new Worker(path.join(__dirname, 'worker.js')));
  let failed = null;
  for (const w of workers) w.on('error', (e) => { failed = e; });

  // Run all jobs under one shared setup; resolves to results in job order.
  function run(setup, jobs) {
    return new Promise((resolve, reject) => {
      const results = new Array(jobs.length);
      let next = 0, done = 0;
      if (jobs.length === 0) return resolve(results);
      const cleanups = [];
      const finish = (err) => { cleanups.forEach(f => f()); err ? reject(err) : resolve(results); };
      for (const w of workers) {
        w.postMessage({ type: 'setup', setup });
        const onMsg = (msg) => {
          if (msg.type !== 'result') return;
          results[msg.result.id] = msg.result;
          done++;
          if (done === jobs.length) return finish();
          if (next < jobs.length) w.postMessage({ type: 'job', job: { ...jobs[next], id: next++ } });
        };
        const onErr = (e) => finish(e);
        const onExit = (code) => finish(new Error(`worker exited (${code})`));
        w.on('message', onMsg);
        w.on('error', onErr);
        w.on('exit', onExit);
        cleanups.push(() => { w.off('message', onMsg); w.off('error', onErr); w.off('exit', onExit); });
      }
      if (failed) return finish(failed);
      for (const w of workers) {
        if (next >= jobs.length) break;
        w.postMessage({ type: 'job', job: { ...jobs[next], id: next++ } });
      }
    });
  }

  const terminate = () => Promise.all(workers.map(w => w.terminate()));
  return { size, run, terminate };
}

module.exports = { createPool, defaultWorkers };
