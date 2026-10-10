'use strict';
// Headless-Chrome check of the training dashboard (trainer/dashboard.js + dashboard.html),
// with the same CDP approach as test/browser-check.js. Run: `node test/dashboard-check.js
// [--shots DIR]` (CHROME env var overrides the Chrome path).
//
// It builds a throwaway runs/ tree in the OS temp dir, so it depends only on committed
// files (runs/v2-h32x32-1m, runs/bc-v2-h32x32):
//   es-run              ES run: best.json, metrics.jsonl, evals.jsonl
//   grp/bc-run          nested BC run, training: train.lock (this pid), bc-metrics.jsonl with
//                       none/jump/duck recall, snapshots/epoch-01..02.json, best.json, and
//                       its log next to it as grp/bc-run.log
//   grp/demos-set       nested bc-gen output, generating: demos.json (teacherName) and
//                       demos.bin.<this pid>.tmp
//   plain               an empty directory (listed as before)
// Checks the JSON API (discovery, kinds, running detection, snapshot weights, name
// whitelist), then the page: BC charts with every recall class, a BC model and a
// snapshot playing, the teacher (test/solver.js through /teacher.js) playing past
// 500 points without crashing, the ES view and playback, and no console errors.
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CDP_PORT = 9800 + Math.floor(Math.random() * 100);
const DASH_PORT = 9900 + Math.floor(Math.random() * 90);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const shotsArg = process.argv.indexOf('--shots');

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` (${detail})` : ''}`);
}

function buildFixture(dir) {
  const w = (f, s) => { fs.mkdirSync(path.dirname(path.join(dir, f)), { recursive: true }); fs.writeFileSync(path.join(dir, f), s); };
  const src = (f) => fs.readFileSync(path.join(ROOT, f));
  for (const f of ['best.json', 'metrics.jsonl', 'evals.jsonl']) w(`es-run/${f}`, src(`runs/v2-h32x32-1m/${f}`));
  fs.mkdirSync(path.join(dir, 'plain'));

  const demosFile = path.join(dir, 'grp', 'demos-set', 'demos.json');
  w('grp/demos-set/demos.json', JSON.stringify({
    format: 'dino-bc-demos-v1', file: 'demos.bin', obsDim: 44, rowFloats: 46, coreHash: 'f'.repeat(64), gameVersion: 'v2',
    teacher: 'test/solver.js', teacherName: 'teacher-duck', seeds: [1, 2, 3], levels: [0, 2000], span0: 3000, span: 5000, keepBoring: 10,
    rows: 1000, counts: { rows: { none: 900, jump: 70, duck: 30 }, weighted: { none: 950, jump: 70, duck: 30 } },
    solverCrashes: 0, sha256: '0'.repeat(64), createdAt: '2026-10-10T00:00:00.000Z', episodes: [],
  }));
  w(`grp/demos-set/demos.bin.${process.pid}.tmp`, '');

  const best = JSON.parse(src('runs/bc-v2-h32x32/best.json'));
  best.config = { target: 1000000, hidden: [32, 32], actionRepeat: 1, gameVersion: 'v2', method: 'behaviour cloning of test/solver.js (supervised; no ES)',
    trainer: 'trainer/cli.js bc-train --backend python', backend: 'python', epochs: 3, batch: 1024, lr: 0.002, seed: 1,
    demos: demosFile, demosRows: 1000, trainSeconds: 6, snapshots: 'grp/bc-run/snapshots' };
  w('grp/bc-run/best.json', JSON.stringify(best));
  w('grp/bc-run/bc-metrics.jsonl', [0, 1].map(e => JSON.stringify({ backend: 'python', epoch: e, loss: 0.1 / (e + 1),
    recall: { none: 0.99 + e * 0.004, jump: 0.98 + e * 0.01, duck: 0.9 + e * 0.05 }, seconds: 2 + e })).join('\n') + '\n');
  for (const e of [1, 2]) w(`grp/bc-run/snapshots/epoch-0${e}.json`, JSON.stringify({ layers: best.layers, params: best.params, epoch: e }));
  w('grp/bc-run/train.lock', JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString() }) + '\n');
  const old = JSON.parse(src('runs/bc-v2-h32x32/best.json'));   // research-trained: no bc-metrics, no demos, no teacherName
  w('grp/bc-old/best.json', JSON.stringify(old));
  w('grp/bc-run.log', 'bc-train: dir=grp/bc-run backend=python layers=44-32-32-3 epochs 3\nepoch 0: loss 0.1000\nepoch 1: loss 0.0500\n');
}

async function connect() {
  const t0 = Date.now();
  for (;;) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      const page = list.find(t => t.type === 'page');
      if (page) return page.webSocketDebuggerUrl;
    } catch (e) { /* not up yet */ }
    if (Date.now() - t0 > 15000) throw new Error('Chrome did not start');
    await sleep(100);
  }
}

function cdpClient(url) {
  const ws = new WebSocket(url);
  let id = 0;
  const pending = new Map();
  const listeners = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method) {
      for (const l of listeners) l(msg);
    }
  };
  const ready = new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const i = ++id;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  return { ready, send, on: (f) => listeners.push(f), close: () => ws.close() };
}

(async () => {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-dash-'));
  const runs = path.join(scratch, 'runs');
  buildFixture(runs);
  const shots = shotsArg >= 0 ? path.resolve(process.argv[shotsArg + 1]) : scratch;
  fs.mkdirSync(shots, { recursive: true });
  const base = `http://127.0.0.1:${DASH_PORT}`;
  const dash = spawn(process.execPath, [path.join(ROOT, 'trainer', 'dashboard.js'), '--port', String(DASH_PORT), '--runs', runs], { stdio: 'ignore' });
  let chrome, cdp;
  try {
    // ---- 1. JSON API ------------------------------------------------------
    let list = null;
    for (let i = 0; i < 50 && !list; i++) { try { list = await (await fetch(base + '/api/runs')).json(); } catch (e) { await sleep(100); } }
    const by = Object.fromEntries((list || []).map(r => [r.name, r]));
    check('run list: top-level, nested and plain dirs; a group is not a run',
      ['es-run', 'grp/bc-run', 'grp/bc-old', 'grp/demos-set', 'plain'].every(n => by[n]) && !by.grp && list.length === 5, (list || []).map(r => r.name).join(', '));
    const bc = by['grp/bc-run'] || {}, dm = by['grp/demos-set'] || {};
    check('ES run is kind es, generation from metrics', by['es-run'] && by['es-run'].kind === 'es' && by['es-run'].generation === 970);
    const old = by['grp/bc-old'] || {};
    check('best.json-only BC run (config.method): kind bc, teacher falls back to teacher-jump',
      old.kind === 'bc' && old.bc.teacher === 'teacher-jump' && old.best.validation.count === 25);
    check('BC run: kind, teacher from its demos manifest, backend, layers, epochs, snapshots',
      bc.kind === 'bc' && bc.bc.teacher === 'teacher-duck' && bc.bc.backend === 'python' && bc.bc.layers.join('-') === '44-32-32-3' &&
      bc.bc.epochsDone === 2 && bc.bc.snapshots.length === 2, JSON.stringify(bc.bc && { ...bc.bc, demos: undefined }));
    check('bc-train with a live train.lock shows as running', bc.running && bc.pid === process.pid && bc.process === 'bc-train');
    check('bc-gen (no lock) shows as running from demos.bin.<pid>.tmp', dm.kind === 'demos' && dm.running && dm.process === 'bc-gen');
    check('teacherName from the demos manifest', dm.bc && dm.bc.teacher === 'teacher-duck');
    const detail = await (await fetch(base + '/api/runs/' + encodeURIComponent('grp/bc-run'))).json();
    check('BC detail: per-epoch rows, demos summary, nested log from <group>/<run>.log',
      detail.metrics.length === 2 && detail.info.bc.demos && detail.info.bc.demos.counts.rows.duck === 30 && detail.log.length === 3);
    const snap = await (await fetch(base + '/api/runs/' + encodeURIComponent('grp/bc-run') + '/model?which=epoch-02')).json();
    check('snapshot weights served in dino-mlp-v1 form', snap.format === 'dino-mlp-v1' && snap.epoch === 2 && snap.gameVersion === 'v2' && snap.params.length > 0);
    const bad = await Promise.all(['..%2F..', 'grp', 'grp%2F..%2Fes-run', 'nope'].map(n => fetch(`${base}/api/runs/${n}`).then(r => r.status)));
    check('names outside the run list are refused', bad.every(s => s === 404), bad.join(','));
    const shim = await (await fetch(base + '/teacher.js')).text();
    check('/teacher.js wraps test/solver.js', shim.includes(fs.readFileSync(path.join(ROOT, 'test', 'solver.js'), 'utf8')));

    // ---- 2. Page ----------------------------------------------------------
    chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${path.join(scratch, 'profile')}`,
      '--mute-audio', '--no-first-run', '--no-default-browser-check', '--window-size=1200,900', 'about:blank'], { stdio: 'ignore' });
    cdp = cdpClient(await connect());
    await cdp.ready;
    const { send } = cdp;
    const problems = [];
    cdp.on((m) => {
      if (m.method === 'Runtime.exceptionThrown') problems.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
      if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) {
        problems.push(`console.${m.params.type}: ` + m.params.args.map(a => a.value ?? a.description).join(' '));
      }
      if (m.method === 'Log.entryAdded' && (m.params.entry.level === 'error' || m.params.entry.level === 'warning')) {
        problems.push(`log ${m.params.entry.level}: ${m.params.entry.text} ${m.params.entry.url || ''}`);
      }
    });
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Log.enable');
    const evaluate = async (expr) => {
      const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error(`evaluate failed: ${expr}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
      return r.result.value;
    };
    const waitUntil = async (expr, ms = 15000) => {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) {
        if (await evaluate(expr)) return true;
        await sleep(100);
      }
      return false;
    };
    const shot = async (name) => {
      const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
      fs.writeFileSync(path.join(shots, name), Buffer.from(data, 'base64'));
    };
    const pick = (run) => evaluate(`document.querySelector('[data-run="${run}"]').click()`);
    const setWhich = (v) => evaluate(`(() => { const s = document.getElementById('pb-which'); s.value = ${JSON.stringify(v)}; s.onchange(); })()`);
    const setSpeed = (v) => evaluate(`document.getElementById('pb-speed').value = '${v}'`);

    const loaded = new Promise((res) => cdp.on((m) => { if (m.method === 'Page.loadEventFired') res(); }));
    await send('Page.navigate', { url: base + '/' });
    await loaded;
    await waitUntil(`document.querySelectorAll('.run-chip').length === 5`);

    // BC run: charts, chip, best weights playing.
    await pick('grp/bc-run');
    const charts = await waitUntil(`(() => { const r = document.querySelector('#c-recall svg');
      return r && r.querySelectorAll('path').length === 3 && document.querySelector('#c-loss svg path') && document.querySelector('#c-secs svg path'); })()`);
    const view = await evaluate(`({ legend: [...document.querySelectorAll('.legend span')].map(s => s.textContent),
      chip: document.querySelector('[data-run="grp/bc-run"]').textContent.replace(/\\s+/g, ' ').trim(),
      snaps: document.querySelectorAll('.snaps button').length, text: document.getElementById('detail').textContent,
      options: [...document.querySelectorAll('#pb-which option')].map(o => o.value) })`);
    check('BC charts render: loss, seconds per epoch, recall for none/jump/duck', charts && view.legend.join() === 'none,jump,duck', view.legend.join());
    check('BC chip shows the badge, teacher, backend and layers', /BC.*grp\/bc-run.*running.*teacher-duck.*44-32-32-3.*python/.test(view.chip), view.chip);
    check('BC view shows the demos manifest, validation and snapshots',
      view.snaps === 2 && /teacher-duck/.test(view.text) && /Solver crashes/.test(view.text) && /25\/25/.test(view.text));
    check('BC weights menu: best, both snapshots, the teacher(s)',
      view.options[0] === 'best' && view.options.includes('epoch-01') && view.options.includes('epoch-02') && view.options.includes('teacher:createSolver'), view.options.join());
    await setSpeed(64);
    const bcPlays = await waitUntil(`__dash.pb.model && __dash.pb.model.which === 'best' && __dash.pb.game && __dash.pb.game.score() >= 300`, 20000);
    check('a BC model plays', bcPlays && (await evaluate('__dash.nn.mismatches')) === 0, `score ${await evaluate('__dash.pb.game && __dash.pb.game.score()')}`);
    await shot('dashboard-bc.png');

    await evaluate(`document.querySelector('.snaps button[data-snap="epoch-01"]').click()`);
    check('a snapshot epoch plays', await waitUntil(`__dash.pb.model && __dash.pb.model.epoch === 1 && __dash.pb.game.score() >= 100`, 20000));

    // Teacher playback.
    await setWhich('teacher:createSolver');
    await setSpeed(256);
    await waitUntil(`__dash.pb.model && __dash.pb.model.teacher === 'createSolver' && __dash.pb.game && __dash.pb.game.state === 'running'`, 10000);
    const ep0 = await evaluate('__dash.pb.episodes');
    const teacherOk = await waitUntil(`__dash.pb.game.score() >= 1500 || __dash.pb.game.state === 'over' || __dash.pb.episodes !== ${ep0}`, 30000);
    const t = await evaluate(`({ score: __dash.pb.game.score(), state: __dash.pb.game.state, episodes: __dash.pb.episodes, version: __dash.pb.game.version,
      caption: document.getElementById('pb-caption').textContent })`);
    check('the teacher (solver) plays past 1500 points without crashing',
      teacherOk && t.score >= 1500 && t.state === 'running' && t.episodes === ep0 && t.version === 'v2' && /teacher-jump/.test(t.caption), JSON.stringify(t));
    await shot('dashboard-teacher.png');

    // Demos-only run: teacher playback by default.
    await pick('grp/demos-set');
    check('demos run: manifest view and teacher playback',
      await waitUntil(`/BC demos.*teacher-duck.*1\.0k rows/.test(document.querySelector('[data-run="grp/demos-set"]').textContent) && __dash.pb.run === 'grp/demos-set' && __dash.pb.model && !!__dash.pb.model.teacher`, 10000));

    // ES run: unchanged view and playback.
    await setSpeed(16);
    await pick('es-run');
    const esView = await waitUntil(`document.querySelector('#c-fit svg path') && document.querySelector('#c-val svg') && __dash.pb.model && __dash.pb.model.generation === 970`, 15000);
    const esOpts = await evaluate(`[...document.querySelectorAll('#pb-which option')].map(o => o.value + '=' + o.textContent).join()`);
    check('ES run renders (fitness/validation charts, original weights menu)', esView && esOpts === 'current=Latest checkpoint,best=Best (validation)', esOpts);
    check('ES model plays', await waitUntil(`__dash.pb.game && __dash.pb.game.score() >= 100`, 20000));
    await shot('dashboard-es.png');

    check('no console errors, warnings or exceptions on the dashboard', problems.length === 0, problems.join('; '));
    console.log(`screenshots: ${shots}`);
  } finally {
    if (cdp) cdp.close();
    if (chrome) chrome.kill();
    dash.kill();
  }
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} dashboard checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
