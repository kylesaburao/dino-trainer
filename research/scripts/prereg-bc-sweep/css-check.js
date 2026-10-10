'use strict';
// CSS-vs-JS check of a model on a scratch copy of the CSS-network page (make-page.sh), in
// headless Chrome over the DevTools protocol (the same method as test/browser-check.js §5).
//   1. The page's self-test passes and its model.css is generated from --model (modelId).
//   2. Played frames: the page plays (speed 2048x, auto-restart after crashes) until
//      --frames CSS decisions; each is compared with policy.js createNet on --model, and
//      timed: performance.now() around cssDecide, i.e. the --x0..--x43 writes plus the
//      forced style recalc and the read of --act (prereg-bc-shapes.md §3.2).
//   3. Demo rows (optional, --rows demos.bin --n K): K rows at an even stride are fed to the
//      page's cssDecide directly and compared with policy.js; this exercises both actions even
//      for a model that never jumps in play, and gives a back-to-back time per decision.
// Prints one JSON object. Exit code 1 on any disagreement or page failure.
// Usage: node css-check.js --page <dir> --model <model.js> [--frames 1000] [--rows demos.bin --n 2000]
//          [--timeout 120000] [--out result.json]
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = '/Users/kyle/Desktop/dino-game';
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const PAGE = path.resolve(opt('--page'));
const MODEL = path.resolve(opt('--model'));
const FRAMES = Number(opt('--frames', 1000));
const ROWS = opt('--rows');
const NROWS = Number(opt('--n', 2000));
const TIMEOUT = Number(opt('--timeout', 120000));
const OUT = opt('--out');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9800 + Math.floor(Math.random() * 500);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const SEED_SCRIPT = `(() => { let a = 4242; Math.random = function () {
  a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();`;

async function connect() {
  const t0 = Date.now();
  for (;;) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
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
    } else if (msg.method) for (const l of listeners) l(msg);
  };
  const ready = new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const i = ++id;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  return { ready, send, on: (f) => listeners.push(f), close: () => ws.close() };
}

function demoRows(file, n) {
  const OBS = 44, ROW = 46;
  const f = new Float32Array(fs.readFileSync(file).buffer.slice(0));
  const total = f.length / ROW, stride = Math.max(1, Math.floor(total / n));
  const rows = [];
  for (let i = 0; i < total && rows.length < n; i += stride) rows.push(Array.from(f.subarray(i * ROW, i * ROW + OBS)));
  return rows;
}

(async () => {
  const model = require(MODEL);
  const { modelId } = require(path.join(ROOT, 'trainer', 'css-model.js'));
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-csscheck-'));
  const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${path.join(scratch, 'profile')}`,
    '--mute-audio', '--no-first-run', '--no-default-browser-check', '--window-size=900,500', '--allow-file-access-from-files', 'about:blank'],
  { stdio: 'ignore' });
  const result = { page: PAGE, model: MODEL, layers: model.layers.join('-'), cssBytes: fs.statSync(path.join(PAGE, 'model.css')).size };
  let cdp, ok = false;
  try {
    cdp = cdpClient(await connect());
    await cdp.ready;
    const { send } = cdp;
    const problems = [];
    cdp.on((m) => {
      if (m.method === 'Runtime.exceptionThrown') problems.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
      if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) problems.push(`console.${m.params.type}: ` + m.params.args.map(a => a.value ?? a.description).join(' '));
    });
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Page.addScriptToEvaluateOnNewDocument', { source: SEED_SCRIPT });
    const evaluate = async (expr) => {
      const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error(`evaluate failed: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
      return r.result.value;
    };
    const loaded = new Promise((res) => cdp.on((m) => { if (m.method === 'Page.loadEventFired') res(); }));
    const tLoad = Date.now();
    await send('Page.navigate', { url: 'file://' + path.join(PAGE, 'index-css.html') });
    await loaded;
    result.loadMs = Date.now() - tLoad;
    await sleep(600);
    const st = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), version: game && game.version })`);
    result.selfTest = st;
    result.idMatches = st.id === modelId(model);
    if (st.failed || !result.idMatches) throw new Error(`page not ready: ${JSON.stringify(st)}`);
    await evaluate(`(() => { const net = DinoPolicy.createNet(${JSON.stringify(model.layers)}, ${JSON.stringify(Array.from(model.params))});
      const cssOnly = cssDecide; window.__net = net; window.__cssOnly = cssOnly;
      window.__cmp = { frames: 0, disagree: 0, acts: [0, 0, 0], ms: [] };
      window.cssDecide = (x) => { const t0 = performance.now(); const a = cssOnly(x); __cmp.ms.push(performance.now() - t0);
        __cmp.frames++; __cmp.acts[a]++; if (net(x) !== a) __cmp.disagree++; return a; }; })()`);
    await evaluate(`document.getElementById('speed').value = '2048'`);
    const t0 = Date.now();
    while (Date.now() - t0 < TIMEOUT && (await evaluate('__cmp.frames')) < FRAMES) await sleep(100);
    const played = await evaluate(`(() => { const ms = __cmp.ms.slice().sort((a, b) => a - b);
      return { frames: __cmp.frames, disagree: __cmp.disagree, acts: __cmp.acts, best, fps: Math.round(rate.fps),
        msMedian: ms[ms.length >> 1], msMean: ms.reduce((a, b) => a + b, 0) / ms.length, msP90: ms[Math.floor(ms.length * 0.9)] }; })()`);
    result.played = played;
    // Stop the game loop before the direct demo-row test (no further simulation).
    await evaluate(`failed = new Error('stopped by css-check'); true`);
    if (ROWS) {
      const rows = demoRows(path.resolve(ROWS), NROWS);
      let disagree = 0, acts = [0, 0, 0], ms = 0;
      for (let i = 0; i < rows.length; i += 250) {
        const r = await evaluate(`(() => { const rows = ${JSON.stringify(rows.slice(i, i + 250))}; let d = 0; const a = [0, 0, 0];
          const t0 = performance.now(); const out = rows.map(x => __cssOnly(Float64Array.from(x))); const ms = performance.now() - t0;
          rows.forEach((x, k) => { a[out[k]]++; if (__net(Float64Array.from(x)) !== out[k]) d++; }); return { d, a, ms }; })()`);
        disagree += r.d; ms += r.ms; acts = acts.map((v, k) => v + r.a[k]);
      }
      result.rows = { n: rows.length, disagree, acts, msPerDecision: ms / rows.length };
    }
    result.problems = problems.filter(p => !p.includes('stopped by css-check'));
    ok = played.frames >= FRAMES && played.disagree === 0 && (!result.rows || result.rows.disagree === 0);
  } catch (e) {
    result.error = e.message;
  } finally {
    if (cdp) cdp.close();
    chrome.kill();
  }
  result.ok = ok;
  const text = JSON.stringify(result);
  console.log(text);
  if (OUT) fs.writeFileSync(path.resolve(OUT), text + '\n');
  process.exit(ok ? 0 : 1);
})();
