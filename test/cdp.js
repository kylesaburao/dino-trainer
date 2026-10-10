'use strict';
// Headless Chrome over the DevTools protocol (no npm deps; Node's built-in
// WebSocket), shared by test/browser-check.js and test/css-agreement.js.
// launch(profileDir) starts one headless Chrome and returns { chrome, cdp, send,
// evaluate, load, waitUntil, problems }; problems collects console errors,
// warnings and exceptions (load() clears it). The caller must close cdp and
// kill chrome.
const { spawn } = require('child_process');

const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// A page script that replaces Math.random with mulberry32(seed), so pages that
// seed games from Math.random play reproducibly.
const seedMathRandom = (seed) => `(() => { let a = ${seed | 0}; Math.random = function () {
  a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();`;

async function connect(port) {
  const t0 = Date.now();
  for (;;) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
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

// Starts Chrome and attaches to its page. `newDocumentScript` (optional) runs
// before every page's own scripts. If attaching fails, Chrome is killed.
async function launch(profileDir, newDocumentScript) {
  const port = 9300 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profileDir}`,
    '--mute-audio', '--no-first-run', '--no-default-browser-check', '--window-size=900,500', '--allow-file-access-from-files', 'about:blank'],
  { stdio: 'ignore' });
  let cdp;
  try {
    cdp = cdpClient(await connect(port));
    await cdp.ready;
  } catch (e) {
    if (cdp) cdp.close();
    chrome.kill();
    throw e;
  }
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
  if (newDocumentScript) await send('Page.addScriptToEvaluateOnNewDocument', { source: newDocumentScript });

  const evaluate = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(`evaluate failed: ${expr}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
    return r.result.value;
  };
  const load = async (file) => {
    problems.length = 0;
    const loaded = new Promise((res) => cdp.on((m) => { if (m.method === 'Page.loadEventFired') res(); }));
    await send('Page.navigate', { url: 'file://' + file });
    await loaded;
    await sleep(600);
  };
  const waitUntil = async (expr, ms = 15000, pollMs = 30) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await evaluate(expr)) return true;
      await sleep(pollMs);
    }
    return false;
  };
  return { chrome, cdp, send, evaluate, load, waitUntil, problems };
}

module.exports = { CHROME, sleep, seedMathRandom, launch };
