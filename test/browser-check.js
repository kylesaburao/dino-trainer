'use strict';
// Headless-Chrome verification of the browser game over the DevTools protocol
// (no npm deps; Node's built-in WebSocket). Run: `node test/browser-check.js`.
//
// 1. Idle canvas is pixel-identical to baseline 2da34de (Math.random seeded the
//    same in both pages so the random ground texture matches), same canvas box.
// 2. Human play with AI off: start, jump, duck, score, crash, HI saved, restart, click-jump.
// 3. AI mode: toggle both ways; the toggle click never presses jump; AI starts,
//    scores, crashes and auto-restarts; AI presses keys only through applyKeys,
//    exactly once per fixed-timestep update; human jump/duck ignored while on;
//    AI runs never write HI; turning AI off mid-run releases its keys and hands
//    control back.
// 4. No console errors or exceptions on the page.
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9300 + Math.floor(Math.random() * 500);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` (${detail})` : ''}`);
}

// Seeded Math.random for both baseline and new pages (mulberry32, seed 12345).
const SEED_SCRIPT = `(() => { let a = 12345; Math.random = function () {
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
  // Baseline page from commit 2da34de in a scratch directory.
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-browser-'));
  const baseDir = path.join(scratch, 'baseline');
  fs.mkdirSync(baseDir);
  for (const f of ['index.html', 'style.css', 'game.js']) {
    fs.writeFileSync(path.join(baseDir, f), execFileSync('git', ['show', `2da34de:${f}`], { cwd: ROOT }));
  }

  const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${path.join(scratch, 'profile')}`,
    '--mute-audio', '--no-first-run', '--no-default-browser-check', '--window-size=900,500', '--allow-file-access-from-files', 'about:blank'],
  { stdio: 'ignore' });
  let cdp;
  try {
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
    await send('Page.addScriptToEvaluateOnNewDocument', { source: SEED_SCRIPT });

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
    const canvasShot = async () => {
      const rect = await evaluate(`(() => { const r = document.getElementById('game').getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; })()`);
      const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: rect.x, y: rect.y, width: rect.w, height: rect.h, scale: 1 } });
      return { rect, png: shot.data };
    };
    const key = async (type, code) => {
      const k = { Space: [' ', 32], ArrowDown: ['ArrowDown', 40], ArrowUp: ['ArrowUp', 38] }[code];
      await send('Input.dispatchKeyEvent', { type, code, key: k[0], windowsVirtualKeyCode: k[1], ...(type === 'keyDown' && code === 'Space' ? { text: ' ' } : {}) });
    };
    const tap = async (code, holdMs = 50) => { await key('keyDown', code); await sleep(holdMs); await key('keyUp', code); };
    const click = async (x, y) => {
      await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
      await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
    };
    const waitUntil = async (expr, ms = 15000) => {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) {
        if (await evaluate(expr)) return true;
        await sleep(30);
      }
      return false;
    };

    // ---- 1. Look: idle canvas vs baseline ---------------------------------
    await load(path.join(baseDir, 'index.html'));
    const base = await canvasShot();
    const baseProblems = [...problems];
    await load(path.join(ROOT, 'index.html'));
    const mine = await canvasShot();
    check('idle canvas pixel-identical to baseline 2da34de', base.png === mine.png);
    check('canvas position and size unchanged', JSON.stringify(base.rect) === JSON.stringify(mine.rect), JSON.stringify(mine.rect));
    check('baseline page itself loads without console errors', baseProblems.length === 0, baseProblems.join('; '));
    const btn = await evaluate(`(() => { const b = document.getElementById('ai-toggle'); const r = b.getBoundingClientRect();
      const c = document.getElementById('game').getBoundingClientRect(); const s = getComputedStyle(b);
      return { text: b.textContent, x: r.x + r.width / 2, y: r.y + r.height / 2, below: r.top >= c.bottom, color: s.color, bg: s.backgroundColor, disabled: b.disabled }; })()`);
    check('toggle button below the canvas, grey on white, reads "AI: OFF"',
      btn.below && btn.text === 'AI: OFF' && btn.color === 'rgb(83, 83, 83)' && btn.bg === 'rgb(255, 255, 255)' && !btn.disabled, JSON.stringify(btn));
    fs.writeFileSync(path.join(scratch, 'page.png'), Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64'));

    // Instrument: count human-path calls into the core (the shell's handlers call
    // game.pressJump/pressDown; the AI only goes through game.applyKeys).
    await evaluate(`(() => {
      window.__t = { humanJump: 0, humanDown: 0, applyKeys: 0, aiPresses: 0, updates: 0, aiFramesApplied: 0, runs: 0, maxScore: 0 };
      for (const [k, c] of [['pressJump', 'humanJump'], ['pressDown', 'humanDown']]) {
        const orig = game[k]; game[k] = (...a) => { __t[c]++; return orig(...a); };
      }
      const apply = game.applyKeys;
      game.applyKeys = (keys, now) => {
        __t.applyKeys++;
        if (game.state === 'running' && ((keys.jump && !game.keys.jump) || (keys.down && !game.keys.down))) __t.aiPresses++;
        return apply(keys, now);
      };
      const update = game.update;
      let last = game.state;   // a run starts between two updates (input applied before update)
      game.update = (ms, now) => { __t.updates++; if (aiOn) __t.aiFramesApplied++;
        if (last !== 'running' && game.state === 'running') __t.runs++;
        update(ms, now); last = game.state; __t.maxScore = Math.max(__t.maxScore, game.score()); };
      window.__applyAtUpdate = [];
    })()`);
    // The loop calls game.update via the property, so the wrapper sees every step.
    const T = (k) => evaluate(`__t.${k}`);

    // ---- 2. Human play, AI off ------------------------------------------
    await key('keyDown', 'Space');
    const started = await waitUntil(`game.state === 'running' && game.dino.jumping`, 2000);
    await key('keyUp', 'Space');
    check('Space starts the game with a jump', started);
    await sleep(800);
    const s1 = await evaluate('game.score()');
    check('score advances while running', s1 > 0, `score ${s1}`);
    await waitUntil(`!game.dino.jumping`, 2000);
    await key('keyDown', 'ArrowDown');
    const ducked = await waitUntil(`game.dino.ducking`, 1000);
    await key('keyUp', 'ArrowDown');
    const unducked = await waitUntil(`!game.dino.ducking`, 1000);
    check('ArrowDown ducks and release stands up', ducked && unducked);
    await key('keyDown', 'Space');
    await sleep(30);
    await key('keyDown', 'ArrowDown');
    const fastDrop = await waitUntil(`game.dino.speedDrop`, 1000);
    await key('keyUp', 'ArrowDown');
    await key('keyUp', 'Space');
    check('ArrowDown mid-air fast-drops', fastDrop);
    const crashed = await waitUntil(`game.state === 'over'`, 20000);
    const humanScore = await evaluate('game.score()');
    check('running into an obstacle ends the game', crashed, `score ${humanScore}`);
    const hiSaved = await evaluate(`localStorage.getItem('dino-hi')`);
    check('human run saves HI to localStorage', hiSaved === String(humanScore), `HI ${hiSaved}`);
    await sleep(900);
    await tap('Space');
    check('Space restarts after the lockout', await waitUntil(`game.state === 'running'`, 1000));
    await waitUntil(`!game.dino.jumping`, 2000);
    const canvasRect = (await canvasShot()).rect;
    await click(canvasRect.x + 100, canvasRect.y + 50);
    check('click/tap on the page jumps', await waitUntil(`game.dino.jumping`, 500));
    await waitUntil(`game.state === 'over'`, 20000);

    // ---- 3. AI mode -----------------------------------------------------
    await evaluate(`hi = 1; localStorage.setItem('dino-hi', '1'); __t.humanJump = 0; __t.humanDown = 0;`);
    await sleep(900);   // past the restart lockout
    await click(btn.x, btn.y);
    await sleep(50);
    check('toggle turns AI on ("AI: ON")', await evaluate(`aiOn && document.getElementById('ai-toggle').textContent === 'AI: ON'`));
    check('clicking the toggle did not press jump', (await T('humanJump')) === 0);
    check('toggle does not keep keyboard focus', await evaluate(`document.activeElement !== document.getElementById('ai-toggle')`));
    const aiStarted = await waitUntil(`game.state === 'running'`, 2000);
    check('AI starts a run by itself', aiStarted);
    await evaluate(`__t.applyKeys = 0; __t.aiFramesApplied = 0; __t.runs = 0; __t.maxScore = 0; __t.aiPresses = 0;`);
    // Human input while AI is on must be ignored.
    await tap('Space');
    await tap('ArrowDown');
    await click(canvasRect.x + 100, canvasRect.y + 50);
    check('human jump/duck input ignored while AI is on', (await T('humanJump')) === 0 && (await T('humanDown')) === 0);
    // A trained model may not crash for minutes, so once it has scored, force a
    // crash (test-only): drop a copy of an on-screen obstacle onto the dino every
    // frame until the run ends, then expect the AI to restart by itself.
    const scored = await waitUntil(`__t.maxScore >= 100 && game.obstacles.length > 0`, 30000);
    await evaluate(`(() => { const id = setInterval(() => {
      if (game.state !== 'running') { clearInterval(id); return; }
      const o = game.obstacles[0]; if (o) game.obstacles.push({ ...o, x: game.dino.x, y: game.dino.y });
    }, 5); })()`);
    const crashedAI = await waitUntil(`game.state === 'over'`, 3000);
    const restarted = crashedAI && await waitUntil(`__t.runs >= 1 && game.state === 'running'`, 3000);
    const t = await evaluate('__t');
    check('AI plays and scores, crashes and auto-restarts', scored && restarted && t.maxScore > 0, `runs ${t.runs}, max score ${t.maxScore}`);
    check('AI picks keys exactly once per fixed-timestep update', t.applyKeys === t.aiFramesApplied && t.applyKeys > 0, `applyKeys ${t.applyKeys}, updates with AI ${t.aiFramesApplied}`);
    check('AI issued key presses through the real input functions', t.aiPresses > 0, `in-run presses ${t.aiPresses}`);
    check('AI runs do not overwrite the human HI', (await evaluate(`localStorage.getItem('dino-hi')`)) === '1' && (await evaluate('hi')) === 1);

    // Turn AI off mid-run: keys released, the click does not jump, the player has control.
    await waitUntil(`game.state === 'running' && !game.dino.jumping && game.score() < 20`, 20000);
    await evaluate(`__t.humanJump = 0`);
    await click(btn.x, btn.y);
    await sleep(50);
    const off = await evaluate(`({ aiOn, text: document.getElementById('ai-toggle').textContent, keys: { ...game.keys }, state: game.state, jumping: game.dino.jumping })`);
    check('toggle turns AI off mid-run ("AI: OFF")', !off.aiOn && off.text === 'AI: OFF' && off.state === 'running', JSON.stringify(off));
    check('turning AI off releases the AI-held keys', !off.keys.jump && !off.keys.down);
    check('clicking the toggle off did not press jump', (await T('humanJump')) === 0 && !off.jumping);
    await waitUntil(`!game.dino.jumping`, 2000);
    await tap('Space');
    check('player has control again after AI off', (await T('humanJump')) === 1 && await waitUntil(`game.dino.jumping || game.state === 'over'`, 500));
    await waitUntil(`game.state === 'over'`, 20000);
    check('a run that had AI on at any point does not write HI', (await evaluate(`localStorage.getItem('dino-hi')`)) === '1');

    check('no console errors, warnings or exceptions on the page', problems.length === 0, problems.join('; '));
    console.log(`screenshot: ${path.join(scratch, 'page.png')}`);
  } finally {
    if (cdp) cdp.close();
    chrome.kill();
  }
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
