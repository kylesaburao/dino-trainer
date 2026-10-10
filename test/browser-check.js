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
// 5. index-css.html: the model.css network (CSS-evaluated weights) plays, every
//    decision equals policy.js on model.js, speeds 1x-2048x, auto-restart.
//    (This covers the opening only; test/css-agreement.js compares any run's
//    model on CSS vs JS decisions through the pterodactyl/duck phase.)
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { sleep, seedMathRandom, launch } = require('./cdp.js');

const ROOT = path.join(__dirname, '..');

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` (${detail})` : ''}`);
}

// Seeded Math.random for both baseline and new pages (mulberry32, seed 12345).
const SEED_SCRIPT = seedMathRandom(12345);

(async () => {
  // Baseline page from commit 2da34de in a scratch directory.
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-browser-'));
  const baseDir = path.join(scratch, 'baseline');
  fs.mkdirSync(baseDir);
  for (const f of ['index.html', 'style.css', 'game.js']) {
    fs.writeFileSync(path.join(baseDir, f), execFileSync('git', ['show', `2da34de:${f}`], { cwd: ROOT }));
  }

  const browser = await launch(path.join(scratch, 'profile'), SEED_SCRIPT);
  const { chrome, cdp } = browser;
  try {
    const { send, evaluate, load, waitUntil, problems } = browser;
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
    const hiSaved = await evaluate(`localStorage.getItem(HI_KEY)`);
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
    await evaluate(`hi = 1; localStorage.setItem(HI_KEY, '1'); __t.humanJump = 0; __t.humanDown = 0;`);
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
    check('AI runs do not overwrite the human HI', (await evaluate(`localStorage.getItem(HI_KEY)`)) === '1' && (await evaluate('hi')) === 1);

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
    check('a run that had AI on at any point does not write HI', (await evaluate(`localStorage.getItem(HI_KEY)`)) === '1');

    check('no console errors, warnings or exceptions on the page', problems.length === 0, problems.join('; '));
    console.log(`screenshot: ${path.join(scratch, 'page.png')}`);

    // ---- 4. Game versions -------------------------------------------------
    const defaultVersion = await evaluate(`({ v: game.version, want: (typeof DINO_MODEL !== 'undefined' && DINO_MODEL.gameVersion) || 'v1' })`);
    check('default page plays the game version its model was trained on', defaultVersion.v === defaultVersion.want, JSON.stringify(defaultVersion));
    await load(path.join(ROOT, 'index.html') + '?game=v2');
    await evaluate(`localStorage.setItem('dino-hi', '7')`);
    check('?game=v2 selects the v2 rules', await evaluate(`game.version === 'v2' && HI_KEY === 'dino-hi-v2'`));
    await tap('Space');
    check('v2 starts and scores', await waitUntil(`game.state === 'running'`, 1000) && await waitUntil(`game.score() > 0`, 3000));
    await waitUntil(`game.state === 'over'`, 20000);
    const v2hi = await evaluate(`({ v1: localStorage.getItem('dino-hi'), v2: localStorage.getItem('dino-hi-v2'), score: game.score() })`);
    check('v2 keeps its own HI (v1 HI untouched)', v2hi.v1 === '7' && v2hi.v2 === String(v2hi.score), JSON.stringify(v2hi));
    check('no console errors, warnings or exceptions on the v2 page', problems.length === 0, problems.join('; '));

    // ---- 5. CSS-network page (index-css.html + model.css) -----------------
    // The page has no JavaScript network; the test compares against model.js itself.
    const shipped = require(path.join(ROOT, 'model.js'));
    const { modelId } = require(path.join(ROOT, 'trainer', 'css-model.js'));
    await load(path.join(ROOT, 'index-css.html'));
    const css = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), version: game && game.version,
      loadsModelJs: typeof DINO_MODEL !== 'undefined', speeds: [...document.querySelectorAll('#speed option')].map(o => Number(o.value)) })`);
    check('CSS page: self-test passes, model.css is from model.js, and model.js is not loaded',
      !css.failed && css.id === modelId(shipped) && css.version === (shipped.gameVersion || 'v1') && !css.loadsModelJs, JSON.stringify(css));
    check('CSS page: speed options 1x to 2048x', css.speeds.join() === '1,2,4,8,16,32,64,128,256,512,1024,2048', css.speeds.join());
    // Test-only: wrap the page's cssDecide to compare each CSS decision with policy.js on model.js.
    await evaluate(`(() => { const net = DinoPolicy.createNet(${JSON.stringify(shipped.layers)}, ${JSON.stringify(Array.from(shipped.params))});
      const cssOnly = cssDecide; window.__cmp = { frames: 0, disagree: 0 };
      window.cssDecide = (x) => { const a = cssOnly(x); __cmp.frames++; if (net(x) !== a) __cmp.disagree++; return a; }; })()`);
    await evaluate(`document.getElementById('speed').value = '2048'`);
    const cssPlays = await waitUntil(`game.state === 'running' && game.score() >= 100 && __cmp.frames >= 1000`, 30000);
    const agree = await evaluate(`({ frames: __cmp.frames, disagree: __cmp.disagree, score: game.score(), fps: Math.round(rate.fps) })`);
    check('CSS page: the CSS network plays and scores', cssPlays, JSON.stringify(agree));
    check('CSS page: every CSS decision equals policy.js on model.js', agree.frames >= 1000 && agree.disagree === 0, JSON.stringify(agree));
    // Force a crash (test-only, as in section 3) and expect a fresh run to start.
    await evaluate(`(() => { const g0 = game; const id = setInterval(() => {
      if (game !== g0 || game.state !== 'running') { clearInterval(id); return; }
      const o = game.obstacles[0]; if (o) game.obstacles.push({ ...o, x: game.dino.x, y: game.dino.y });
    }, 5); window.__cssGame = g0; })()`);
    check('CSS page: crash, then a new run starts by itself', await waitUntil(`game !== window.__cssGame && game.state === 'running'`, 5000));
    check('no console errors, warnings or exceptions on the CSS page', problems.length === 0, problems.join('; '));
  } finally {
    cdp.close();
    chrome.kill();
  }
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
