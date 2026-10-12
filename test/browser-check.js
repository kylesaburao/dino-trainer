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
// 4. No console errors or exceptions on the page. Rules: both pages always play the
//    latest rules (the last of DinoCore.GAME_VERSIONS); ?game= is ignored; HI is
//    kept under the latest rules' key; a root model for other rules plays the
//    latest rules with a status-line warning (none when it is for the latest).
// 5. index-css.html: the model.css network (CSS-evaluated weights) plays, every
//    decision equals policy.js on model.js, speeds 1x-4096x, auto-restart.
//    (This covers the opening only; test/css-agreement.js compares any run's
//    model on CSS vs JS decisions through the pterodactyl/duck phase.)
// 6. Model selector and speed on both pages: the default is the shipped model
//    (root model.js / model.css, and models.js marks the same model shipped); the
//    selectors list exactly the latest-rules models; every latest-rules bundled
//    model (models.js, models-css/) loads with ?model=<id> and plays a few hundred
//    frames at 4096x with the policy called once per update (index.html) and 0
//    CSS/JS disagreements (index-css.html); a pick switches the model in place on
//    both pages (no reload; the same game and run continue; index.html: with AI off
//    and mid-run at 4096x, a new agent for the pick; index-css.html: the new
//    stylesheet replaces the old one, then 0 disagreements with the pick; a pick
//    with the active --nn-model-id loads nothing; a missing stylesheet keeps the
//    old model and warns without console output) and is remembered as ?model=<id>
//    only; a ?model= or remembered choice for other rules falls back to the root
//    model and the remembered one is cleared. Prints the achieved speed.
//    Independent of which model is the root.
// 7. Network view (netview.js) under the game on both pages: it shows the selected
//    model's shape (and follows a model change), is idle with AI off, draws live
//    decisions during AI play at 4096x with no mismatch warning (index.html: vs the
//    agent's keys; index-css.html: the JS argmax vs the CSS --act), its show/hide
//    toggle works without pressing jump, and both pages fit 320px- and 375px-wide phones.
//    The CSS page loads models.js for the view only; it never loads the root model.js.
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { sleep, seedMathRandom, launch } = require('./cdp.js');

const ROOT = path.join(__dirname, '..');
const { GAME_VERSIONS } = require(path.join(ROOT, 'core.js'));
const LATEST = GAME_VERSIONS[GAME_VERSIONS.length - 1];   // the rules both pages must play
const LATEST_HI = LATEST === 'v1' ? 'dino-hi' : `dino-hi-${LATEST}`;

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` (${detail})` : ''}`);
}

// Seeded Math.random for both baseline and new pages (mulberry32, seed 12345).
const SEED_SCRIPT = seedMathRandom(12345);
const SPEEDS = '1,2,4,8,16,32,64,128,256,512,1024,2048,4096';
const MODEL_FRAMES = 300;   // AI-played frames per bundled model and page in section 6

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
    const hiSaved = await evaluate(`({ key: HI_KEY, hi: localStorage.getItem('${LATEST_HI}') })`);
    check(`human run saves HI to localStorage under the latest rules' key ${LATEST_HI}`, hiSaved.key === LATEST_HI && hiSaved.hi === String(humanScore), JSON.stringify(hiSaved));
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
    check('toggle turns AI on ("AI: ON") and enables the speed selector', await evaluate(`aiOn && document.getElementById('ai-toggle').textContent === 'AI: ON' && !speedSelect.disabled`));
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
    check('speed selector is AI-only: disabled again with AI off', await evaluate(`speedSelect.disabled`));
    check('turning AI off releases the AI-held keys', !off.keys.jump && !off.keys.down);
    check('clicking the toggle off did not press jump', (await T('humanJump')) === 0 && !off.jumping);
    await waitUntil(`!game.dino.jumping`, 2000);
    await tap('Space');
    check('player has control again after AI off', (await T('humanJump')) === 1 && await waitUntil(`game.dino.jumping || game.state === 'over'`, 500));
    await waitUntil(`game.state === 'over'`, 20000);
    check('a run that had AI on at any point does not write HI', (await evaluate(`localStorage.getItem(HI_KEY)`)) === '1');

    check('no console errors, warnings or exceptions on the page', problems.length === 0, problems.join('; '));
    console.log(`screenshot: ${path.join(scratch, 'page.png')}`);

    // ---- 4. Game versions: always the latest rules -------------------------
    const shipped = require(path.join(ROOT, 'model.js'));
    const shippedLatest = (shipped.gameVersion || 'v1') === LATEST;
    // The status line warns exactly when the (root) model is for other rules.
    const warnRe = `/trained for rules v\\d+; the page plays rules ${LATEST}/`;
    const versionInfo = `({ v: game.version, attr: document.documentElement.dataset.gameVersion, hiKey: HI_KEY, warned: ${warnRe}.test(speedStatus.textContent), status: speedStatus.textContent })`;
    await sleep(1100);   // past one status-line refresh
    const defaultVersion = await evaluate(versionInfo);
    check(`index.html plays the latest rules ${LATEST} (data-game-version too), HI key ${LATEST_HI}`, defaultVersion.v === LATEST && defaultVersion.attr === LATEST &&
      defaultVersion.hiKey === LATEST_HI, JSON.stringify(defaultVersion));
    check(`index.html: rules warning in the status line exactly when the root model is for other rules (root ${shipped.gameVersion})`,
      defaultVersion.warned === !shippedLatest, JSON.stringify(defaultVersion));
    const otherHi = GAME_VERSIONS.filter(v => v !== LATEST).map(v => v === 'v1' ? 'dino-hi' : `dino-hi-${v}`);
    for (const v of GAME_VERSIONS.filter(v => v !== LATEST)) {
      await load(path.join(ROOT, 'index.html') + `?game=${v}`);
      await evaluate(`(() => { ${JSON.stringify(otherHi)}.forEach((k, i) => localStorage.setItem(k, String(7 + i))); localStorage.removeItem('${LATEST_HI}'); hi = 0; })()`);
      const asked = await evaluate(versionInfo);
      check(`index.html ?game=${v} is ignored: plays rules ${LATEST}, HI key ${LATEST_HI}`, asked.v === LATEST && asked.attr === LATEST && asked.hiKey === LATEST_HI, JSON.stringify(asked));
    }
    await tap('Space');
    check(`rules ${LATEST} start and score`, await waitUntil(`game.state === 'running'`, 1000) && await waitUntil(`game.score() > 0`, 3000));
    await waitUntil(`game.state === 'over'`, 20000);
    const latestHi = await evaluate(`({ latest: localStorage.getItem('${LATEST_HI}'), others: ${JSON.stringify(otherHi)}.map(k => localStorage.getItem(k)), score: game.score() })`);
    check(`a human run writes HI under ${LATEST_HI} only (other rules' HI untouched)`, latestHi.latest === String(latestHi.score) &&
      latestHi.others.every((h, i) => h === String(7 + i)), JSON.stringify(latestHi));
    check('no console errors, warnings or exceptions on the ?game= pages', problems.length === 0, problems.join('; '));

    // ---- 5. CSS-network page (index-css.html + model.css) -----------------
    // The page decides only in CSS (models.js is loaded for the network view only);
    // the test compares against the root model.js itself.
    const { modelId } = require(path.join(ROOT, 'trainer', 'css-model.js'));
    await load(path.join(ROOT, 'index-css.html'));
    const css = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), version: game && game.version,
      attr: document.documentElement.dataset.gameVersion,
      loadsModelJs: typeof DINO_MODEL !== 'undefined', speeds: [...document.querySelectorAll('#speed option')].map(o => Number(o.value)) })`);
    check(`CSS page: self-test passes, model.css is from model.js, plays the latest rules ${LATEST}, and the root model.js is not loaded`,
      !css.failed && css.id === modelId(shipped) && css.version === LATEST && css.attr === LATEST && !css.loadsModelJs, JSON.stringify(css));
    check('CSS page: speed options 1x to 4096x', css.speeds.join() === SPEEDS, css.speeds.join());
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
    const cssWarn = await evaluate(`({ warned: ${warnRe}.test($status.textContent), status: $status.textContent })`);
    check(`CSS page: rules warning in the status line exactly when model.css is for other rules (root ${shipped.gameVersion})`,
      cssWarn.warned === !shippedLatest, JSON.stringify(cssWarn));
    check('no console errors, warnings or exceptions on the CSS page', problems.length === 0, problems.join('; '));
    for (const v of GAME_VERSIONS.filter(v => v !== LATEST)) {
      await load(path.join(ROOT, 'index-css.html') + `?game=${v}`);
      await waitUntil(`!!game || !!failed`, 5000);
      const asked = await evaluate(`({ failed: failed && failed.message, v: game && game.version, attr: document.documentElement.dataset.gameVersion })`);
      check(`CSS page: ?game=${v} is ignored: plays rules ${LATEST}`, !asked.failed && asked.v === LATEST && asked.attr === LATEST &&
        await waitUntil(`game.state === 'running' && game.score() > 0`, 5000), JSON.stringify(asked));
    }
    check('no console errors, warnings or exceptions on the ?game= CSS pages', problems.length === 0, problems.join('; '));

    // ---- 6. Model selector and speed (both pages) --------------------------
    const bundled = require(path.join(ROOT, 'models.js'));
    const index = require(path.join(ROOT, 'models-css', 'index.js'));
    const same = (a, b) => a.layers.join() === b.layers.join() && a.params.length === b.params.length && a.params.every((v, i) => v === b.params[i]);
    const shippedEntries = bundled.filter(m => m.shipped);
    check('models.js: exactly one shipped entry, and it is the root model.js', shippedEntries.length === 1 && same(shippedEntries[0], shipped) &&
      shippedEntries[0].source === shipped.source, shippedEntries.map(m => m.id).join());
    check('root model.css still matches model.js', fs.readFileSync(path.join(ROOT, 'model.css'), 'utf8').includes(`--nn-model-id: "${modelId(shipped)}"`));
    check('models-css/index.js lists the same models as models.js, with each CSS file\'s id',
      index.length === bundled.length && index.every((m, i) => m.id === bundled[i].id && m.cssId === modelId(bundled[i]) &&
        fs.readFileSync(path.join(ROOT, m.css), 'utf8').includes(`--nn-model-id: "${m.cssId}"`)), index.map(m => m.id).join());

    // Only latest-rules models are offered. The root model appears as its models.js
    // entry when it is for the latest rules, else as its own '' (model.js / model.css) entry.
    const latest = bundled.filter(m => m.gameVersion === LATEST);
    const others = bundled.filter(m => m.gameVersion !== LATEST);
    const rootSelect = shippedLatest ? shippedEntries[0].id : '';
    const wantOptions = (shippedLatest ? [] : ['']).concat(latest.map(m => m.id)).join();
    const optionsOf = (sel) => `[...${sel}.options].map(o => o.value).join()`;

    // Defaults with nothing remembered: the shipped model on both pages.
    await evaluate(`localStorage.removeItem('dino-model')`);
    await load(path.join(ROOT, 'index.html'));
    const def = await evaluate(`({ shipped: MODEL === DINO_MODEL, select: modelSelect.value, options: ${optionsOf('modelSelect')},
      speeds: [...speedSelect.options].map(o => Number(o.value)), speedDisabled: speedSelect.disabled, version: game.version })`);
    check(`index.html: default is the shipped model (${shippedLatest ? 'its models.js entry' : 'its own model.js entry'}); the selector lists exactly the rules-${LATEST} models`,
      def.shipped && def.select === rootSelect && def.options === wantOptions && def.version === LATEST, `${JSON.stringify(def)} want options ${wantOptions}`);
    check('index.html: speed options 1x to 4096x, disabled while AI is off', def.speeds.join() === SPEEDS && def.speedDisabled, def.speeds.join());
    // Network view: idle with AI off, under the game, and its show/hide toggle.
    const netInfo = `({ arch: document.getElementById('net-arch').textContent, live: netView.live, decisions: netView.decisions, draws: netView.draws,
      mismatches: netView.mismatches, warnHidden: document.getElementById('net-warn').hidden, caption: document.getElementById('net-caption').textContent,
      width: document.getElementById('net-canvas').clientWidth, height: document.getElementById('net-canvas').clientHeight,
      below: document.getElementById('net-canvas').getBoundingClientRect().top >= document.getElementById('game').getBoundingClientRect().bottom })`;
    const arch = (m) => m.layers.join('-') + ' · ';
    await sleep(300);
    const net0 = await evaluate(netInfo);
    check('index.html: network view under the game shows the shipped model\'s shape, idle while AI is off',
      net0.arch.startsWith(arch(shipped)) && !net0.live && net0.decisions === 0 && /turn AI on/.test(net0.caption) && net0.below &&
      net0.width > 0 && net0.height > 0 && net0.draws > 0, JSON.stringify(net0));
    const netToggle = async () => {
      const r = await evaluate(`(() => { const b = document.getElementById('net-toggle'); b.scrollIntoView({ block: 'center' }); const r = b.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
      await click(r.x, r.y);
      await sleep(50);
      return evaluate(`({ hidden: document.getElementById('net-body').hidden, text: document.getElementById('net-toggle').textContent,
        stored: localStorage.getItem('dino-netview'), state: game.state })`);
    };
    const hid = await netToggle();
    const shownAgain = await netToggle();
    await evaluate(`window.scrollTo(0, 0)`);
    check('index.html: the network toggle hides and shows the view, remembers it, and never presses jump',
      hid.hidden && hid.text === 'Network: hidden' && hid.stored === 'hidden' && !shownAgain.hidden && shownAgain.text === 'Network: shown' &&
      shownAgain.stored === 'shown' && hid.state === 'idle' && shownAgain.state === 'idle', `${JSON.stringify(hid)} ${JSON.stringify(shownAgain)}`);
    await load(path.join(ROOT, 'index-css.html'));
    await waitUntil(`!!game || !!failed`, 5000);
    const cdef = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), href: document.getElementById('model-css').getAttribute('href'),
      select: $model.value, options: ${optionsOf('$model')} })`);
    check(`index-css.html: default is model.css (${shippedLatest ? 'shown as the shipped entry' : 'its own entry'}); the selector lists exactly the rules-${LATEST} models`,
      !cdef.failed && cdef.href === 'model.css' && cdef.id === modelId(shipped) && cdef.select === rootSelect && cdef.options === wantOptions,
      `${JSON.stringify(cdef)} want options ${wantOptions}`);

    // A pick switches the model in place on both pages: no reload (a page marker
    // survives), the same game and run continue, the new model decides from then on,
    // and the choice is remembered (?model=<id> only, via history.replaceState; a
    // stray ?game= is dropped) for both pages' next visit.
    const pick = latest.find(m => !m.shipped);
    const pick2 = pick && latest.find(m => m.id !== pick.id);
    if (!pick || !pick2) {
      check(`two rules-${LATEST} models to pick in models.js`, false, 'models.js needs a non-shipped latest-rules model and a second one; the switch checks need them');
    } else {
      const pickJs = (id) => `(() => { modelSelect.value = '${id}'; modelSelect.dispatchEvent(new Event('change')); })()`;
      const remembered = `({ search: location.search, stored: localStorage.getItem('dino-model'), marker: window.__marker === 1 })`;
      // index.html, AI off: only the model changes; turning AI on then plays the pick.
      await evaluate(`localStorage.removeItem('dino-model')`);
      await load(path.join(ROOT, 'index.html') + `?game=${GAME_VERSIONS[0]}`);
      await evaluate(`(() => { window.__marker = 1; window.__made = [];
        const create = DinoPolicy.createAgent; DinoPolicy.createAgent = (m) => { __made.push(m.id || m.source); return create(m); }; })()`);
      await evaluate(pickJs(pick.id));
      const off = await evaluate(`({ ...${remembered}, id: model.id, loadTime: MODEL === DINO_MODEL, agent: agent === null, aiOn, state: game.state, select: modelSelect.value,
        net: ${netInfo}, made: __made.length })`);
      check(`index.html, AI off: a pick switches to ${pick.id} in place (no reload, no agent), the idle view shows its shape, remembered as ?model=<id> only`,
        off.marker && off.id === pick.id && off.loadTime && off.agent && !off.aiOn && off.made === 0 && off.select === pick.id &&
        off.net.arch.startsWith(arch(pick)) && !off.net.live && /turn AI on/.test(off.net.caption) &&
        off.search === `?model=${pick.id}` && off.stored === pick.id, JSON.stringify(off));
      const tb = await evaluate(`(() => { const r = document.getElementById('ai-toggle').getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
      await click(tb.x, tb.y);
      await sleep(50);
      check(`index.html: turning AI on after the pick plays ${pick.id}`, await evaluate(`aiOn && __made.length === 1 && __made[0] === '${pick.id}' && model.id === '${pick.id}'`),
        JSON.stringify(await evaluate(`({ aiOn, made: __made })`)));
      // index.html, AI on at 4096x: switch mid-run to pick2.
      await evaluate(`(() => {
        window.__s = { updates: 0, runs: 0, running: 0 };
        const update = game.update; let last = game.state;
        game.update = (ms, n) => { __s.updates++; if (last !== 'running' && game.state === 'running') __s.runs++; if (game.state === 'running') __s.running++;
          const r = update(ms, n); last = game.state; return r; };
        speedSelect.value = '4096'; })()`);
      const running = await waitUntil(`game.state === 'running' && __s.running >= ${MODEL_FRAMES}`, 20000);
      const before = await evaluate(`(() => { window.__agent0 = agent; return { updates: __s.updates, runs: __s.runs, score: game.score(), mismatches: netView.mismatches }; })()`);
      await evaluate(pickJs(pick2.id));
      const now = await evaluate(`({ ...${remembered}, id: model.id, newAgent: agent !== __agent0, made: __made[__made.length - 1], updates: __s.updates, runs: __s.runs })`);
      const goes = await waitUntil(`__s.updates >= ${now.updates + MODEL_FRAMES} && netView.decisions > 0`, 20000);
      const after = await evaluate(`({ updates: __s.updates, runs: __s.runs, state: game.state, score: game.score(), net: ${netInfo} })`);
      check(`index.html, AI on at 4096x: a mid-run pick switches ${pick.id} -> ${pick2.id} in place: no reload, a new agent for ${pick2.id}, remembered as ?model=<id>`,
        running && now.marker && now.id === pick2.id && now.newAgent && now.made === (pick2.id) && now.search === `?model=${pick2.id}` && now.stored === pick2.id,
        `${JSON.stringify(before)} ${JSON.stringify(now)}`);
      check(`index.html: the same run continues after the switch (${MODEL_FRAMES}+ more updates, score rising, run counter unchanged)`,
        goes && after.runs === before.runs && after.state === 'running' && after.score > before.score && after.updates > before.updates,
        `${JSON.stringify(before)} ${JSON.stringify(after)}`);
      check(`index.html: after the switch the network view shows ${pick2.layers.join('-')}, live, no mismatch warning, no console problems`,
        after.net.arch.startsWith(arch(pick2)) && after.net.live && after.net.mismatches === 0 && after.net.warnHidden && problems.length === 0,
        `${JSON.stringify(after.net)} ${problems.join('; ')}`);

      // index-css.html, started on the root model.css (nothing remembered).
      const modelLinks = `[...document.querySelectorAll('link[rel=stylesheet]')].filter(l => /^(model\\.css|models-css\\/)/.test(l.getAttribute('href')))`;
      const cssState = `({ ...${remembered}, failed: failed && failed.message, id: cssString('--nn-model-id'), links: ${modelLinks}.map(l => l.getAttribute('href') + (l.id ? '#' + l.id : '')),
        href: document.getElementById('model-css').getAttribute('href'), text: $modelCssLink.textContent, select: $model.value, pending: !!pending,
        sameGame: game === window.__g0, frame: game.frameIndex, state: game.state, status: $status.textContent, warn: !!$status.querySelector('.warn') })`;
      await evaluate(`localStorage.removeItem('dino-model')`);
      await load(path.join(ROOT, 'index-css.html') + `?game=${GAME_VERSIONS[0]}`);
      await waitUntil(`!!game || !!failed`, 5000);
      await evaluate(`(() => { window.__marker = 1; window.__g0 = game; })()`);
      // A pick whose --nn-model-id is already active (the root model.css holds the
      // shipped entry's model): nothing loads, the choice is remembered.
      const rootEntry = latest.find(m => m.cssId === modelId(shipped));
      if (!rootEntry) {
        console.log('(the root model.css is no latest-rules entry: same-cssId pick check skipped)');
      } else {
        await evaluate(`(() => { $model.value = '${rootEntry.id}'; $model.dispatchEvent(new Event('change')); })()`);
        await sleep(200);
        const same = await evaluate(cssState);
        check(`index-css.html: a pick of ${rootEntry.id}, whose --nn-model-id the root model.css already reports, loads nothing and is remembered`,
          same.marker && !same.failed && !same.pending && same.links.join() === 'model.css#model-css' && same.href === 'model.css' && same.id === rootEntry.cssId &&
          same.sameGame && same.search === `?model=${rootEntry.id}` && same.stored === rootEntry.id && problems.length === 0, JSON.stringify(same));
      }
      await evaluate(`$speed.value = '4096'`);
      await waitUntil(`game.state === 'running' && game.frameIndex >= ${MODEL_FRAMES}`, 20000);
      const c0 = await evaluate(cssState);
      await evaluate(`(() => { $model.value = '${pick.id}'; $model.dispatchEvent(new Event('change')); })()`);
      const switched = await waitUntil(`document.getElementById('model-css').getAttribute('href') === '${pick.css}'`, 15000);
      const c1 = await evaluate(cssState);
      check(`index-css.html at 4096x: a mid-run pick switches to ${pick.css} in place: no reload, one model stylesheet, --nn-model-id ${pick.cssId}, remembered as ?model=<id>, link updated`,
        switched && c1.marker && !c1.failed && !c1.pending && c1.id === pick.cssId && c1.links.join() === `${pick.css}#model-css` && c1.text === pick.css &&
        c1.select === pick.id && c1.search === `?model=${pick.id}` && c1.stored === pick.id && !c1.warn, `${JSON.stringify(c0)} ${JSON.stringify(c1)}`);
      // From now on every decision must be the pick's (policy.js on its weights).
      await evaluate(`(() => { const net = DinoPolicy.createNet(${JSON.stringify(pick.layers)}, ${JSON.stringify(pick.params)});
        const cssOnly = cssDecide; window.__cmp = { frames: 0, disagree: 0 };
        window.cssDecide = (x) => { const a = cssOnly(x); __cmp.frames++; if (net(x) !== a) __cmp.disagree++; return a; }; })()`);
      const cplayed = await waitUntil(`__cmp.frames >= ${MODEL_FRAMES}`, 30000);
      const c2 = await evaluate(`({ ...${cssState}, cmp: __cmp, net: ${netInfo} })`);
      check(`index-css.html: the same game continues (frameIndex ${c0.frame} -> ${c2.frame}), ${MODEL_FRAMES}+ decisions after the switch, 0 disagreements with ${pick.id}`,
        cplayed && c2.sameGame && c1.sameGame && c2.frame > c1.frame && c1.frame > c0.frame && c2.cmp.disagree === 0 && !c2.failed,
        `${JSON.stringify(c2.cmp)} frames ${c0.frame}/${c1.frame}/${c2.frame} sameGame ${c1.sameGame}/${c2.sameGame}`);
      check(`index-css.html: after the switch the network view shows ${pick.layers.join('-')}, JS argmax = --act (no warning), no console problems`,
        c2.net.arch.startsWith(arch(pick)) && c2.net.live && c2.net.mismatches === 0 && c2.net.warnHidden && problems.length === 0,
        `${JSON.stringify(c2.net)} ${problems.join('; ')}`);
      // A pick whose stylesheet is missing: the old model keeps playing, the selector
      // goes back, the status line warns, and the page logs nothing (the browser's own
      // failed-load entry for that file is the only problem allowed).
      const missing = 'models-css/no-such-model.css';
      await evaluate(`(() => { MODEL_INDEX.push({ ...MODEL_INDEX.find(m => m.id === '${pick.id}'), id: 'no-such-model', label: 'missing', css: '${missing}', cssId: 'na-missing' });
        $model.add(new Option('missing', 'no-such-model')); $model.value = 'no-such-model'; $model.dispatchEvent(new Event('change')); })()`);
      const aborted = await waitUntil(`!pending && /model not switched/.test($status.textContent)`, 15000);
      const c3 = await evaluate(cssState);
      const own = problems.filter(p => !(p.startsWith('log error: ') && p.includes('no-such-model.css')));
      check('index-css.html: a pick whose stylesheet is missing keeps the old model, restores the selector, warns in the status line, and logs nothing itself',
        aborted && !c3.failed && c3.id === pick.cssId && c3.links.join() === `${pick.css}#model-css` && c3.select === pick.id && c3.stored === pick.id &&
        c3.search === `?model=${pick.id}` && c3.warn && /could not load/.test(c3.status) && c3.sameGame && own.length === 0,
        `${JSON.stringify(c3)} problems: ${problems.join('; ')}`);
      await waitUntil(`game.frameIndex > ${c3.frame}`, 5000);
      problems.splice(0, problems.length, ...own);
      check('index-css.html: the old model keeps playing after the failed switch', (await evaluate(`game.frameIndex`)) > c3.frame && !(await evaluate(`failed`)));
      // The remembered pick, on both pages' next visit without ?model=.
      await load(path.join(ROOT, 'index-css.html'));
      await waitUntil(`!!game || !!failed`, 15000);
      const back = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), select: $model.value })`);
      await load(path.join(ROOT, 'index.html'));
      const backJs = await evaluate(`({ id: MODEL.id, select: modelSelect.value })`);
      check(`both pages play the remembered pick ${pick.id} without ?model=`, !back.failed && back.id === pick.cssId && back.select === pick.id &&
        backJs.id === pick.id && backJs.select === pick.id, `${JSON.stringify(back)} ${JSON.stringify(backJs)}`);
      await evaluate(`localStorage.removeItem('dino-model')`);
    }

    // A model for other rules, asked for with ?model= and remembered: both pages
    // fall back to the root model, still play the latest rules, and forget the choice.
    if (!others.length) {
      console.log(`(no models for rules other than ${LATEST} in models.js: fallback checks skipped)`);
    } else {
      const old = others[0];
      for (const how of ['?model=', 'remembered', '?model= and remembered']) {
        const url = how.startsWith('?model=') ? `?model=${old.id}` : '';
        const remember = how.includes('remembered');
        await evaluate(remember ? `localStorage.setItem('dino-model', '${old.id}')` : `localStorage.removeItem('dino-model')`);
        await load(path.join(ROOT, 'index.html') + url);
        const j = await evaluate(`({ root: MODEL === DINO_MODEL, version: game.version, select: modelSelect.value, stored: localStorage.getItem('dino-model') })`);
        check(`index.html, rules-${old.gameVersion} model ${old.id} (${how}): falls back to the root model on rules ${LATEST}${remember ? ', remembered choice cleared' : ''}`,
          j.root && j.version === LATEST && j.select === rootSelect && j.stored === null && problems.length === 0, `${JSON.stringify(j)} ${problems.join('; ')}`);
        await evaluate(remember ? `localStorage.setItem('dino-model', '${old.id}')` : `localStorage.removeItem('dino-model')`);
        await load(path.join(ROOT, 'index-css.html') + url);
        await waitUntil(`!!game || !!failed`, 15000);
        const c = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), href: document.getElementById('model-css').getAttribute('href'),
          version: game && game.version, select: $model.value, stored: localStorage.getItem('dino-model') })`);
        check(`index-css.html, rules-${old.gameVersion} model ${old.id} (${how}): falls back to model.css on rules ${LATEST}${remember ? ', remembered choice cleared' : ''}`,
          !c.failed && c.href === 'model.css' && c.id === modelId(shipped) && c.version === LATEST && c.select === rootSelect && c.stored === null &&
          problems.length === 0, `${JSON.stringify(c)} ${problems.join('; ')}`);
      }

      // A root model for other rules (a scratch copy of both pages whose model.js /
      // model.css hold `old`): the pages still play the latest rules and warn.
      const oddDir = path.join(scratch, 'other-rules-root');
      for (const f of ['index.html', 'index-css.html', 'style.css', 'core.js', 'pages.js', 'render.js', 'policy.js', 'game.js', 'css-nn.js',
        'netview.js', 'models.js', 'models-css/index.js']) {
        fs.mkdirSync(path.dirname(path.join(oddDir, f)), { recursive: true });
        fs.copyFileSync(path.join(ROOT, f), path.join(oddDir, f));
      }
      fs.writeFileSync(path.join(oddDir, 'model.js'), `window.DINO_MODEL = ${JSON.stringify(old)};\n`);
      fs.copyFileSync(path.join(ROOT, old.css), path.join(oddDir, 'model.css'));
      await evaluate(`localStorage.removeItem('dino-model')`);
      await load(path.join(oddDir, 'index.html'));
      await sleep(1100);
      const oj = await evaluate(`({ ...${versionInfo}, root: MODEL === DINO_MODEL, select: modelSelect.value, options: ${optionsOf('modelSelect')} })`);
      check(`index.html with a rules-${old.gameVersion} root model: plays rules ${LATEST}, warns in the status line, lists the root as its own entry`,
        oj.v === LATEST && oj.attr === LATEST && oj.warned && oj.root && oj.select === '' && oj.options === ['', ...latest.map(m => m.id)].join() &&
        problems.length === 0, `${JSON.stringify(oj)} ${problems.join('; ')}`);
      await load(path.join(oddDir, 'index-css.html'));
      await waitUntil(`(!!game && /trained for/.test($status.textContent)) || !!failed`, 5000);
      const oc = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), v: game && game.version, attr: document.documentElement.dataset.gameVersion,
        warned: ${warnRe}.test($status.textContent), status: $status.textContent, select: $model.value, options: ${optionsOf('$model')} })`);
      check(`index-css.html with a rules-${old.gameVersion} root model.css: plays rules ${LATEST}, warns in the status line, lists the root as its own entry`,
        !oc.failed && oc.id === old.cssId && oc.v === LATEST && oc.attr === LATEST && oc.warned && oc.select === '' &&
        oc.options === ['', ...latest.map(m => m.id)].join() && problems.length === 0, `${JSON.stringify(oc)} ${problems.join('; ')}`);
    }

    // Every latest-rules bundled model on both pages, at 4096x.
    const speedRows = [];
    for (const m of latest) {
      await load(path.join(ROOT, 'index.html') + `?model=${m.id}`);
      const page = await evaluate(`({ id: MODEL.id, version: game.version, select: modelSelect.value, hiKey: HI_KEY })`);
      await evaluate(`(() => {
        window.__m = { applyKeys: 0, updates: 0, running: 0 };
        const apply = game.applyKeys; game.applyKeys = (k, n) => { if (aiOn) __m.applyKeys++; return apply(k, n); };
        const update = game.update; game.update = (ms, n) => { if (aiOn) { __m.updates++; if (game.state === 'running') __m.running++; } return update(ms, n); };
      })()`);
      const tb = await evaluate(`(() => { const r = document.getElementById('ai-toggle').getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
      await click(tb.x, tb.y);   // a real click (a user gesture, so the AudioContext may start)
      await evaluate(`speedSelect.value = '4096'`);
      const played = await waitUntil(`__m.running >= ${MODEL_FRAMES}`, 20000);
      await sleep(2500);   // two full rate windows at 4096x
      const js = await evaluate(`({ ...__m, achieved: Math.round(rate.speed), status: speedStatus.textContent })`);
      check(`index.html ?model=${m.id}: plays rules ${m.gameVersion} and ${MODEL_FRAMES}+ frames at 4096x, policy once per update`,
        page.id === m.id && page.select === m.id && page.version === m.gameVersion && played && js.applyKeys === js.updates && problems.length === 0,
        `${JSON.stringify(page)} ${JSON.stringify(js)} ${problems.length} problems ${problems.slice(0, 3).join('; ')}`);
      const jnet = await evaluate(netInfo);
      check(`index.html ?model=${m.id}: network view shows ${m.layers.join('-')} and draws live AI decisions at 4096x, no mismatch warning`,
        jnet.arch.startsWith(arch(m)) && jnet.live && jnet.decisions > 0 && jnet.draws > 0 && jnet.mismatches === 0 && jnet.warnHidden, JSON.stringify(jnet));

      await load(path.join(ROOT, 'index-css.html') + `?model=${m.id}`);
      await waitUntil(`!!game || !!failed`, 15000);
      const cpage = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), version: game && game.version, select: $model.value,
        loadsWeightsJs: typeof DINO_MODEL !== 'undefined' })`);
      await evaluate(`(() => { const net = DinoPolicy.createNet(${JSON.stringify(m.layers)}, ${JSON.stringify(m.params)});
        const cssOnly = cssDecide; window.__cmp = { frames: 0, disagree: 0 };
        window.cssDecide = (x) => { const a = cssOnly(x); __cmp.frames++; if (net(x) !== a) __cmp.disagree++; return a; };
        $speed.value = '4096'; })()`);
      const cplayed = await waitUntil(`__cmp.frames >= ${MODEL_FRAMES}`, 30000);
      await sleep(2500);
      const c = await evaluate(`({ ...__cmp, achieved: Math.round(rate.fps * FRAME_MS / 1000), status: $status.textContent })`);
      check(`index-css.html ?model=${m.id}: models-css/${m.id}.css loads, plays rules ${m.gameVersion}, 0 CSS/JS disagreements over ${MODEL_FRAMES}+ decisions at 4096x`,
        !cpage.failed && cpage.id === m.cssId && cpage.version === m.gameVersion && cpage.select === m.id && !cpage.loadsWeightsJs &&
        cplayed && c.disagree === 0 && problems.length === 0, `${JSON.stringify(cpage)} frames ${c.frames} disagree ${c.disagree} ${problems.length} problems ${problems.slice(0, 3).join('; ')}`);
      const cnet = await evaluate(netInfo);
      check(`index-css.html ?model=${m.id}: network view shows ${m.layers.join('-')} and draws CSS decisions at 4096x, JS argmax = --act (no warning)`,
        cnet.arch.startsWith(arch(m)) && cnet.live && cnet.decisions > 0 && cnet.draws > 0 && cnet.mismatches === 0 && cnet.warnHidden && cnet.below, JSON.stringify(cnet));
      speedRows.push({ id: m.id, jsFrames: js.running, jsAchieved: js.achieved, cssFrames: c.frames, cssDisagree: c.disagree, cssAchieved: c.achieved });
    }
    console.log('achieved speed at 4096x (simulated / real time):');
    for (const r of speedRows) console.log(`  ${r.id.padEnd(22)} index.html ${String(r.jsAchieved).padStart(5)}x   index-css.html ${String(r.cssAchieved).padStart(4)}x   (CSS ${r.cssDisagree}/${r.cssFrames} disagreements)`);

    // ---- 7. Phone width: nothing outside the screen, the network view fits ---
    // Every element box must lie inside [0, width]: a centred element that is too wide
    // overflows to the left too, which scrollWidth alone never shows (a long model name
    // in the selector did, 2026-10-12), and a too-wide page widens the layout viewport.
    for (const width of [320, 375]) {
      await send('Emulation.setDeviceMetricsOverride', { width, height: 800, deviceScaleFactor: 2, mobile: true });
      for (const page of ['index.html', 'index-css.html']) {
        await load(path.join(ROOT, page) + (latest.length ? `?model=${latest[latest.length - 1].id}` : ''));
        await waitUntil(`typeof game !== 'undefined' && !!game`, 15000);
        await sleep(300);
        const phone = await evaluate(`({ scroll: document.documentElement.scrollWidth, inner: innerWidth,
          net: document.getElementById('net-canvas').getBoundingClientRect().width, draws: netView.draws,
          outside: [...document.querySelectorAll('body *')].map(el => [el, el.getBoundingClientRect()])
            .filter(([, b]) => b.width && (b.left < -0.5 || b.right > innerWidth + 0.5))
            .map(([el, b]) => (el.id ? '#' + el.id : el.tagName.toLowerCase()) + ' ' + Math.round(b.left) + '..' + Math.round(b.right)) })`);
        check(`${page} at ${width}px: every element on screen, no horizontal scroll, network view fits and draws`,
          phone.inner === width && phone.scroll <= phone.inner && phone.outside.length === 0 && phone.net > 0 && phone.net <= phone.inner && phone.draws > 0 && problems.length === 0, JSON.stringify(phone));
      }
    }
    await send('Emulation.clearDeviceMetricsOverride');
  } finally {
    cdp.close();
    chrome.kill();
  }
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
