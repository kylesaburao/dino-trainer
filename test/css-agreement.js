'use strict';
// CSS/JS decision agreement for a CANDIDATE model, through the pterodactyl and
// duck phase (test/browser-check.js section 5 covers the root model's opening only).
//
//   node test/css-agreement.js --dir runs/<run>          (its best.json)
//   node test/css-agreement.js --model path/to/model.js
//     [--min-ducks 50] [--min-score 3000] [--max-frames 2000000] [--timeout 900]
//     [--seed 12345] [--start-score S] [--speed 2048]
//
// The candidate is exported with the real export path (`trainer/cli.js export
// --out` or `export-css` on a copy) into a scratch directory, next to copies of
// index-css.html and its scripts, and played there over file:// in one headless
// Chrome. The root model.js / model.css are never read or written.
//
// On every decision the page's CSS network makes, the same observation also goes
// through policy.js createNet on the candidate's params (the JS decision), and a
// float64 forward pass gives the top-2 logit margin. Mid pterodactyls (box overlaps
// the standing box vertically, clears the ducking box) are classified as in
// research/scripts/prereg-teacher-duck-step2/duckshare.js: over the frames (state
// after each step) when the box overlaps the standing box's x-range, ducked =
// grounded and ducking on every one, jumped = airborne on every one, else mixed;
// crashed = the run ended during the overlap.
// Plays until >= --min-ducks ducked mid pterodactyls and a run reached --min-score,
// or --max-frames compared decisions, or --timeout seconds.
// Pass (exit 0): 0 disagreements, >= --min-ducks frames whose JS action is duck,
// and the play target reached. Prints a JSON summary line last.
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { seedMathRandom, launch } = require('./cdp.js');

const ROOT = path.join(__dirname, '..');
// models-css/index.js only lists the bundled models for the selector; with no
// ?model= and nothing remembered (a fresh profile), the page plays model.css.
// models.js and netview.js are the network view's; a candidate that is not in
// models.js gets no view (and no view check), the decisions are still CSS.
// The shipped page always plays the latest rules (pages.js GAME_VERSION); a candidate
// trained for other rules gets that line replaced in the scratch copy only, so it
// plays its own rules.
const PAGE_FILES = ['index-css.html', 'style.css', 'core.js', 'pages.js', 'render.js', 'policy.js', 'css-nn.js', 'models-css/index.js', 'models.js', 'netview.js'];
const VERSION_LINE = /^const GAME_VERSION = .*;$/m;

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const dir = opt('--dir', null), modelArg = opt('--model', null);
if (!!dir === !!modelArg) { console.error('give exactly one of --dir <run> or --model <model.js>'); process.exit(2); }
const MIN_DUCKS = Number(opt('--min-ducks', 50));
const MIN_SCORE = Number(opt('--min-score', 3000));
const MAX_FRAMES = Number(opt('--max-frames', 2000000));
const TIMEOUT_S = Number(opt('--timeout', 900));
const SEED = Number(opt('--seed', 12345));
const START_SCORE = opt('--start-score', null);
const SPEED = String(opt('--speed', 2048));

// Builds the scratch page; returns { scratch, pageDir, model }.
function preparePage() {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'dino-css-agree-'));
  const pageDir = path.join(scratch, 'page');
  fs.mkdirSync(pageDir);
  const out = path.join(pageDir, 'model.js');
  const cli = path.join(ROOT, 'trainer', 'cli.js');
  if (dir) {
    execFileSync(process.execPath, [cli, 'export', '--dir', path.resolve(dir), '--which', 'best', '--out', out], { cwd: ROOT, stdio: 'inherit' });
  } else {
    fs.copyFileSync(path.resolve(modelArg), out);
    execFileSync(process.execPath, [cli, 'export-css', '--model', out], { cwd: ROOT, stdio: 'inherit' });
  }
  if (!fs.existsSync(path.join(pageDir, 'model.css'))) throw new Error('export did not write model.css into the scratch dir');
  for (const f of PAGE_FILES) {
    fs.mkdirSync(path.dirname(path.join(pageDir, f)), { recursive: true });
    fs.copyFileSync(path.join(ROOT, f), path.join(pageDir, f));
  }
  const model = require(out);
  const want = model.gameVersion || 'v1';
  if (want !== require(path.join(ROOT, 'pages.js')).GAME_VERSION) {
    const file = path.join(pageDir, 'pages.js'), src = fs.readFileSync(file, 'utf8');
    if (!VERSION_LINE.test(src)) throw new Error('pages.js has no GAME_VERSION line to set the candidate\'s rules in the scratch copy');
    fs.writeFileSync(file, src.replace(VERSION_LINE, `const GAME_VERSION = ${JSON.stringify(want)};`));
    console.error(`scratch page set to the candidate's rules ${want} (the shipped pages play the latest rules)`);
  }
  return { scratch, pageDir, model };
}

// Test-only page instrumentation (evaluated after the page has started).
function instrumentation(model) {
  return `(() => {
  const L = ${JSON.stringify(model.layers)}, p = ${JSON.stringify(Array.from(model.params))};
  const net = DinoPolicy.createNet(L, p);
  const bufs = L.map(n => new Float64Array(n));
  const logits = (x) => {
    let input = x, k = 0;
    for (let l = 1; l < L.length; l++) {
      const nIn = L[l - 1], nOut = L[l], out = bufs[l], bBase = k + nOut * nIn, last = l === L.length - 1;
      for (let i = 0; i < nOut; i++) {
        let s = p[bBase + i];
        const row = k + i * nIn;
        for (let j = 0; j < nIn; j++) s += p[row + j] * input[j];
        out[i] = last ? s : Math.tanh(s);
      }
      input = out; k = bBase + nOut;
    }
    return input;
  };
  const C = window.__cmp = { frames: 0, disagree: 0, byJs: [0, 0, 0], byCss: [0, 0, 0], confusion: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    minMargin: Infinity, minMarginByJs: [Infinity, Infinity, Infinity], argmaxMismatch: 0, firstDisagreements: [],
    games: 0, maxScore: 0, mid: { ducked: 0, jumped: 0, mixed: 0, crashed: 0 } };
  const cssOnly = cssDecide;
  window.cssDecide = (x) => {
    const a = cssOnly(x), j = net(x), o = logits(x);
    let b = 0; for (let i = 1; i < o.length; i++) if (o[i] > o[b]) b = i;
    if (b !== j) C.argmaxMismatch++;
    const s = Array.from(o).sort((u, v) => v - u), m = s[0] - s[1];
    C.frames++; C.byJs[j]++; C.byCss[a]++; C.confusion[j][a]++;
    if (m < C.minMargin) C.minMargin = m;
    if (m < C.minMarginByJs[j]) C.minMarginByJs[j] = m;
    if (a !== j) { C.disagree++; if (C.firstDisagreements.length < 10) C.firstDisagreements.push({ js: j, css: a, margin: m, score: game.score(), logits: Array.from(o) }); }
    return a;
  };
  // Mid-pterodactyl classification on every step of every game (newGame replaced the game).
  const Core = DinoCore, stand = Core.standBox(Core.STAND_TOP), duckY = Core.GROUND - Core.DUCK_H + 4;
  const duckable = (b) => b.y < stand.y + stand.h && b.y + b.h > stand.y && b.y + b.h <= duckY;
  const overStand = (b) => b.x < stand.x + stand.w && b.x + b.w > stand.x;
  const close = (t) => { C.mid[t.crashed ? 'crashed' : t.ducked === t.overlap ? 'ducked' : t.air === t.overlap ? 'jumped' : 'mixed']++; };
  const instrument = (g) => {
    C.games++;
    const track = new Map(), step = g.step;
    g.step = (keys) => {
      step(keys);
      C.maxScore = Math.max(C.maxScore, g.score());
      if (g.state === 'over') {
        for (const [o, t] of track) if (!t.done) { t.done = true; t.crashed = overStand(Core.obstacleBox(o)); close(t); }
        return;
      }
      if (g.state !== 'running') return;
      for (const o of g.obstacles) {
        if (o.t.name !== 'ptero') continue;
        const b = Core.obstacleBox(o);
        if (!duckable(b)) continue;
        let t = track.get(o);
        if (overStand(b)) {
          if (!t) { t = { overlap: 0, ducked: 0, air: 0, done: false, crashed: false }; track.set(o, t); }
          t.overlap++;
          if (g.dino.jumping) t.air++; else if (g.dino.ducking) t.ducked++;
        } else if (t && !t.done && b.x + b.w <= stand.x) { t.done = true; close(t); }
      }
    };
  };
  ${START_SCORE !== null ? `
  // --start-score: every run starts later (core startScore option).
  window.newGame = () => {
    document.body.classList.remove('night');
    game = DinoCore.createGame({ rng: DinoCore.createRng((Math.random() * 4294967296) >>> 0), version: gameVersion,
      startScore: ${Number(START_SCORE)}, hooks: { night: (on) => document.body.classList.toggle('night', on) } });
    wait = 0; current = DinoPolicy.ACTIONS[0]; overAt = 0;
    instrument(game);
  };
  newGame();` : `
  const origNewGame = newGame;
  window.newGame = () => { origNewGame(); instrument(game); };
  instrument(game);`}
  document.getElementById('speed').value = '${SPEED}';
})()`;
}

(async () => {
  const { scratch, pageDir, model } = preparePage();
  const { modelId } = require(path.join(ROOT, 'trainer', 'css-model.js'));
  const browser = await launch(path.join(scratch, 'profile'), seedMathRandom(SEED));
  const { chrome, cdp, evaluate, load, waitUntil, problems } = browser;
  let summary;
  try {
    await load(path.join(pageDir, 'index-css.html'));
    const page = await evaluate(`({ failed: failed && failed.message, id: cssString('--nn-model-id'), version: game && game.version,
      state: game && game.state, frameIndex: game && game.frameIndex, loadsModelJs: typeof DINO_MODEL !== 'undefined' })`);
    const pageOk = !page.failed && page.id === modelId(model) && page.version === (model.gameVersion || 'v1') && !page.loadsModelJs;
    if (!pageOk) throw new Error(`CSS page not usable for this model: ${JSON.stringify(page)}`);
    await evaluate(instrumentation(model));
    const t0 = Date.now();
    let last = 0;
    const done = `(__cmp.mid.ducked >= ${MIN_DUCKS} && __cmp.maxScore >= ${MIN_SCORE}) || __cmp.frames >= ${MAX_FRAMES} || !!failed`;
    while (!(await evaluate(done)) && Date.now() - t0 < TIMEOUT_S * 1000) {
      await waitUntil(done, 10000, 500);
      if (Date.now() - last > 10000) {
        last = Date.now();
        const p = await evaluate(`({ frames: __cmp.frames, disagree: __cmp.disagree, duckFrames: __cmp.byJs[2], ducked: __cmp.mid.ducked, score: game.score(), fps: Math.round(rate.fps) })`);
        console.error(`${((Date.now() - t0) / 1000).toFixed(0)} s: ${JSON.stringify(p)}`);
      }
    }
    const r = await evaluate(`({ ...__cmp, failed: failed && failed.message, score: game.score(), fps: Math.round(rate.fps) })`);
    const reached = r.mid.ducked >= MIN_DUCKS && r.maxScore >= MIN_SCORE;
    const pass = reached && r.disagree === 0 && r.byJs[2] >= MIN_DUCKS && r.argmaxMismatch === 0 && !r.failed && problems.length === 0;
    summary = {
      pass, model: dir ? path.join(path.resolve(dir), 'best.json') : path.resolve(modelArg), layers: model.layers, modelId: page.id,
      seed: SEED, startScore: START_SCORE === null ? 0 : Number(START_SCORE), speed: Number(SPEED), seconds: (Date.now() - t0) / 1000,
      reached, frames: r.frames, disagree: r.disagree, duckFrames: r.byJs[2], byJs: r.byJs, byCss: r.byCss, confusion: r.confusion,
      disagreeByJsClass: r.confusion.map((row, j) => row.reduce((s, n, a) => s + (a === j ? 0 : n), 0)),
      minMargin: r.minMargin, minMarginByJs: r.minMarginByJs, argmaxMismatch: r.argmaxMismatch, midPteros: r.mid,
      games: r.games, maxScore: r.maxScore, failed: r.failed, problems: [...problems], firstDisagreements: r.firstDisagreements,
    };
  } finally {
    cdp.close();
    chrome.kill();
    await new Promise((res) => { chrome.once('exit', res); setTimeout(res, 3000); });
    try { fs.rmSync(scratch, { recursive: true, force: true, maxRetries: 5 }); } catch (e) { console.error(`scratch left at ${scratch}: ${e.message}`); }
  }
  console.log(JSON.stringify(summary));
  console.log(summary.pass ? 'PASS  CSS/JS agreement through the duck phase' : 'FAIL  CSS/JS agreement through the duck phase');
  process.exit(summary.pass ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
