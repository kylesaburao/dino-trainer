'use strict';

// CSS-network page (index-css.html): the AI dino with its MLP evaluated by the
// browser's style engine. model.css (generated from model.js by
// trainer/css-model.js) turns the inputs --x0..--x43 into the action --act.
// This script writes the observation (policy.js observe(), exactly what
// training saw) into those properties and reads --act back. There is no
// JavaScript fallback: if the CSS self-test fails, the dino doesn't play.
// The page never loads model.js, and every decision comes from the stylesheet.
//
// Network view (netview.js, under the game): once per display frame it draws the
// latest CSS decision: the activations from a JavaScript forward pass on the same
// --x inputs, with the weights of the same model taken from models.js (matched by
// --nn-model-id), and a ring on the action --act chose. If the forward pass's
// argmax differs from --act it shows a warning. It only reads; it never decides.
// A stylesheet that is not in models.js has no JavaScript weights, so no view.
//
// Model selector: ?model=<id>, else the choice remembered by either page
// (localStorage 'dino-model'), switches the stylesheet to models-css/<id>.css
// (`node trainer/cli.js export-models`; models-css/index.js lists ids, labels and
// each file's --nn-model-id, never weights). With neither, the page plays model.css.
// Only models trained for the latest rules are offered or accepted (pages.js).
//
// Rules: always the latest (DinoPages.GAME_VERSION); nothing on the page picks
// them. A stylesheet's --nn-game only checks consistency with its index entry,
// and a root model.css for other rules plays the latest rules with a warning.
const { FRAME_MS } = DinoCore;
const { OBS_DIM, ACTIONS, observe } = DinoPolicy;

const renderer = DinoRender.createRenderer(document.getElementById('game'));
const nn = document.getElementById('nn');
const $status = document.getElementById('status');
const $speed = document.getElementById('speed');
const $model = document.getElementById('model');
const $modelCss = document.getElementById('model-css');
const $modelCssLink = document.getElementById('model-css-link');

const MODEL_INDEX = DinoPages.forPages(typeof DINO_MODEL_INDEX !== 'undefined' ? DINO_MODEL_INDEX : []);
const SELECTED = DinoPages.chooseModel(MODEL_INDEX, location.search);
const MODEL_LOAD_MS = 10000;
const VIEW_MODELS = typeof DINO_MODELS !== 'undefined' ? DINO_MODELS : [];   // weights for the network view only

const STEP_BUDGET_MS = 12;     // simulation time per display frame before playback falls behind
const RESTART_MS = 1000;       // pause on the crash frame before the next run

// ---------------------------------------------------------------------------
// The CSS network
// ---------------------------------------------------------------------------
const cssString = (name) => getComputedStyle(nn).getPropertyValue(name).trim().replace(/^"(.*)"$/, '$1');

function cssNumber(name) {
  const v = nn.computedStyleMap ? nn.computedStyleMap().get(name) : null;
  const n = v && v.unit === 'number' ? v.value : Number(getComputedStyle(nn).getPropertyValue(name));
  if (!Number.isFinite(n)) throw new Error(`model.css produced no number for ${name}`);
  return n;
}

// Write the 44 inputs in one style mutation, then let the style engine decide.
function cssDecide(x) {
  let text = '';
  for (let i = 0; i < OBS_DIM; i++) text += `--x${i}:${x[i]};`;
  nn.style.cssText = text;
  const act = cssNumber('--act');
  if (!(act === 0 || act === 1 || act === 2)) throw new Error(`model.css chose action ${act}`);
  return act;
}

// model.css must be loaded and the browser must evaluate exp()/sign() in calc().
function selfTest() {
  if (!cssString('--nn-model-id')) throw new Error('model.css is not loaded');
  const probe = cssNumber('--nn-probe');
  if (Math.abs(probe - Math.tanh(1)) > 1e-9) throw new Error(`this browser cannot evaluate exp() in calc() (tanh(1) = ${probe})`);
  if (cssNumber('--nn-probe-sign') !== 0) throw new Error('this browser cannot evaluate sign() in calc()');
}

// ---------------------------------------------------------------------------
// Network view (netview.js); see the header
// ---------------------------------------------------------------------------
const netView = DinoNetView.createView(document.getElementById('net-canvas'), {
  arch: document.getElementById('net-arch'),
  warn: document.getElementById('net-warn'),
  tip: document.getElementById('net-tip'),
  font: 'monospace',
  warnText: (n) => `The JavaScript forward pass disagreed with the CSS network's --act on ${n} decision(s)`,
});
const netCaption = document.getElementById('net-caption');
const netPanel = DinoNetView.bindToggle(document.getElementById('net-toggle'), document.getElementById('net-body'), 'dino-netview');
netView.setModel('loading…');

// The weights of the stylesheet with --nn-model-id `loadedId`, from models.js.
function setViewModel(loadedId) {
  const m = VIEW_MODELS.find(e => e.cssId === loadedId);
  netView.setModel(m || 'no JavaScript weights for this stylesheet (not in models.js)');
  netCaption.textContent = m ? 'the latest CSS decision; activations from the JavaScript forward pass on the same --x inputs' : '';
}

// ---------------------------------------------------------------------------
// Game and agent
// ---------------------------------------------------------------------------
let gameVersion = DinoPages.GAME_VERSION, actionRepeat = 1;
document.documentElement.dataset.gameVersion = gameVersion;
let rulesWarning = '';   // set in start() when the loaded stylesheet is for other rules
const $rulesWarn = document.createElement('span');
$rulesWarn.className = 'warn';
let game = null, wait = 0, current = ACTIONS[0], overAt = 0, best = 0;
let lastAct = -1, decided = false;   // the latest CSS decision (on x), for the network view
const x = new Float64Array(OBS_DIM);

function newGame() {
  document.body.classList.remove('night');
  game = DinoCore.createGame({
    rng: DinoCore.createRng((Math.random() * 4294967296) >>> 0),
    version: gameVersion,
    hooks: { night: (on) => document.body.classList.toggle('night', on) },
  });
  wait = 0;
  current = ACTIONS[0];
  overAt = 0;
}

// Same cadence as policy.js createAgent(): jump to start from idle, otherwise a
// fresh decision every actionRepeat frames (here from CSS instead of createNet).
function act() {
  if (game.state === 'idle') return ACTIONS[1];
  if (game.state !== 'running') return ACTIONS[0];
  if (wait <= 0) {
    observe(game, x);
    lastAct = cssDecide(x);
    current = ACTIONS[lastAct];
    decided = true;
    wait = actionRepeat;
  }
  wait--;
  return current;
}

// ---------------------------------------------------------------------------
// Main loop: speed × real time, at most STEP_BUDGET_MS of simulation per frame
// ---------------------------------------------------------------------------
let last = 0, acc = 0, behind = false, failed = null;
const rate = { t0: 0, frames: 0, fps: 0 };   // simulated frames per real second

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(Math.max(now - last, 0), 100);
  last = now;
  if (failed || !game) return;
  try {
    if (game.state === 'over') {
      if (!overAt) overAt = now;
      if (now - overAt > RESTART_MS) newGame();
    } else {
      acc += dt * Number($speed.value);
      const deadline = performance.now() + STEP_BUDGET_MS;
      behind = false;
      while (acc >= FRAME_MS) {
        game.step(act());
        rate.frames++;
        acc -= FRAME_MS;
        if (game.state === 'over') { acc = 0; best = Math.max(best, game.score()); break; }
        if (performance.now() > deadline) { acc = 0; behind = true; break; }
      }
    }
  } catch (e) {
    fail(e);
    return;
  }
  if (now - rate.t0 >= 1000) {
    rate.fps = rate.frames * 1000 / (now - rate.t0);
    rate.t0 = now;
    rate.frames = 0;
  }
  renderer.draw(game, game.frameIndex * FRAME_MS, best);
  if (decided && netPanel.shown) {
    decided = false;
    netView.capture(x, lastAct, true);
  }
  netView.draw();
  showStatus();
}

// The achieved speed is measured (simulated frames per real second), not the
// selected one; at high speeds the style engine falls far short of it.
function showStatus() {
  const eff = rate.fps * FRAME_MS / 1000;
  const want = Number($speed.value);
  $status.textContent = `game ${game.version}${game.version !== 'v1' ? ` (level ${game.level.toFixed(2)}${game.superLevel > 0 ? `, v3 level ${game.superLevel.toFixed(2)}` : ''})` : ''} · score ${game.score()}` +
    ` · achieved ${eff >= 10 ? Math.round(eff) : eff.toFixed(1)}× real time` + (behind || eff < want * 0.9 ? ` of ${want}× (CSS can't keep up)` : '') +
    ` · ${Math.round(rate.fps / actionRepeat)} CSS decisions/s`;
  if (rulesWarning) $status.prepend($rulesWarn, ' · ');
}

// Fill the model selector (latest-rules entries only); picking a model remembers it
// and reloads with only ?model=<id>. A loaded stylesheet that is not among them (the
// root model.css when it holds another model) appears as its own "model.css" entry.
function fillModelSelect(loadedId) {
  const current = SELECTED || MODEL_INDEX.find(m => m.cssId === loadedId);
  if (!current) $model.add(new Option(`model.css (${loadedId})`, ''));
  for (const m of MODEL_INDEX) $model.add(new Option(m.label, m.id));
  $model.value = current ? current.id : '';
  $model.disabled = $model.options.length < 2;
  $model.addEventListener('change', () => DinoPages.pickModel($model.value));
}
for (const el of [$model, $speed]) el.addEventListener('change', () => el.blur());

// Switch the stylesheet to the selected model's and wait until the style engine
// reports that model's --nn-model-id.
function loadSelected() {
  if (!SELECTED) return Promise.resolve();
  $modelCss.href = SELECTED.css;
  $modelCssLink.href = SELECTED.css;
  $modelCssLink.textContent = SELECTED.css;
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    const poll = () => {
      const id = cssString('--nn-model-id');
      if (id === SELECTED.cssId) resolve();
      else if (performance.now() - t0 > MODEL_LOAD_MS) reject(new Error(`${SELECTED.css} did not load (--nn-model-id "${id}", expected "${SELECTED.cssId}")`));
      else setTimeout(poll, 20);
    };
    poll();
  });
}

function fail(e) {
  failed = e;
  $status.innerHTML = '';
  const span = document.createElement('span');
  span.className = 'warn';
  span.textContent = `CSS network unavailable: ${e.message}`;
  $status.appendChild(span);
  console.error(e);
}

loadSelected().then(start, fail);

function start() {
  try {
    selfTest();
    fillModelSelect(cssString('--nn-model-id'));
    setViewModel(cssString('--nn-model-id'));
    const cssGame = cssString('--nn-game') || 'v1';   // the rules the loaded stylesheet was trained for
    if (SELECTED && cssGame !== SELECTED.gameVersion) throw new Error(`${SELECTED.css} is for rules ${cssGame}, the index says ${SELECTED.gameVersion}`);
    if (!DinoCore.GAME_VERSIONS.includes(cssGame)) throw new Error(`model.css is for unknown game ${cssGame}`);
    actionRepeat = Math.max(1, cssNumber('--nn-action-repeat') | 0);
    rulesWarning = DinoPages.rulesWarning(cssGame);
    $rulesWarn.textContent = rulesWarning;
  } catch (e) {
    fail(e);
    return;
  }
  newGame();
  requestAnimationFrame((t) => { last = t; rate.t0 = t; frame(t); });
}
