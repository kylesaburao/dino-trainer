'use strict';

// CSS-network page (index-css.html): the AI dino with its MLP evaluated by the
// browser's style engine. model.css (generated from model.js by
// trainer/css-model.js) turns the inputs --x0..--x43 into the action --act.
// This script writes the observation (policy.js observe(), exactly what
// training saw) into those properties and reads --act back. There is no
// JavaScript fallback: if the CSS self-test fails, the dino doesn't play.
// model.js, when present, is used only to cross-check the CSS decisions.
const { FRAME_MS } = DinoCore;
const { OBS_DIM, ACTIONS, observe } = DinoPolicy;

const renderer = DinoRender.createRenderer(document.getElementById('game'));
const nn = document.getElementById('nn');
const $status = document.getElementById('status');
const $check = document.getElementById('check');
const $speed = document.getElementById('speed');

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
// Cross-check against model.js (same weights in JavaScript), if present
// ---------------------------------------------------------------------------
const MODEL = typeof DINO_MODEL !== 'undefined' ? DINO_MODEL : null;
const jsDecide = MODEL ? DinoPolicy.createNet(MODEL.layers, MODEL.params) : null;
const check = { frames: 0, disagree: 0, stale: null };

// Same identity as trainer/css-model.js modelId(): generation + sha256 prefix.
async function modelJsId() {
  const json = JSON.stringify({ layers: MODEL.layers, params: Array.from(MODEL.params) });
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(json));
  const hex = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
  return `${MODEL.generation ?? 'na'}-${hex.slice(0, 16)}`;
}

// ---------------------------------------------------------------------------
// Game and agent
// ---------------------------------------------------------------------------
let gameVersion = 'v1', actionRepeat = 1;
let game = null, wait = 0, current = ACTIONS[0], overAt = 0, best = 0;
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
    const a = cssDecide(x);
    if (jsDecide) {
      check.frames++;
      if (jsDecide(x) !== a) check.disagree++;
    }
    current = ACTIONS[a];
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
  showStatus();
}

function showStatus() {
  const eff = rate.fps * FRAME_MS / 1000;
  $status.textContent = `game ${game.version}${game.version === 'v2' ? ` (level ${game.level.toFixed(2)})` : ''} · score ${game.score()}` +
    ` · ${eff.toFixed(1)}× real time` + (behind ? ` (CSS can't keep up with ${$speed.value}×)` : '') +
    ` · ${Math.round(rate.fps / actionRepeat)} CSS decisions/s`;
  if (!jsDecide) {
    $check.textContent = 'model.js not loaded: no JavaScript cross-check';
  } else {
    $check.innerHTML = '';
    const span = document.createElement('span');
    if (check.stale) span.className = 'warn';
    span.textContent = (check.stale ? `${check.stale} · ` : '') +
      `cross-check vs model.js: ${check.frames - check.disagree}/${check.frames} decisions identical`;
    $check.appendChild(span);
  }
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

(async function start() {
  try {
    selfTest();
    gameVersion = cssString('--nn-game') || 'v1';
    actionRepeat = Math.max(1, cssNumber('--nn-action-repeat') | 0);
    if (!DinoCore.GAME_VERSIONS.includes(gameVersion)) throw new Error(`model.css is for unknown game ${gameVersion}`);
  } catch (e) {
    fail(e);
    return;
  }
  if (MODEL) {
    try {
      const id = await modelJsId(), cssId = cssString('--nn-model-id');
      if (id !== cssId) check.stale = `model.css (${cssId}) is not from this model.js (${id}); run node trainer/cli.js export-css`;
    } catch (e) {
      check.stale = 'could not compare model.css with model.js';
    }
  }
  newGame();
  requestAnimationFrame((t) => { last = t; rate.t0 = t; frame(t); });
})();
