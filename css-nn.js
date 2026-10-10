'use strict';

// CSS-network page (index-css.html): the AI dino with its MLP evaluated by the
// browser's style engine. model.css (generated from model.js by
// trainer/css-model.js) turns the inputs --x0..--x43 into the action --act.
// This script writes the observation (policy.js observe(), exactly what
// training saw) into those properties and reads --act back. There is no
// JavaScript fallback: if the CSS self-test fails, the dino doesn't play.
// The page never loads model.js; the weights exist only in model.css.
const { FRAME_MS } = DinoCore;
const { OBS_DIM, ACTIONS, observe } = DinoPolicy;

const renderer = DinoRender.createRenderer(document.getElementById('game'));
const nn = document.getElementById('nn');
const $status = document.getElementById('status');
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
    current = ACTIONS[cssDecide(x)];
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
  $status.textContent = `game ${game.version}${game.version !== 'v1' ? ` (level ${game.level.toFixed(2)}${game.superLevel > 0 ? `, v3 level ${game.superLevel.toFixed(2)}` : ''})` : ''} · score ${game.score()}` +
    ` · ${eff.toFixed(1)}× real time` + (behind ? ` (CSS can't keep up with ${$speed.value}×)` : '') +
    ` · ${Math.round(rate.fps / actionRepeat)} CSS decisions/s`;
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

(function start() {
  try {
    selfTest();
    gameVersion = cssString('--nn-game') || 'v1';
    // ?game=v1|v2|v3 plays those rules instead of the model's own (as on index.html).
    const asked = new URLSearchParams(location.search).get('game');
    if (DinoCore.GAME_VERSIONS.includes(asked)) gameVersion = asked;
    actionRepeat = Math.max(1, cssNumber('--nn-action-repeat') | 0);
    if (!DinoCore.GAME_VERSIONS.includes(gameVersion)) throw new Error(`model.css is for unknown game ${gameVersion}`);
  } catch (e) {
    fail(e);
    return;
  }
  newGame();
  requestAnimationFrame((t) => { last = t; rate.t0 = t; frame(t); });
})();
