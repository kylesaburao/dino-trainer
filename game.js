'use strict';

// Browser shell: sound, input events, high score storage, the AI toggle, the
// model and speed selectors and the main loop. All game rules live in core.js
// (DinoCore); drawing is render.js (DinoRender); the optional AI player uses
// policy.js (DinoPolicy) with the shipped model in model.js (DINO_MODEL) or one
// of the bundled models in models.js (DINO_MODELS, `node trainer/cli.js export-models`).
const { FRAME_MS } = DinoCore;

// Created first: its random ground texture is the page's first Math.random use.
const renderer = DinoRender.createRenderer(document.getElementById('game'));

// ---------------------------------------------------------------------------
// Model: ?model=<id> (an id in models.js), else the choice remembered from the
// selector, else the shipped model.js. Changing the selector reloads the page.
// ---------------------------------------------------------------------------
const MODEL_KEY = 'dino-model';
const MODELS = typeof DINO_MODELS !== 'undefined' ? DINO_MODELS : [];
const SHIPPED = typeof DINO_MODEL !== 'undefined' ? DINO_MODEL : null;
const MODEL = (() => {
  const byId = (id) => (id && MODELS.find(m => m.id === id)) || null;
  let stored = null;
  try { stored = localStorage.getItem(MODEL_KEY); } catch (e) { /* storage unavailable */ }
  return byId(new URLSearchParams(location.search).get('model')) || byId(stored) || SHIPPED;
})();

// ---------------------------------------------------------------------------
// Game version: ?game=v1|v2|v3 in the URL, else the version the selected model was
// trained on (so AI mode plays the game it learned), else v2. See core.js.
// ---------------------------------------------------------------------------
const GAME_VERSION = (() => {
  const asked = new URLSearchParams(location.search).get('game');
  if (DinoCore.GAME_VERSIONS.includes(asked)) return asked;
  return MODEL ? MODEL.gameVersion || 'v1' : 'v2';
})();

// ---------------------------------------------------------------------------
// High score (human runs only), kept per game version ('dino-hi' is v1's)
// ---------------------------------------------------------------------------
const HI_KEY = GAME_VERSION === 'v1' ? 'dino-hi' : `dino-hi-${GAME_VERSION}`;
let hi = 0;
try { hi = parseInt(localStorage.getItem(HI_KEY), 10) || 0; } catch (e) { /* storage unavailable */ }
function saveHi() {
  try { localStorage.setItem(HI_KEY, String(hi)); } catch (e) { /* storage unavailable */ }
}

// ---------------------------------------------------------------------------
// Sound (tiny WebAudio blips, created on first input)
// ---------------------------------------------------------------------------
let actx = null;
function initAudio() {
  if (actx) return;
  try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { /* no audio */ }
}
function beep(freq, dur, delay = 0) {
  if (!actx) return;
  const t = actx.currentTime + delay;
  const o = actx.createOscillator();
  const g = actx.createGain();
  o.type = 'square';
  o.frequency.value = freq;
  g.gain.value = 0.04;
  o.connect(g).connect(actx.destination);
  o.start(t);
  o.stop(t + dur);
}
const sounds = {
  jump: () => beep(620, 0.05),
  score: () => { beep(1050, 0.07); beep(1400, 0.1, 0.08); },
  hit: () => { beep(140, 0.08); beep(110, 0.12, 0.09); },
};

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------
let aiRun = false;             // AI was on at some point during the current run

const game = DinoCore.createGame({
  rng: DinoCore.createRng((Math.random() * 4294967296) >>> 0),
  version: GAME_VERSION,
  hooks: {
    sound: (name) => { if (!aiOn || Number(speedSelect.value) <= 2) sounds[name](); },   // silent when fast-forwarding
    night: (on) => document.body.classList.toggle('night', on),
    gameOver: (s) => {
      if (!aiRun && s > hi) {
        hi = s;
        saveHi();
      }
    },
  },
});
const dino = game.dino;

// ---------------------------------------------------------------------------
// AI mode
// ---------------------------------------------------------------------------
// While on, the policy picks the key state inside the fixed-timestep loop right
// before each update, through the same input functions a human drives. Human
// jump/duck input is ignored; after a game over the AI restarts by itself once
// the restart lockout has passed (auto-restart).
const aiButton = document.getElementById('ai-toggle');
const model = MODEL;
let aiOn = false;
let agent = null;

function setAI(on) {
  if (on && !model) return;
  if (on) {
    // Hand over from the player: drop any key the player is holding.
    game.releaseJump();
    game.releaseDown();
    agent = DinoPolicy.createAgent(model);
  } else {
    // Hand back to the player: release whatever the AI is holding.
    game.applyKeys(DinoPolicy.ACTIONS[0], clock);
    agent = null;
  }
  aiOn = on;
  aiButton.textContent = on ? 'AI: ON' : 'AI: OFF';
  aiButton.setAttribute('aria-pressed', String(on));
  speedSelect.disabled = !on;
}

function aiStep(now) {
  const keys = agent.act(game);
  if (game.state === 'over') {
    // Auto-restart: tap jump (the core ignores presses during the lockout).
    game.applyKeys(game.keys.jump ? DinoPolicy.ACTIONS[0] : DinoPolicy.ACTIONS[1], now);
  } else {
    game.applyKeys(keys, now);
  }
}

if (!model) {
  aiButton.disabled = true;
  aiButton.title = 'model.js not loaded';
}
aiButton.addEventListener('click', () => {
  initAudio();
  setAI(!aiOn);
  aiButton.blur();
});

// ---------------------------------------------------------------------------
// Model and speed selectors
// ---------------------------------------------------------------------------
// The model selector lists models.js; the shipped model.js appears as its own
// entry if models.js doesn't contain it. Picking one remembers it and reloads
// with ?model=<id> (keeping ?game=). The speed applies to AI mode only: a human
// always plays at 1x, so the human HI stays comparable.
const modelSelect = document.getElementById('model');
const speedSelect = document.getElementById('speed');
const speedStatus = document.getElementById('speed-status');
const sameModel = (a, b) => !!a && !!b && a.layers.join() === b.layers.join() && a.params.length === b.params.length &&
  a.params.every((v, i) => v === b.params[i]);

(() => {
  const shippedEntry = MODELS.find(m => sameModel(m, SHIPPED));
  const options = MODELS.map(m => [m.id, m.label]);
  if (SHIPPED && !shippedEntry) options.unshift(['', `model.js (${SHIPPED.source || 'shipped'})`]);
  for (const [id, label] of options) modelSelect.add(new Option(label, id));
  const current = MODELS.find(m => m === MODEL) || (MODEL === SHIPPED ? shippedEntry : null);
  modelSelect.value = current ? current.id : '';
  modelSelect.disabled = options.length < 2;
  modelSelect.addEventListener('change', () => {
    const id = modelSelect.value;
    try { if (id) localStorage.setItem(MODEL_KEY, id); else localStorage.removeItem(MODEL_KEY); } catch (e) { /* storage unavailable */ }
    const q = new URLSearchParams(location.search);
    if (id) q.set('model', id); else q.delete('model');
    location.search = q.toString();
  });
})();

for (const el of [modelSelect, speedSelect]) el.addEventListener('change', () => el.blur());

// ---------------------------------------------------------------------------
// Input (ignored for jump/duck while AI mode is on)
// ---------------------------------------------------------------------------
function pressJump() {
  if (aiOn) return;
  initAudio();
  if (actx && actx.state === 'suspended') actx.resume();
  game.pressJump(clock);
}

function releaseJump() {
  if (aiOn) return;
  game.releaseJump();
}

function pressDown() {
  if (aiOn) return;
  game.pressDown();
}

function releaseDown() {
  if (aiOn) return;
  game.releaseDown();
}

// A focused selector keeps its own keys (it blurs itself after a change).
const inControls = (e) => e.target instanceof Element && !!e.target.closest('.controls');

document.addEventListener('keydown', (e) => {
  if (inControls(e)) return;
  if (e.code === 'Space' || e.code === 'ArrowUp') {
    e.preventDefault();
    if (!e.repeat) pressJump();
  } else if (e.code === 'ArrowDown') {
    e.preventDefault();
    if (!e.repeat) pressDown();
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code === 'Space' || e.code === 'ArrowUp') releaseJump();
  else if (e.code === 'ArrowDown') releaseDown();
});

document.addEventListener('pointerdown', (e) => {
  if (e.target === aiButton || inControls(e)) return;    // the toggle and selectors never make the dino jump
  e.preventDefault();
  pressJump();
});
document.addEventListener('pointerup', (e) => {
  if (e.target === aiButton || inControls(e)) return;
  releaseJump();
});

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let lastTime = performance.now();

document.addEventListener('visibilitychange', () => {
  lastTime = performance.now();
});

// Fixed timestep: physics always advances in whole 60fps frames (dt = 1), so play
// at any refresh rate matches the jump arcs the fairness model computes. The small
// negative tolerance absorbs rAF jitter on 60Hz displays without dropping steps.
//
// The game clock (restart lockout, night, score flash, idle blink) advances
// FRAME_MS per update, so at speed N every timing scales with the game, not with
// real time. At 1x it tracks real time. Speed N (AI mode only) adds N times the real
// elapsed time per display frame; at most STEP_BUDGET_MS of updates run per frame,
// and the rest is dropped (the achieved speed then falls short, and says so). The
// policy still runs inside this loop right before each update; drawing happens
// once per display frame.
const STEP_BUDGET_MS = 12;
let clock = performance.now();
let acc = 0;
let prevState = game.state;
let behind = false;
const rate = { t0: performance.now(), frames: 0, speed: 0 };   // achieved speed: simulated / real time
function frame(now) {
  const speed = aiOn ? Number(speedSelect.value) : 1;
  acc += Math.min(Math.max(now - lastTime, 0), 50) * speed;
  lastTime = now;
  const deadline = performance.now() + STEP_BUDGET_MS;
  behind = false;
  while (acc >= FRAME_MS - 2) {
    clock += FRAME_MS;
    if (aiOn) aiStep(clock);
    if (game.state === 'running' && prevState !== 'running') aiRun = false;   // a new run began
    if (aiOn && game.state === 'running') aiRun = true;
    game.update(FRAME_MS, clock);
    prevState = game.state;
    acc -= FRAME_MS;
    rate.frames++;
    if (performance.now() > deadline) { acc = 0; behind = true; break; }
  }
  renderer.draw(game, clock, hi);
  if (now - rate.t0 >= 1000) {
    rate.speed = rate.frames * FRAME_MS / (now - rate.t0);
    rate.t0 = now;
    rate.frames = 0;
    speedStatus.textContent = aiOn
      ? `achieved ${rate.speed >= 10 ? Math.round(rate.speed) : rate.speed.toFixed(1)}×` + (behind || rate.speed < speed * 0.9 ? ` of ${speed}× (CPU-limited)` : '')
      : 'speed applies to AI mode';
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
