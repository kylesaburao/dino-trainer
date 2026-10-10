'use strict';

// Browser shell: sound, input events, high score storage, the AI toggle and the
// main loop. All game rules live in core.js (DinoCore); drawing is render.js
// (DinoRender); the optional AI player uses policy.js (DinoPolicy) with the
// model in model.js (DINO_MODEL).
const { FRAME_MS } = DinoCore;

// Created first: its random ground texture is the page's first Math.random use.
const renderer = DinoRender.createRenderer(document.getElementById('game'));

// ---------------------------------------------------------------------------
// Game version: ?game=v1|v2 in the URL, else the version the shipped model was
// trained on (so AI mode plays the game it learned), else v2. See core.js.
// ---------------------------------------------------------------------------
const MODEL = typeof DINO_MODEL !== 'undefined' ? DINO_MODEL : null;
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
    sound: (name) => sounds[name](),
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
    game.applyKeys(DinoPolicy.ACTIONS[0], performance.now());
    agent = null;
  }
  aiOn = on;
  aiButton.textContent = on ? 'AI: ON' : 'AI: OFF';
  aiButton.setAttribute('aria-pressed', String(on));
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
// Input (ignored for jump/duck while AI mode is on)
// ---------------------------------------------------------------------------
function pressJump() {
  if (aiOn) return;
  initAudio();
  if (actx && actx.state === 'suspended') actx.resume();
  game.pressJump(performance.now());
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

document.addEventListener('keydown', (e) => {
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
  if (e.target === aiButton) return;    // the toggle never makes the dino jump
  e.preventDefault();
  pressJump();
});
document.addEventListener('pointerup', (e) => {
  if (e.target === aiButton) return;
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
let acc = 0;
let prevState = game.state;
function frame(now) {
  acc += Math.min(Math.max(now - lastTime, 0), 50);
  lastTime = now;
  while (acc >= FRAME_MS - 2) {
    if (aiOn) aiStep(now);
    if (game.state === 'running' && prevState !== 'running') aiRun = false;   // a new run began
    if (aiOn && game.state === 'running') aiRun = true;
    game.update(FRAME_MS, now);
    prevState = game.state;
    acc -= FRAME_MS;
  }
  renderer.draw(game, now, hi);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
