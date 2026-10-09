'use strict';

// ---------------------------------------------------------------------------
// Constants (approximations of Chrome's T-Rex runner values; tune freely)
// ---------------------------------------------------------------------------
const W = 600;
const H = 150;
const INK = '#535353';
const GROUND = 140;            // y where feet and cacti rest
const FRAME_MS = 1000 / 60;    // physics values below are "per 60fps frame"

const CFG = {
  speed: 6,
  maxSpeed: 13,
  accel: 0.001,
  gravity: 0.6,
  jumpVel: -10,
  dropVel: -5,
  minJump: 30,                 // px above ground before a released jump is cut
  maxJumpY: 30,                // absolute top y; jump is cut once above this
  speedDropCoef: 3,
  clearTime: 3000,             // ms of running before the first obstacle
  gapCoef: 0.6,
  maxGapCoef: 1.5,
  scoreCoef: 0.025,
  flashMs: 250,
  flashTimes: 3,
  invertEvery: 700,
  invertMs: 12000,
  cloudFreq: 0.5,
  maxClouds: 6,
  restartLockMs: 750,
  minWindowFrames: 4,          // fairness: min takeoff timing tolerance per obstacle
  minGroundFrames: 4,          // fairness: min time on the ground between two jumps
};

// ---------------------------------------------------------------------------
// Sprites: '#' = ink pixel, each map pixel is 2x2 on screen
// ---------------------------------------------------------------------------
function sprite(rows, color = INK, s = 2) {
  const w = Math.max(...rows.map(r => r.length));
  const c = document.createElement('canvas');
  c.width = w * s;
  c.height = rows.length * s;
  const g = c.getContext('2d');
  g.fillStyle = color;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      if (row[x] === '#') g.fillRect(x * s, y * s, s, s);
    }
  });
  return c;
}

const DINO_HEAD = [
  '............#########.',
  '...........##.########',
  '...........###########',
  '...........###########',
  '...........###########',
  '...........######.....',
  '...........#########..',
];
const DINO_BODY = [
  '#.........######......',
  '#........#######......',
  '##......##########....',
  '###....########.#.....',
  '###############.......',
  '.##############.......',
  '..#############.......',
  '...###########........',
  '....#########.........',
  '.....#######..........',
];
const LEGS_STAND = [
  '......###.##..........',
  '......##...#..........',
  '......#....#..........',
  '......##...##.........',
];
const LEGS_RUN_A = [
  '......##..##..........',
  '......###..#..........',
  '...........#..........',
  '...........##.........',
];
const LEGS_RUN_B = [
  '......###.###.........',
  '......##..............',
  '......#...............',
  '......##..............',
];
const withEye = (rows) => [...DINO_HEAD.slice(0, 1), ...rows, ...DINO_HEAD.slice(rows.length + 1)];

const DUCK_BODY = [
  '......................#######.',
  '#.......##############.#######',
  '##....########################',
  '##############################',
  '.#############################',
  '..##################.#####....',
  '...################...........',
  '....#############.##..........',
  '.....##########...............',
];

const S = {
  stand: sprite([...DINO_HEAD, ...DINO_BODY, ...LEGS_STAND]),
  blink: sprite([...withEye(['...........###########']), ...DINO_BODY, ...LEGS_STAND]),
  crash: sprite([
    ...withEye(['...........#.#.#######', '...........##.########', '...........#.#.#######']),
    ...DINO_BODY, ...LEGS_STAND,
  ]),
  run: [
    sprite([...DINO_HEAD, ...DINO_BODY, ...LEGS_RUN_A]),
    sprite([...DINO_HEAD, ...DINO_BODY, ...LEGS_RUN_B]),
  ],
  duck: [
    sprite([...DUCK_BODY,
      '......###..##.................',
      '......##....#.................',
      '......#.....##................']),
    sprite([...DUCK_BODY,
      '......##..###.................',
      '.......#...##.................',
      '.......##.....................']),
  ],
  cactusSmall: sprite([
    '...##...',
    '..####..',
    '..####.#',
    '#.####.#',
    '#.####.#',
    '#.######',
    '#.#####.',
    '######..',
    '.#####..',
    ...Array(8).fill('..####..'),
  ]),
  cactusLarge: sprite([
    '.....##.....',
    '....####....',
    '....####....',
    '....####..#.',
    '.#..####.###',
    '###.####.###',
    '###.####.###',
    '###.####.###',
    '###.####.###',
    '###.#######.',
    '###.######..',
    '########....',
    '.#######....',
    ...Array(12).fill('....####....'),
  ]),
  ptero: [
    sprite([
      '........#..............',
      '........##.............',
      '........###............',
      '.........###...........',
      '...#......###..........',
      '..##.......###.........',
      '.###.......####........',
      '#####################..',
      '....################...',
      '.....############......',
    ]),
    sprite([
      '', '', '', '',
      '...#...................',
      '..##...................',
      '.###...................',
      '#####################..',
      '....################...',
      '.....############......',
      '........#####..........',
      '.........####..........',
      '..........###..........',
      '...........##..........',
      '............#..........',
    ]),
  ],
  cloud: sprite([
    '..........####.........',
    '........##....##.......',
    '......##........#......',
    '..####...........###...',
    '.#..................#..',
    '#######################',
  ], '#dadada'),
};

// Pixel font (5x6 glyphs)
const FONT = {
  0: ['.###.', '#...#', '#...#', '#...#', '#...#', '.###.'],
  1: ['..#..', '.##..', '..#..', '..#..', '..#..', '.###.'],
  2: ['.###.', '#...#', '...#.', '..#..', '.#...', '#####'],
  3: ['####.', '....#', '.###.', '....#', '....#', '####.'],
  4: ['#..#.', '#..#.', '#####', '...#.', '...#.', '...#.'],
  5: ['#####', '#....', '####.', '....#', '....#', '####.'],
  6: ['.###.', '#....', '####.', '#...#', '#...#', '.###.'],
  7: ['#####', '....#', '...#.', '..#..', '.#...', '.#...'],
  8: ['.###.', '#...#', '.###.', '#...#', '#...#', '.###.'],
  9: ['.###.', '#...#', '#...#', '.####', '....#', '.###.'],
  H: ['#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  I: ['#####', '..#..', '..#..', '..#..', '..#..', '#####'],
  G: ['.####', '#....', '#..##', '#...#', '#...#', '.###.'],
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#'],
  M: ['#...#', '##.##', '#.#.#', '#...#', '#...#', '#...#'],
  E: ['#####', '#....', '####.', '#....', '#....', '#####'],
  O: ['.###.', '#...#', '#...#', '#...#', '#...#', '.###.'],
  V: ['#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
  R: ['####.', '#...#', '####.', '#.#..', '#..#.', '#...#'],
};
const GLYPHS = {};
for (const ch in FONT) GLYPHS[ch] = sprite(FONT[ch]);
const GLYPH_W = 10;

// Ground texture: a line with bumps, plus pebbles below; tiled while scrolling
function makeGround() {
  const c = document.createElement('canvas');
  c.width = W * 2;
  c.height = 12;
  const g = c.getContext('2d');
  g.fillStyle = INK;
  for (let x = 0; x < c.width;) {
    if (Math.random() < 0.012 && x < c.width - 8) {
      g.fillRect(x, 1, 1, 1);
      g.fillRect(x + 1, 0, 4, 1);
      g.fillRect(x + 5, 1, 1, 1);
      x += 6;
      continue;
    }
    g.fillRect(x, 2, 1, 1);
    if (Math.random() < 0.05) g.fillRect(x, 4 + Math.floor(Math.random() * 6), Math.random() < 0.5 ? 1 : 2, 1);
    x++;
  }
  return c;
}
const groundImg = makeGround();

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const dpr = window.devicePixelRatio || 1;
canvas.width = W * dpr;
canvas.height = H * dpr;
ctx.scale(dpr, dpr);
ctx.imageSmoothingEnabled = false;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const STAND_TOP = GROUND - S.stand.height;
const OBSTACLE_TYPES = [
  { name: 'small', frames: [S.cactusSmall], ys: [GROUND - S.cactusSmall.height], minGap: 120, multiSpeed: 4 },
  { name: 'large', frames: [S.cactusLarge], ys: [GROUND - S.cactusLarge.height], minGap: 120, multiSpeed: 7 },
  { name: 'ptero', frames: S.ptero, ys: [108, 82, 58], minGap: 150, minSpeed: 8.5, speedOffset: 0.8 },
];

let state = 'idle';            // 'idle' | 'running' | 'over'
let speed, distance, runTime, obstacles, history, milestone, nightMilestone;
let groundX = 0;
let clouds = [{ x: W - 150, y: 40, gap: 200 }];
let hi = 0;
let downHeld = false;
let overAt = 0;
let flashStart = -1, flashScore = 0;
let night = false, nightUntil = 0, nightAlpha = 0;
let moonX = W - 100;
const stars = [{ x: 100, y: 30 }, { x: 400, y: 55 }];

try { hi = parseInt(localStorage.getItem('dino-hi'), 10) || 0; } catch (e) { /* storage unavailable */ }
function saveHi() {
  try { localStorage.setItem('dino-hi', String(hi)); } catch (e) { /* storage unavailable */ }
}

const dino = {
  x: 50, y: STAND_TOP, vy: 0,
  jumping: false, ducking: false, speedDrop: false, reachedMin: false, released: false,
  crashed: false, frame: 0, timer: 0, blinkUntil: 0, nextBlink: 0,
};

function reset() {
  speed = CFG.speed;
  distance = 0;
  runTime = 0;
  obstacles = [];
  history = [];
  milestone = 0;
  nightMilestone = 0;
  flashStart = -1;
  setNight(false);
  Object.assign(dino, {
    y: STAND_TOP, vy: 0, jumping: false, ducking: downHeld, speedDrop: false,
    reachedMin: false, released: false, crashed: false, frame: 0, timer: 0,
  });
}

const score = () => Math.floor(distance * CFG.scoreCoef);

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
// Dino
// ---------------------------------------------------------------------------
const jumpVelocity = (s) => CFG.jumpVel - s / 10;

// One physics step of a jump. Shared by the dino and the fairness model, so the
// model's jump arcs are exactly the arcs the player gets. Returns true while airborne.
function stepJump(j, dt) {
  j.y += j.vy * dt * (j.speedDrop ? CFG.speedDropCoef : 1);
  j.vy += CFG.gravity * dt;
  if (j.y < STAND_TOP - CFG.minJump || j.speedDrop) j.reachedMin = true;
  if ((j.y < CFG.maxJumpY || j.speedDrop || (j.reachedMin && j.released)) && j.vy < CFG.dropVel) {
    j.vy = CFG.dropVel;
  }
  return j.y < STAND_TOP;
}

function startJump() {
  if (dino.jumping || dino.ducking) return;
  dino.jumping = true;
  dino.vy = jumpVelocity(speed);
  dino.reachedMin = false;
  dino.released = false;
  sounds.jump();
}

function updateDino(ms, dt, now) {
  if (dino.jumping) {
    if (!stepJump(dino, dt)) {
      dino.y = STAND_TOP;
      dino.vy = 0;
      dino.jumping = false;
      dino.speedDrop = false;
      dino.ducking = downHeld;
    }
  }

  if (state === 'running') {
    dino.timer += ms;
    if (dino.timer >= 1000 / 12) {
      dino.timer = 0;
      dino.frame ^= 1;
    }
  } else if (state === 'idle' && now > dino.nextBlink) {
    dino.blinkUntil = now + 150;
    dino.nextBlink = now + 1000 + Math.random() * 6000;
  }
}

const standBox = (y) => ({ x: dino.x + 6, y: y + 2, w: 30, h: 38 });

function dinoBox() {
  if (dino.ducking) return { x: dino.x + 4, y: GROUND - S.duck[0].height + 4, w: 50, h: 18 };
  return standBox(dino.y);
}

// ---------------------------------------------------------------------------
// Obstacles
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fairness: never generate a sequence that no input can survive.
//
// Time t is in frames from now, measured relative to the dino (x is fixed).
// For obstacle O moving at v = speed + speedOffset, with hitbox B_O and the
// standing dino hitbox D, the boxes overlap horizontally during
//     O_O = [τ, τ + δ],  τ = (B_O.x − (D.x + D.w)) / v,  δ = (B_O.w + D.w) / v.
//
// Overhead obstacles (B_O.bottom ≤ D.top, i.e. high pterodactyls) require the
// dino to be GROUNDED during O_O.
//
// Every other obstacle must be JUMPED: the dino's box bottom must be higher
// than H = D.bottom − B_O.top for the whole of O_O. A jump released at frame r
// has height arc h_r(t) (from stepJump, the real physics) and airtime T_r.
// Let [a, b] be the frames with h_r(t) > H. Taking off at t0 clears O iff
//     t0 + a ≤ τ  and  τ + δ ≤ t0 + b   ⇔   t0 ∈ [τ + δ − b, τ − a],
// a takeoff window of length L_r = (b − a) − δ. O is clearable iff some r has
// L_r ≥ minWindowFrames (a humanly hittable window). Among those we take the
// shortest airtime T_r and centre the takeoff in its window, so O demands the
// dino be AIRBORNE during A_O = [t0, t0 + T_r].
//
// A sequence is fair iff, for every pair of upcoming obstacles P, Q:
//   - airborne/airborne: A_P and A_Q are separated by ≥ minGroundFrames
//     (time to land and press jump again);
//   - airborne/grounded: A_P and O_Q are disjoint;
//   - grounded/grounded: always compatible.
// Pairwise checks are enough because each interval is fixed per obstacle.
// A new obstacle is pushed right (shifting its interval later by Δx / v) until
// it is compatible with every upcoming obstacle; cactus groups that no jump
// can clear are shrunk. Fast-drop and ducking only add options, so the model
// is conservative.
// ---------------------------------------------------------------------------
// Candidate key-release frames. The shortest is a realistic quick tap (~100ms);
// shorter releases give shorter jumps than people can actually input.
const RELEASE_FRAMES = [6, 8, 10, 12, 15, 20, Infinity];

// Heights above standing for a jump at speed s whose key is released at frame `release`.
function jumpArc(s, release) {
  const j = { y: STAND_TOP, vy: jumpVelocity(s), reachedMin: false, released: false, speedDrop: false };
  const h = [];
  for (let f = 1; f < 200; f++) {
    j.released = f >= release;
    if (!stepJump(j, 1)) break;
    h.push(STAND_TOP - j.y);
  }
  return h; // h[i] is the height at t = i + 1; airtime is h.length + 1
}

// The interval obstacle o forces on the dino, or null if no jump clears it.
function requirement(o, s) {
  const box = obstacleBox(o);
  const stand = standBox(STAND_TOP);
  const v = s + o.speedOffset;
  const tau = (box.x - (stand.x + stand.w)) / v;
  const delta = (box.w + stand.w) / v;
  if (box.y + box.h <= stand.y) return { grounded: true, from: tau, to: tau + delta };

  const H = stand.y + stand.h - box.y;
  let best = null;
  for (const release of RELEASE_FRAMES) {
    const h = jumpArc(s, release);
    const a = h.findIndex(x => x > H) + 1;
    const b = h.findLastIndex(x => x > H) + 1;
    const window = b - a - delta;
    if (a === 0 || window < CFG.minWindowFrames) continue;
    const airtime = h.length + 1;
    if (!best || airtime < best.to - best.from) {
      const t0 = tau + delta - b + window / 2;
      best = { grounded: false, from: t0, to: t0 + airtime, release };
    }
  }
  return best;
}

function compatible(p, q) {
  if (p.grounded && q.grounded) return true;
  const sep = p.grounded || q.grounded ? 0 : CFG.minGroundFrames;
  return p.to + sep <= q.from || q.to + sep <= p.from;
}

// Push o right until its requirement is compatible with every upcoming obstacle.
function placeFairly(o) {
  const upcoming = obstacles.map(p => requirement(p, speed)).filter(r => r && r.to > 0);
  for (let i = 0; i < 100; i++) {
    const r = requirement(o, speed);
    const clash = upcoming.find(p => !compatible(p, r));
    if (!clash) return;
    const sep = clash.grounded || r.grounded ? 0 : CFG.minGroundFrames;
    o.x += Math.max(1, (clash.to + sep - r.from) * (speed + o.speedOffset));
  }
}

function spawnObstacle() {
  let o;
  do {
    const t = OBSTACLE_TYPES[Math.floor(Math.random() * OBSTACLE_TYPES.length)];
    if ((history.length === 2 && history.every(n => n === t.name)) || (t.minSpeed && speed < t.minSpeed)) continue;

    let size = 1 + Math.floor(Math.random() * 3);
    if (!t.multiSpeed || speed < t.multiSpeed) size = 1;

    const img = t.frames[0];
    o = {
      t, size, frame: 0, timer: 0,
      x: W,
      y: t.ys[Math.floor(Math.random() * t.ys.length)],
      w: img.width * size,
      h: img.height,
      speedOffset: t.speedOffset ? (Math.random() < 0.5 ? -1 : 1) * t.speedOffset : 0,
    };
    while (o.size > 1 && !requirement(o, speed)) {
      o.size--;
      o.w = img.width * o.size;
    }
  } while (!o || !requirement(o, speed));

  const minGap = Math.round(o.w * speed + o.t.minGap * CFG.gapCoef);
  const maxGap = Math.round(minGap * CFG.maxGapCoef);
  o.gap = minGap + Math.random() * (maxGap - minGap);
  placeFairly(o);

  obstacles.push(o);
  history.unshift(o.t.name);
  history.length = Math.min(history.length, 2);
}

function obstacleBox(o) {
  if (o.t.name === 'ptero') return { x: o.x + 4, y: o.y + 13, w: 36, h: 8 };
  return { x: o.x + 2, y: o.y + 2, w: o.w - 4, h: o.h - 2 };
}

function updateObstacles(ms, dt) {
  for (const o of obstacles) {
    o.x -= (speed + o.speedOffset) * dt;
    if (o.t.frames.length > 1) {
      o.timer += ms;
      if (o.timer >= 1000 / 6) {
        o.timer = 0;
        o.frame ^= 1;
      }
    }
  }
  obstacles = obstacles.filter(o => o.x + o.w > 0);

  const last = obstacles[obstacles.length - 1];
  if (runTime > CFG.clearTime && (!last || last.x + last.w + last.gap < W)) spawnObstacle();
}

const hits = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

// ---------------------------------------------------------------------------
// Horizon: ground, clouds, night sky
// ---------------------------------------------------------------------------
function updateHorizon(dt) {
  groundX = (groundX + speed * dt) % groundImg.width;

  for (const c of clouds) c.x -= speed * 0.2 * dt;
  clouds = clouds.filter(c => c.x + S.cloud.width > 0);
  const last = clouds[clouds.length - 1];
  if (clouds.length < CFG.maxClouds && (!last || W - last.x > last.gap) && Math.random() < CFG.cloudFreq) {
    clouds.push({ x: W, y: 30 + Math.random() * 41, gap: 100 + Math.random() * 300 });
  }

  if (nightAlpha > 0) {
    moonX -= 0.25 * dt;
    if (moonX < -30) moonX = W + 30;
    for (const s of stars) {
      s.x -= 0.3 * dt;
      if (s.x < -5) { s.x = W + 5; s.y = 10 + Math.random() * 60; }
    }
  }
}

function setNight(on) {
  night = on;
  document.body.classList.toggle('night', on);
}

// ---------------------------------------------------------------------------
// Game flow
// ---------------------------------------------------------------------------
function start() {
  reset();
  state = 'running';
}

function gameOver(now) {
  state = 'over';
  overAt = now;
  dino.crashed = true;
  if (dino.ducking) {
    dino.ducking = false;
    dino.y = STAND_TOP;
  }
  sounds.hit();
  const s = score();
  if (s > hi) {
    hi = s;
    saveHi();
  }
}

function update(ms, now) {
  const dt = ms / FRAME_MS;
  nightAlpha = Math.min(1, Math.max(0, nightAlpha + (night ? 1 : -1) * ms / 750));

  if (state !== 'running') {
    updateDino(ms, dt, now);
    return;
  }

  runTime += ms;
  if (speed < CFG.maxSpeed) speed = Math.min(CFG.maxSpeed, speed + CFG.accel * dt);
  distance += speed * dt;

  updateDino(ms, dt, now);
  updateObstacles(ms, dt);
  updateHorizon(dt);

  const db = dinoBox();
  if (obstacles.some(o => hits(db, obstacleBox(o)))) {
    gameOver(now);
    return;
  }

  const s = score();
  if (Math.floor(s / 100) > milestone) {
    milestone = Math.floor(s / 100);
    flashStart = now;
    flashScore = milestone * 100;
    sounds.score();
  }
  if (Math.floor(s / CFG.invertEvery) > nightMilestone) {
    nightMilestone = Math.floor(s / CFG.invertEvery);
    setNight(true);
    nightUntil = now + CFG.invertMs;
  }
  if (night && now > nightUntil) setNight(false);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
function drawText(str, x, y, gap = 2) {
  for (const ch of str) {
    if (GLYPHS[ch]) ctx.drawImage(GLYPHS[ch], x, y);
    x += GLYPH_W + gap;
  }
}
const textWidth = (str, gap = 2) => str.length * (GLYPH_W + gap) - gap;
const pad = (n) => String(Math.min(n, 99999)).padStart(5, '0');

function drawNightSky() {
  if (nightAlpha <= 0) return;
  ctx.globalAlpha = nightAlpha;
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.arc(moonX, 35, 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(moonX + 5, 32, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = INK;
  for (const s of stars) {
    ctx.fillRect(s.x, s.y - 2, 1, 5);
    ctx.fillRect(s.x - 2, s.y, 5, 1);
  }
  ctx.globalAlpha = 1;
}

function drawScore(now) {
  const right = W - 10;
  let shown = pad(score());
  if (flashStart >= 0) {
    const t = now - flashStart;
    if (t < CFG.flashMs * 2 * CFG.flashTimes) {
      shown = Math.floor(t / CFG.flashMs) % 2 === 0 ? '' : pad(flashScore);
    } else {
      flashStart = -1;
    }
  }
  const scoreX = right - textWidth('00000');
  drawText(shown, scoreX, 6);
  if (hi > 0) {
    ctx.globalAlpha = 0.8;
    drawText('HI ' + pad(hi), scoreX - 20 - textWidth('HI 00000'), 6);
    ctx.globalAlpha = 1;
  }
}

function drawGameOver() {
  const label = 'GAME OVER';
  drawText(label, Math.round((W - textWidth(label, 8)) / 2), 40, 8);

  const cx = W / 2, cy = 80, r = 11;
  const a0 = -Math.PI * 0.35, a1 = Math.PI * 1.45;
  ctx.strokeStyle = INK;
  ctx.fillStyle = INK;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, r, a0, a1);
  ctx.stroke();
  // arrowhead at the end of the arc, pointing clockwise
  const px = cx + r * Math.cos(a1), py = cy + r * Math.sin(a1);
  const tx = -Math.sin(a1), ty = Math.cos(a1);
  const nx = Math.cos(a1), ny = Math.sin(a1);
  ctx.beginPath();
  ctx.moveTo(px + tx * 7, py + ty * 7);
  ctx.lineTo(px + nx * 6, py + ny * 6);
  ctx.lineTo(px - nx * 6, py - ny * 6);
  ctx.closePath();
  ctx.fill();
}

function draw(now) {
  ctx.clearRect(0, 0, W, H);
  drawNightSky();

  for (const c of clouds) ctx.drawImage(S.cloud, Math.round(c.x), Math.round(c.y));

  const gx = Math.round(groundX);
  ctx.drawImage(groundImg, -gx, GROUND - 4);
  ctx.drawImage(groundImg, groundImg.width - gx, GROUND - 4);

  if (obstacles) {
    for (const o of obstacles) {
      const img = o.t.frames[o.frame];
      for (let i = 0; i < o.size; i++) ctx.drawImage(img, Math.round(o.x) + i * img.width, o.y);
    }
  }

  let img;
  if (dino.crashed) img = S.crash;
  else if (dino.ducking) img = S.duck[dino.frame];
  else if (state === 'running' && !dino.jumping) img = S.run[dino.frame];
  else img = now < dino.blinkUntil ? S.blink : S.stand;
  ctx.drawImage(img, dino.x, dino.ducking ? GROUND - img.height : Math.round(dino.y));

  if (state !== 'idle') drawScore(now);
  if (state === 'over') drawGameOver();
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
function pressJump() {
  initAudio();
  if (actx && actx.state === 'suspended') actx.resume();
  if (state === 'idle') {
    start();
    startJump();
  } else if (state === 'over') {
    if (performance.now() - overAt > CFG.restartLockMs) start();
  } else {
    startJump();
  }
}

function releaseJump() {
  dino.released = true;
}

function pressDown() {
  downHeld = true;
  if (state !== 'running') return;
  if (dino.jumping) {
    dino.speedDrop = true;
    dino.vy = 1;
  } else {
    dino.ducking = true;
  }
}

function releaseDown() {
  downHeld = false;
  dino.speedDrop = false;
  dino.ducking = false;
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
  e.preventDefault();
  pressJump();
});
document.addEventListener('pointerup', releaseJump);

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
function frame(now) {
  acc += Math.min(Math.max(now - lastTime, 0), 50);
  lastTime = now;
  while (acc >= FRAME_MS - 2) {
    update(FRAME_MS, now);
    acc -= FRAME_MS;
  }
  draw(now);
  requestAnimationFrame(frame);
}

reset();
requestAnimationFrame(frame);
