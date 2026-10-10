// Dino game simulation core: all game rules, no DOM, no clock, no Math.random.
//
// Loads both as a classic <script> (sets the global `DinoCore`) and as a Node
// require(). The browser shell (game.js) renders the state this core exposes; the
// trainer runs many independent games from it in one thread.
//
// Three game versions share these rules (createGame({ version })):
//   v1 - the original game: a behaviour-preserving extraction of game.js at
//        commit 2da34de. The golden traces in test/golden/ must keep matching.
//        Difficulty stops rising at max speed (score ~1,700).
//   v2 - identical to v1 up to 1,000 points, then progressively harder on a log
//        scale up to 1,000,000 points (see V2 below). Same physics, sprites,
//        observations and fairness model, so every sequence stays survivable.
//   v3 - identical to v2 up to 100,000 points; there the speed cap steps up, then
//        a second log-scale segment (see V3 below) raises it toward superhuman reaction
//        budgets, and the fairness model anchors each takeoff late in its window
//        so the canonical (solver) action comes after the obstacle is visible.
//        The v2 schedule of every other rule continues unchanged.
(function (root) {
'use strict';

// ---------------------------------------------------------------------------
// Constants (approximations of Chrome's T-Rex runner values; tune freely)
// ---------------------------------------------------------------------------
const W = 600;
const H = 150;
const GROUND = 140;            // y where feet and cacti rest
const FRAME_MS = 1000 / 60;    // physics values below are "per 60fps frame"
const PIXEL = 2;               // each sprite map pixel is 2x2 on screen

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
// PRNG: mulberry32 (Tommy Ettinger, public domain). 32-bit state, period 2^32,
// passes PractRand to 2^32 bytes; plenty for game randomness and very fast.
// rng() returns a float in [0, 1) like Math.random. rng.getState()/setState()
// expose the single uint32 of state for checkpoints.
// ---------------------------------------------------------------------------
function createRng(seed) {
  let a = seed >>> 0;
  const rng = function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rng.getState = () => a >>> 0;
  rng.setState = (s) => { a = s >>> 0; };
  return rng;
}

// ---------------------------------------------------------------------------
// Sprite pixel maps: '#' = ink pixel. The single source for both gameplay sizes
// (computed here) and rendering (done by the shell).
// ---------------------------------------------------------------------------
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

const SPRITES = {
  stand: [...DINO_HEAD, ...DINO_BODY, ...LEGS_STAND],
  blink: [...withEye(['...........###########']), ...DINO_BODY, ...LEGS_STAND],
  crash: [
    ...withEye(['...........#.#.#######', '...........##.########', '...........#.#.#######']),
    ...DINO_BODY, ...LEGS_STAND,
  ],
  run: [
    [...DINO_HEAD, ...DINO_BODY, ...LEGS_RUN_A],
    [...DINO_HEAD, ...DINO_BODY, ...LEGS_RUN_B],
  ],
  duck: [
    [...DUCK_BODY,
      '......###..##.................',
      '......##....#.................',
      '......#.....##................'],
    [...DUCK_BODY,
      '......##..###.................',
      '.......#...##.................',
      '.......##.....................'],
  ],
  cactusSmall: [
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
  ],
  cactusLarge: [
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
  ],
  ptero: [
    [
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
    ],
    [
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
    ],
  ],
  cloud: [
    '..........####.........',
    '........##....##.......',
    '......##........#......',
    '..####...........###...',
    '.#..................#..',
    '#######################',
  ],
};

// On-screen size of a pixel map, exactly as the 2x canvas the shell draws it into.
const spriteW = (rows) => Math.max(...rows.map(r => r.length)) * PIXEL;
const spriteH = (rows) => rows.length * PIXEL;

const STAND_TOP = GROUND - spriteH(SPRITES.stand);
const DUCK_H = spriteH(SPRITES.duck[0]);
const CLOUD_W = spriteW(SPRITES.cloud);
const GROUND_TILE_W = W * 2;   // width of the shell's scrolling ground texture
const DINO_X = 50;

const OBSTACLE_TYPES = [
  { name: 'small', sprite: 'cactusSmall', frameCount: 1, w: spriteW(SPRITES.cactusSmall), h: spriteH(SPRITES.cactusSmall),
    ys: [GROUND - spriteH(SPRITES.cactusSmall)], minGap: 120, multiSpeed: 4 },
  { name: 'large', sprite: 'cactusLarge', frameCount: 1, w: spriteW(SPRITES.cactusLarge), h: spriteH(SPRITES.cactusLarge),
    ys: [GROUND - spriteH(SPRITES.cactusLarge)], minGap: 120, multiSpeed: 7 },
  { name: 'ptero', sprite: 'ptero', frameCount: 2, w: spriteW(SPRITES.ptero[0]), h: spriteH(SPRITES.ptero[0]),
    ys: [108, 82, 58], minGap: 150, minSpeed: 8.5, speedOffset: 0.8 },
];

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

const standBox = (y) => ({ x: DINO_X + 6, y: y + 2, w: 30, h: 38 });

function obstacleBox(o) {
  if (o.t.name === 'ptero') return { x: o.x + 4, y: o.y + 13, w: 36, h: 8 };
  return { x: o.x + 2, y: o.y + 2, w: o.w - 4, h: o.h - 2 };
}

const hits = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

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
// Tolerances (o.minWindow, o.minGround) are frozen on the obstacle when it
// spawns, so a later change of difficulty never re-plans obstacles already placed.
function requirement(o, s) {
  const box = obstacleBox(o);
  const stand = standBox(STAND_TOP);
  const v = s + o.speedOffset;
  const tau = (box.x - (stand.x + stand.w)) / v;
  const delta = (box.w + stand.w) / v;
  if (box.y + box.h <= stand.y) return { grounded: true, from: tau, to: tau + delta, minGround: o.minGround };

  const H = stand.y + stand.h - box.y;
  let best = null;
  for (const release of RELEASE_FRAMES) {
    const h = jumpArc(s, release);
    const a = h.findIndex(x => x > H) + 1;
    const b = h.findLastIndex(x => x > H) + 1;
    const window = b - a - delta;
    if (a === 0 || window < o.minWindow) continue;
    const airtime = h.length + 1;
    if (!best || airtime < best.to - best.from) {
      // Centred in the takeoff window; rules v3 above 100k (o.late, frozen at spawn):
    // in its last minWindow frames, so the takeoff follows the obstacle's appearance.
    const t0 = o.late ? tau - a - o.minWindow / 2 : tau + delta - b + window / 2;
      best = { grounded: false, from: t0, to: t0 + airtime, release, minGround: o.minGround };
    }
  }
  return best;
}

// Ground time needed between two airborne requirements: the stricter of the two.
const groundSep = (p, q) => (p.grounded || q.grounded ? 0 : Math.max(p.minGround, q.minGround));

function compatible(p, q) {
  if (p.grounded && q.grounded) return true;
  const sep = groundSep(p, q);
  return p.to + sep <= q.from || q.to + sep <= p.from;
}

// ---------------------------------------------------------------------------
// Game versions and the v2 difficulty schedule.
//
// v2 difficulty level d rises on a log scale from 0 at 1,000 points to 1 at
// 1,000,000 points (d = log10(score / 1000) / 3: 10k -> 1/3, 100k -> 2/3) and
// stays at 1 after that. Each rule below interpolates from its v1 value at d = 0
// to the v2 value at d = 1, so v2 is exactly v1 below 1,000 points.
// ---------------------------------------------------------------------------
const GAME_VERSIONS = ['v1', 'v2', 'v3'];
const V2 = {
  startScore: 1000,        // difficulty starts rising here (d = 0 below)
  endScore: 1000000,       // and is at its maximum (d = 1) from here on
  maxSpeed: 22,            // speed cap (v1: 13)
  gapCoef: 0.35,           // obstacle spacing (v1: 0.6, 1.5); the fairness model still pushes
  maxGapCoef: 1.2,         //   obstacles apart whenever a sequence would be unsurvivable
  minWindowFrames: 2,      // fairness timing tolerances in frames (v1: 4, 4)
  minGroundFrames: 2,
  pteroWeight: 2,          // pterodactyl spawn weight vs 1 per cactus type (v1: 1, i.e. 1/3 -> 1/2)
  pteroSpeedOffset: 1.6,   // pterodactyl speed spread +/- (v1: 0.8)
  groupOf4Level: 0.5,      // cactus groups of up to 4 from this level (~31.6k points; v1: 3)
};

function difficulty(version, score) {
  if ((version !== 'v2' && version !== 'v3') || score <= V2.startScore) return 0;
  return Math.min(1, Math.log10(score / V2.startScore) / Math.log10(V2.endScore / V2.startScore));
}

// ---------------------------------------------------------------------------
// Rules v3: the v2 schedule (d above) plus a second level e that rises on a log
// scale from 0 at 100,000 points to 1 at V3.endScore (e = log10(score / 100k) /
// log10(endScore / 100k)) and stays at 1 after that. Only the speed cap changes:
// it STEPS at 100k from v2's 19 to stepSpeed, then rises toward maxSpeed as
// e^curve. Every other rule keeps following v2. research/game-v3.md has the
// definition (a reaction budget below 150 ms) and the probe that chose these numbers.
//
// The step itself (createGame update()): once e > 0, obstacle spawning pauses
// until the track is empty and the dino is on the ground; then the speed jumps
// to the cap in one frame and spawning resumes. No obstacle placed at the old
// speed is ever on the track at the new one, so the fairness model (whose
// intervals are in frames, and frames per pixel change with speed) stays exact.
// ---------------------------------------------------------------------------
const V3 = {
  startScore: 100000,      // e = 0 below (v3 is exactly v2 there)
  endScore: 1000000,       // e = 1 from here on
  stepSpeed: 41,           // speed cap just above 100k (v2 at 100k: 19)
  maxSpeed: 60,            // speed cap at e = 1 (v2 at 1M: 22)
  curve: 1,                // the cap moves as e^curve (1: plain log scale)
};

function superLevel(version, score) {
  if (version !== 'v3' || score <= V3.startScore) return 0;
  return Math.min(1, Math.log10(score / V3.startScore) / Math.log10(V3.endScore / V3.startScore));
}

const lerp = (a, b, d) => a + (b - a) * d;

// The rules in force at difficulty d (v1 is always d = 0) and v3 level e (0 on
// v1 and v2, where nothing below changes). Writes into `r`.
function rulesAt(d, r, e = 0) {
  r.level = d;
  r.maxSpeed = lerp(CFG.maxSpeed, V2.maxSpeed, d);
  r.gapCoef = lerp(CFG.gapCoef, V2.gapCoef, d);
  r.maxGapCoef = lerp(CFG.maxGapCoef, V2.maxGapCoef, d);
  r.minWindowFrames = lerp(CFG.minWindowFrames, V2.minWindowFrames, d);
  r.minGroundFrames = lerp(CFG.minGroundFrames, V2.minGroundFrames, d);
  r.pteroWeight = lerp(1, V2.pteroWeight, d);
  r.pteroSpeedOffset = lerp(OBSTACLE_TYPES[2].speedOffset, V2.pteroSpeedOffset, d);
  r.maxGroup = d >= V2.groupOf4Level ? 4 : 3;
  r.superLevel = e;
  if (e > 0) {
    r.maxSpeed = lerp(V3.stepSpeed, V3.maxSpeed, Math.pow(e, V3.curve));
  }
  return r;
}

// ---------------------------------------------------------------------------
// createGame({ rng, hooks, version, startScore }): one independent game.
//   rng   - () => [0, 1), e.g. createRng(seed). The only randomness source.
//   hooks - optional { sound(name), night(on), gameOver(score) } for the shell.
//   version    - 'v1' (default), 'v2' or 'v3'.
//   startScore - optional: each run starts at this score (distance) with speed at
//                that score's cap, e.g. to test or train late-game difficulty.
// Time is injected: update(ms, now) and pressJump(now) take the clock; step()
// drives a simulated clock now = frameIndex * FRAME_MS.
// ---------------------------------------------------------------------------
function createGame(opts) {
  const rng = opts.rng;
  const hooks = opts.hooks || {};
  const noop = () => {};
  const sound = hooks.sound || noop;
  const onNight = hooks.night || noop;
  const onGameOver = hooks.gameOver || noop;
  const version = opts.version || 'v1';
  if (!GAME_VERSIONS.includes(version)) throw new Error(`unknown game version ${version}`);
  const startDistance = (opts.startScore || 0) / CFG.scoreCoef;
  const rules = rulesAt(0, {});
  const setRules = () => rulesAt(difficulty(version, score()), rules, superLevel(version, score()));

  let state = 'idle';            // 'idle' | 'running' | 'over'
  let speed, distance, runTime, obstacles, history, milestone, nightMilestone;
  // v2 and v3 only: airborne intervals still owed to obstacles that already left the
  // screen ({ from, to } in absolute run frames). At v2 speeds a jump can outlast
  // the obstacle's time on screen, so placement must keep respecting it.
  let committed, runFrames;
  let groundX = 0;
  let clouds = [{ x: W - 150, y: 40, gap: 200 }];
  let downHeld = false;
  let overAt = 0;
  let flashStart = -1, flashScore = 0;
  let night = false, nightUntil = 0, nightAlpha = 0;
  let moonX = W - 100;
  const stars = [{ x: 100, y: 30 }, { x: 400, y: 55 }];

  // Key state as last applied by applyKeys() (the per-frame action interface).
  const keys = { jump: false, down: false };
  let frameIndex = 0;            // step() clock: now = frameIndex * FRAME_MS

  const dino = {
    x: DINO_X, y: STAND_TOP, vy: 0,
    jumping: false, ducking: false, speedDrop: false, reachedMin: false, released: false,
    crashed: false, frame: 0, timer: 0, blinkUntil: 0, nextBlink: 0,
  };

  function reset() {
    distance = startDistance;
    setRules();
    speed = startDistance > 0 ? Math.max(CFG.speed, rules.maxSpeed) : CFG.speed;
    runTime = 0;
    obstacles = [];
    history = [];
    committed = [];
    runFrames = 0;
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

  // -------------------------------------------------------------------------
  // Dino
  // -------------------------------------------------------------------------
  function startJump() {
    if (dino.jumping || dino.ducking) return;
    dino.jumping = true;
    dino.vy = jumpVelocity(speed);
    dino.reachedMin = false;
    dino.released = false;
    sound('jump');
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
      dino.nextBlink = now + 1000 + rng() * 6000;
    }
  }

  function dinoBox() {
    if (dino.ducking) return { x: dino.x + 4, y: GROUND - DUCK_H + 4, w: 50, h: 18 };
    return standBox(dino.y);
  }

  // -------------------------------------------------------------------------
  // Obstacles
  // -------------------------------------------------------------------------
  // Push o right until its requirement is compatible with every upcoming obstacle.
  function placeFairly(o) {
    const upcoming = obstacles.map(p => requirement(p, speed)).filter(r => r && r.to > 0);
    if (version !== 'v1') {
      committed = committed.filter(c => c.to > runFrames);
      for (const c of committed) upcoming.push({ grounded: false, from: c.from - runFrames, to: c.to - runFrames, minGround: c.minGround });
    }
    for (let i = 0; i < 100; i++) {
      const r = requirement(o, speed);
      const clash = upcoming.find(p => !compatible(p, r));
      if (!clash) {
        // v2 and v3 only: a takeoff must still lie ahead (at v2 speeds a freshly spawned
        // obstacle can otherwise demand a jump that is already due).
        if (version !== 'v1' && !r.grounded && r.from < 1) {
          o.x += Math.max(1, (1 - r.from) * (speed + o.speedOffset));
          continue;
        }
        return;
      }
      const sep = groundSep(clash, r);
      o.x += Math.max(1, (clash.to + sep - r.from) * (speed + o.speedOffset));
    }
  }

  function spawnObstacle() {
    let o;
    do {
      // Weighted pick; with weights 1/1/1 (v1, and v2 below 1,000 points) this is
      // exactly floor(rng() * 3).
      const x = rng() * (2 + rules.pteroWeight);
      const t = OBSTACLE_TYPES[x < 1 ? 0 : x < 2 ? 1 : 2];
      if ((history.length === 2 && history.every(n => n === t.name)) || (t.minSpeed && speed < t.minSpeed)) continue;

      let size = 1 + Math.floor(rng() * rules.maxGroup);
      if (!t.multiSpeed || speed < t.multiSpeed) size = 1;

      o = {
        t, size, frame: 0, timer: 0,
        x: W,
        y: t.ys[Math.floor(rng() * t.ys.length)],
        w: t.w * size,
        h: t.h,
        speedOffset: t.speedOffset ? (rng() < 0.5 ? -1 : 1) * rules.pteroSpeedOffset : 0,
        minWindow: rules.minWindowFrames,
        minGround: rules.minGroundFrames,
        late: rules.superLevel > 0,
      };
      while (o.size > 1 && !requirement(o, speed)) {
        o.size--;
        o.w = t.w * o.size;
      }
    } while (!o || !requirement(o, speed));

    const minGap = Math.round(o.w * speed + o.t.minGap * rules.gapCoef);
    const maxGap = Math.round(minGap * rules.maxGapCoef);
    o.gap = minGap + rng() * (maxGap - minGap);
    placeFairly(o);

    obstacles.push(o);
    history.unshift(o.t.name);
    history.length = Math.min(history.length, 2);
  }

  function updateObstacles(ms, dt) {
    for (const o of obstacles) {
      o.x -= (speed + o.speedOffset) * dt;
      if (o.t.frameCount > 1) {
        o.timer += ms;
        if (o.timer >= 1000 / 6) {
          o.timer = 0;
          o.frame ^= 1;
        }
      }
    }
    if (version !== 'v1') {
      for (const o of obstacles) {
        if (o.x + o.w > 0) continue;
        const r = requirement(o, speed);
        if (r && !r.grounded && r.to > 0) committed.push({ from: runFrames + r.from, to: runFrames + r.to, minGround: r.minGround });
      }
    }
    obstacles = obstacles.filter(o => o.x + o.w > 0);

    const last = obstacles[obstacles.length - 1];
    if (runTime > CFG.clearTime && !stepPending() && (!last || last.x + last.w + last.gap < W)) spawnObstacle();
  }

  // Rules v3 only: the speed step at 100k is due but not yet taken (see V3).
  const stepPending = () => rules.superLevel > 0 && speed < V3.stepSpeed;

  // -------------------------------------------------------------------------
  // Horizon: ground, clouds, night sky (state only; the shell draws it)
  // -------------------------------------------------------------------------
  function updateHorizon(dt) {
    groundX = (groundX + speed * dt) % GROUND_TILE_W;

    for (const c of clouds) c.x -= speed * 0.2 * dt;
    clouds = clouds.filter(c => c.x + CLOUD_W > 0);
    const last = clouds[clouds.length - 1];
    if (clouds.length < CFG.maxClouds && (!last || W - last.x > last.gap) && rng() < CFG.cloudFreq) {
      clouds.push({ x: W, y: 30 + rng() * 41, gap: 100 + rng() * 300 });
    }

    if (nightAlpha > 0) {
      moonX -= 0.25 * dt;
      if (moonX < -30) moonX = W + 30;
      for (const s of stars) {
        s.x -= 0.3 * dt;
        if (s.x < -5) { s.x = W + 5; s.y = 10 + rng() * 60; }
      }
    }
  }

  function setNight(on) {
    night = on;
    onNight(on);
  }

  // -------------------------------------------------------------------------
  // Game flow
  // -------------------------------------------------------------------------
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
    sound('hit');
    onGameOver(score());
  }

  function update(ms, now) {
    const dt = ms / FRAME_MS;
    nightAlpha = Math.min(1, Math.max(0, nightAlpha + (night ? 1 : -1) * ms / 750));

    if (state !== 'running') {
      updateDino(ms, dt, now);
      return;
    }

    runTime += ms;
    runFrames++;
    if (version !== 'v1') setRules();
    // Rules v3: take the speed step once the track is clear and the dino has landed.
    if (stepPending() && obstacles.length === 0 && !dino.jumping) speed = rules.maxSpeed;
    if (speed < rules.maxSpeed) speed = Math.min(rules.maxSpeed, speed + CFG.accel * dt);
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
      sound('score');
    }
    if (Math.floor(s / CFG.invertEvery) > nightMilestone) {
      nightMilestone = Math.floor(s / CFG.invertEvery);
      setNight(true);
      nightUntil = now + CFG.invertMs;
    }
    if (night && now > nightUntil) setNight(false);
  }

  // -------------------------------------------------------------------------
  // Input (the same four functions a human drives)
  // -------------------------------------------------------------------------
  function pressJump(now) {
    if (state === 'idle') {
      start();
      startJump();
    } else if (state === 'over') {
      if (now - overAt > CFG.restartLockMs) start();
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

  // Per-frame key state -> input-function calls, on transitions only (a held key
  // does not re-press, exactly like keydown without repeats). Fixed order:
  // releases before presses, jump before down.
  function applyKeys(next, now) {
    const jump = !!next.jump, down = !!next.down;
    if (keys.jump && !jump) { keys.jump = false; releaseJump(); }
    if (keys.down && !down) { keys.down = false; releaseDown(); }
    if (!keys.jump && jump) { keys.jump = true; pressJump(now); }
    if (!keys.down && down) { keys.down = true; pressDown(); }
  }

  // Headless step: apply the key state for this frame, then update with dt = 1.
  function step(next) {
    const now = frameIndex * FRAME_MS;
    applyKeys(next, now);
    update(FRAME_MS, now);
    frameIndex++;
  }

  reset();

  return {
    pressJump, releaseJump, pressDown, releaseDown, applyKeys, step, update, score,
    dino, stars, keys,
    get state() { return state; },
    get speed() { return speed; },
    get distance() { return distance; },
    get obstacles() { return obstacles; },
    get clouds() { return clouds; },
    get groundX() { return groundX; },
    get nightAlpha() { return nightAlpha; },
    get moonX() { return moonX; },
    get flashStart() { return flashStart; },
    get flashScore() { return flashScore; },
    get frameIndex() { return frameIndex; },
    version,
    get level() { return rules.level; },
    get superLevel() { return rules.superLevel; },
  };
}

// Test-only access to the fairness solver. The policy must never use it
// (test/policy.test.js checks that policy.js does not reference it).
const _internal = { requirement, rulesAt, jumpArc };

// The fastest pterodactyl speed offset any version can draw (rules v2 and v3: 1.6).
const MAX_PTERO_SPEED_OFFSET = Math.max(OBSTACLE_TYPES[2].speedOffset, V2.pteroSpeedOffset);

const DinoCore = {
  W, H, GROUND, FRAME_MS, PIXEL, CFG, SPRITES, STAND_TOP, DUCK_H, CLOUD_W, GROUND_TILE_W, DINO_X,
  OBSTACLE_TYPES, RELEASE_FRAMES, GAME_VERSIONS, V2, V3, difficulty, superLevel, MAX_PTERO_SPEED_OFFSET,
  createRng, createGame, spriteW, spriteH, standBox, obstacleBox, _internal,
};

root.DinoCore = DinoCore;
if (typeof module !== 'undefined' && module.exports) module.exports = DinoCore;
})(typeof globalThis !== 'undefined' ? globalThis : this);
