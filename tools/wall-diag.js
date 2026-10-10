'use strict';
// Death diagnosis for a checkpoint's centre policy (theta, no perturbation), v2 from score 0.
// Usage: node tools/wall-diag.js <checkpoint.json> [--seeds 50] [--seed-base 900000] [--cap 100000] [--json]
//
// Per episode: score at death, the killing obstacle (type; ptero y class), speed, the dino's
// state at the crash (ground / landing / air / ducking), the actions over the 30 frames before death,
// and the share of all frames with each action. Ptero y classes against the hitboxes
// (standing dino box y 100-138, ducking 120-138; ptero box y+13..y+21):
//   y=108 low  (box 121-129): hits standing and ducking -> must jump
//   y=82  mid  (box 95-103):  hits standing only        -> duck (or a very high jump)
//   y=58  high (box 71-79):   misses a grounded dino    -> only a jump can hit it
// Seeds default to 900000+ (outside the trainer's seed ranges).
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { createRng, createGame, standBox, obstacleBox, GROUND, DUCK_H, CFG } = require(path.join(ROOT, 'core.js'));
const { createAgent } = require(path.join(ROOT, 'policy.js'));

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? Number(args[i + 1]) : def; };
const file = args.find((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && args[i - 1] !== '--json'));
const SEEDS = opt('--seeds', 50), BASE = opt('--seed-base', 900000), CAP = opt('--cap', 100000);

const ck = require(path.resolve(file));
const model = { layers: ck.layers, params: ck.theta, actionRepeat: ck.config.actionRepeat, gameVersion: ck.config.gameVersion };
const PTERO_CLASS = { 108: 'low', 82: 'mid', 58: 'high' };
const actName = (k) => (k.jump ? 'jump' : k.down ? 'duck' : 'none');
const overlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

function play(seed) {
  const game = createGame({ rng: createRng(seed), version: model.gameVersion });
  const agent = createAgent(model);
  const capDistance = CAP / CFG.scoreCoef;
  const recent = [];
  const totals = { none: 0, jump: 0, duck: 0 };
  let wasDucking = false, frame = 0, lastAir = -Infinity;
  do {
    const k = agent.act(game);
    const a = actName(k);
    totals[a]++;
    recent.push(a); if (recent.length > 30) recent.shift();
    if (game.dino.jumping) lastAir = frame;
    // applyKeys runs before the collision check, so a down key on the ground ducks this frame;
    // gameOver then un-ducks the dino, hence recording it here.
    wasDucking = !game.dino.jumping && (game.dino.ducking || k.down);
    game.step(k);
    frame++;
  } while (game.state === 'running' && game.distance < capDistance);
  const crashed = game.state === 'over';
  const d = game.dino;
  const box = wasDucking && !d.jumping ? { x: d.x + 4, y: GROUND - DUCK_H + 4, w: 50, h: 18 } : standBox(d.y);
  let killer = null;
  if (crashed) {
    const hitters = game.obstacles.filter(o => overlap(box, obstacleBox(o)) || overlap(standBox(d.y), obstacleBox(o)));
    killer = hitters.sort((a, b) => a.x - b.x)[0] || null;
  }
  const last30 = { none: 0, jump: 0, duck: 0 };
  for (const a of recent) last30[a]++;
  return {
    seed, score: game.score(), crashed, speed: game.speed,
    killer: killer ? (killer.t.name === 'ptero' ? `ptero-${PTERO_CLASS[killer.y] || killer.y}` : `cactus-${killer.t.name}${killer.size > 1 ? 'x' + killer.size : ''}`) : (crashed ? 'unknown' : 'survived'),
    // landing: the dino was airborne within the 3 frames before the crash (a jump too short
    // for the obstacle's width), whatever key it pressed on touchdown
    state: !crashed ? '-' : d.jumping ? 'air' : frame - 1 - lastAir <= 3 ? 'landing' : wasDucking ? 'ducking' : 'ground',
    last30, totals,
  };
}

const eps = [];
for (let i = 0; i < SEEDS; i++) eps.push(play(BASE + i));
if (args.includes('--json')) { console.log(JSON.stringify(eps)); process.exit(0); }

const med = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const allFrames = eps.reduce((t, e) => { for (const k in e.totals) t[k] += e.totals[k]; return t; }, { none: 0, jump: 0, duck: 0 });
const nFrames = allFrames.none + allFrames.jump + allFrames.duck;
console.log(`${path.basename(path.dirname(path.resolve(file)))}/${path.basename(file)}: generation ${ck.generation}, ${SEEDS} seeds from ${BASE}, cap ${CAP}`);
console.log(`scores: median ${med(eps.map(e => e.score))}, min ${Math.min(...eps.map(e => e.score))}, max ${Math.max(...eps.map(e => e.score))}; survived to cap ${eps.filter(e => !e.crashed).length}`);
console.log(`actions over all frames: none ${(allFrames.none / nFrames * 100).toFixed(1)}%, jump ${(allFrames.jump / nFrames * 100).toFixed(1)}%, duck ${(allFrames.duck / nFrames * 100).toFixed(2)}%; episodes with any duck frame: ${eps.filter(e => e.totals.duck > 0).length}/${SEEDS}`);
console.log('| Killer | Deaths | Median score | Score range | Median speed | State at crash (ground/landing/air/ducking) | Last-30 actions none/jump/duck (mean) |');
console.log('|---|---|---|---|---|---|---|');
const groups = {};
for (const e of eps) (groups[e.killer] = groups[e.killer] || []).push(e);
for (const [k, g] of Object.entries(groups).sort((a, b) => b[1].length - a[1].length)) {
  const st = (s) => g.filter(e => e.state === s).length;
  const m30 = (a) => (g.reduce((t, e) => t + e.last30[a], 0) / g.length).toFixed(1);
  console.log(`| ${k} | ${g.length} | ${med(g.map(e => e.score))} | ${Math.min(...g.map(e => e.score))}–${Math.max(...g.map(e => e.score))} | ${med(g.map(e => e.speed)).toFixed(2)} | ${st('ground')}/${st('landing')}/${st('air')}/${st('ducking')} | ${m30('none')}/${m30('jump')}/${m30('duck')} |`);
}
