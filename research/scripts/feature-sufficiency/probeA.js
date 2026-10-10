'use strict';
// Probe A: play the shipped model and record, for every obstacle encounter,
// the 44 features + action at fixed tau alignment points (tau = frames until the
// obstacle hitbox reaches the dino's front), plus full-state extras the features
// do not carry. Crashes get a 45-frame ring buffer.
// usage: node probeA.js early <seedFrom> <seedTo> <out.jsonl>
//        node probeA.js late  <seedFrom> <seedTo> <out.jsonl> [keepFrac]
const R = '/Users/kyle/Desktop/dino-game/';
const Core = require(R + 'core.js');
const P = require(R + 'policy.js');
const model = require(R + 'model.js');
const fs = require('fs');
const { createGame, createRng, obstacleBox, standBox, STAND_TOP, W, DINO_X, GROUND, DUCK_H, CFG, _internal: { requirement } } = Core;
const FRONT = standBox(STAND_TOP).x + standBox(STAND_TOP).w;
const TAUS = [40, 30, 20, 12, 6, 2];

const [mode, a, b, out, keepArg] = process.argv.slice(2);
const keepFrac = keepArg ? +keepArg : (mode === 'early' ? 0.05 : 0.005);
const keepPtero = mode === 'early' ? 1 : keepFrac;
const highOnly = process.argv[7] === 'high';  // keep every high (yc2) ptero, nothing else
const LEVELS = mode === 'late' ? [950000, 960000, 970000, 980000, 990000] : [0];
const WINDOW = mode === 'late' ? 10000 : 2000;
const ws = fs.createWriteStream(out);

const hash = (x) => ((Math.imul(x ^ 0x9E3779B1, 0x85EBCA6B) >>> 0) / 4294967296);
const tauOf = (g, o) => (obstacleBox(o).x - FRONT) / (g.speed + o.speedOffset);
const r2 = (x) => Math.round(x * 1000) / 1000;

function fullState(g, o) {
  const d = g.dino;
  const req = requirement(o, g.speed);
  const sorted = g.obstacles.filter(p => obstacleBox(p).x + obstacleBox(p).w > 54).sort((p, q) => p.x - q.x);
  const i = sorted.indexOf(o);
  const nx = sorted[i + 1];
  const res = {
    gap: o.gap, req: req && { g: req.grounded, from: r2(req.from), to: r2(req.to), rel: req.release },
    released: d.released, speedDrop: d.speedDrop, reachedMin: d.reachedMin, keyJump: g.keys.jump, keyDown: g.keys.down,
    nVisible: g.obstacles.filter(p => p.x < W && p.x + p.w > DINO_X).length, nAll: g.obstacles.length,
  };
  if (nx) {
    const rn = requirement(nx, g.speed);
    res.next = { t: nx.t.name, x: r2(nx.x), visible: nx.x < W, tau: r2(tauOf(g, nx)), gapPx: r2(obstacleBox(nx).x - (obstacleBox(o).x + obstacleBox(o).w)),
      req: rn && { g: rn.grounded, from: r2(rn.from), to: r2(rn.to), rel: rn.release } };
  }
  return res;
}

function run(seed, level) {
  const game = createGame({ rng: createRng(seed), version: 'v2', startScore: level || undefined });
  const agent = P.createAgent(model);
  const cap = (level + WINDOW) / CFG.scoreCoef;
  const x = new Float64Array(P.OBS_DIM);
  const enc = new Map();
  const ring = [];
  let f = 0;
  do {
    const act = agent.act(game);
    if (game.state === 'running') {
      P.observe(game, x);
      const ai = act === P.ACTIONS[1] ? 1 : act === P.ACTIONS[2] ? 2 : 0;
      ring.push({ f, x: Array.from(x, r2), a: ai, y: r2(game.dino.y), vy: r2(game.dino.vy), j: game.dino.jumping, rel: game.dino.released, kj: game.keys.jump });
      if (ring.length > 45) ring.shift();
      for (const o of game.obstacles) {
        if (!(o.x < W)) continue;
        let e = enc.get(o);
        if (!e) {
          e = { seed, level, type: o.t.name, size: o.size, yClass: o.t.name === 'ptero' ? o.t.ys.indexOf(o.y) : -1, speed: r2(game.speed), v: r2(game.speed + o.speedOffset), spawnScore: game.score(), snaps: {}, firstF: f };
          enc.set(o, e);
        }
        const tau = tauOf(game, o);
        for (const T of TAUS) if (!e.snaps[T] && tau < T) e.snaps[T] = { tau: r2(tau), x: Array.from(x, r2), a: ai, rank: null, fs: fullState(game, o) };
        if (!e.done && obstacleBox(o).x + obstacleBox(o).w <= 54) {
          e.done = true; e.out = 'pass';
          if (highOnly ? e.yClass === 2 : hash(seed * 7919 + f) < (e.type === 'ptero' ? keepPtero : keepFrac)) ws.write(JSON.stringify(e) + '\n');
        }
      }
    }
    game.step(act);
    f++;
  } while (game.state === 'running' && game.distance < cap);
  if (game.state === 'over') {
    const d = game.dino;
    const db = d.ducking ? { x: d.x + 4, y: GROUND - DUCK_H + 4, w: 50, h: 18 } : standBox(d.y);
    const hit = game.obstacles.find(o => { const q = obstacleBox(o); return db.x < q.x + q.w && db.x + db.w > q.x && db.y < q.y + q.h && db.y + db.h > q.y; });
    const e = hit && enc.get(hit);
    const rec = e || { seed, level, type: hit ? hit.t.name : '?' };
    rec.out = 'crash'; rec.score = game.score(); rec.ring = ring;
    rec.crashDino = { y: r2(d.y), vy: r2(d.vy), jumping: d.jumping, wasDucking: game.keys.down };
    if (hit) rec.hitX = r2(obstacleBox(hit).x);
    ws.write(JSON.stringify(rec) + '\n');
    return 1;
  }
  return 0;
}

let crashes = 0, n = 0;
for (let s = +a; s < +b; s++) for (const L of LEVELS) { crashes += run(s, L); n++; }
ws.end();
console.error(`${mode} ${a}-${b}: ${crashes}/${n} crashed`);
