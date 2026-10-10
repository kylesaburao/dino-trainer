'use strict';
// Premise probe for Run B': what do the curriculum's episodes look like for a stuck centre theta?
// (a) from-0 episodes (cap 50k): score and killer; (b) late windows: start uniform in points on
// [2000, 1M) (start-dist points), window 5000 (the gated floor) and 25000: progress, killer, and
// pteros by height encountered per 1000 frames vs from-0 play.
const path = require('path');
const ROOT = process.argv[3];
const { createRng, createGame, standBox, obstacleBox, CFG } = require(path.join(ROOT, 'core.js'));
const { createAgent } = require(path.join(ROOT, 'policy.js'));
const ck = require(path.resolve(process.argv[2]));
const model = { layers: ck.layers, params: ck.theta, actionRepeat: 1, gameVersion: 'v2' };
const CLS = { 108: 'low', 82: 'mid', 58: 'high' };
function play(seed, start, cap) {
  const opts = { rng: createRng(seed), version: 'v2' }; if (start) opts.startScore = start;
  const g = createGame(opts); const a = createAgent(model);
  const seen = new Set(); const pt = { low: 0, mid: 0, high: 0 }; let frames = 0;
  do {
    g.step(a.act(g)); frames++;
    for (const o of g.obstacles) if (o.x < 600 && !seen.has(o)) { seen.add(o); if (o.t.name === 'ptero') pt[CLS[o.y]]++; }
  } while (g.state === 'running' && g.score() < cap);
  let killer = 'pass';
  if (g.state === 'over') {
    const k = g.obstacles.filter(o => o.x < 120).sort((x, y) => x.x - y.x)[0];
    killer = k ? (k.t.name === 'ptero' ? 'ptero-' + CLS[k.y] : 'cactus') : '?';
  }
  return { progress: (g.score() - start) / (cap - start), score: g.score(), killer, frames, pt, obstacles: seen.size };
}
const N = 40;
const summ = (label, eps) => {
  const k = {}; for (const e of eps) k[e.killer] = (k[e.killer] || 0) + 1;
  const fr = eps.reduce((t, e) => t + e.frames, 0);
  const pt = eps.reduce((t, e) => { for (const c in e.pt) t[c] += e.pt[c]; return t; }, { low: 0, mid: 0, high: 0 });
  const prog = eps.map(e => e.progress).sort((x, y) => x - y);
  console.log(`${label}: median progress ${prog[N >> 1].toFixed(4)} (min ${prog[0].toFixed(4)}, max ${prog[N - 1].toFixed(4)}); killers ${JSON.stringify(k)}; mean frames ${(fr / N).toFixed(0)}; pteros seen per 1000 frames low/mid/high ${(pt.low / fr * 1000).toFixed(2)}/${(pt.mid / fr * 1000).toFixed(2)}/${(pt.high / fr * 1000).toFixed(2)}; obstacles per episode ${(eps.reduce((t, e) => t + e.obstacles, 0) / N).toFixed(1)}`);
};
const r = createRng(4242);
const zero = [], late5 = [], late25 = [];
for (let i = 0; i < N; i++) zero.push(play(910000 + i, 0, 50000));
for (let i = 0; i < N; i++) { const s = Math.min(999999, Math.round(2000 + r() * (1e6 - 2000))); late5.push(play(920000 + i, s, Math.min(s + 5000, 1e6))); }
for (let i = 0; i < N; i++) { const s = Math.min(999999, Math.round(2000 + r() * (1e6 - 2000))); late25.push(play(930000 + i, s, Math.min(s + 25000, 1e6))); }
console.log(path.basename(process.argv[2]));
summ('from 0, cap 50k     ', zero); summ('late, window 5000   ', late5); summ('late, window 25000  ', late25);
