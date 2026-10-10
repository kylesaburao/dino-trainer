'use strict';
// Hand-review aid for §5.3 step 6 (research/prereg-teacher-duck-step2.md): for each failing
// tier-B seed of a model, replays the game (policy.js createAgent, JS forward pass) and
// describes the final DOWN run before the crash: its length in frames, the score at its onset,
// whether a mid pterodactyl overlapped the standing box's x-range during it, the nearest
// on-screen obstacle at onset, and the killer. Single-threaded.
// Usage: node duckrun.js --model <best.json> --scan <scan.jsonl> [--out f.json]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const { createRng, createGame, standBox, obstacleBox, CFG } = require(path.join(ROOT, 'core.js'));
const { createAgent } = require(path.join(ROOT, 'policy.js'));
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const m = JSON.parse(fs.readFileSync(path.resolve(opt('--model')), 'utf8'));
const lines = fs.readFileSync(path.resolve(opt('--scan')), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const TARGET = lines[0].target, cap = TARGET / CFG.scoreCoef;
const PC = { 108: 'low', 82: 'mid', 58: 'high' };
const name = o => o.t.name === 'ptero' ? `ptero-${PC[o.y]}` : `cactus-${o.t.name}${o.size > 1 ? 'x' + o.size : ''}`;
const s0 = standBox(98);
const res = [];
for (const r of lines.filter(l => l.seed !== undefined && !l.pass)) {
  const g = createGame({ rng: createRng(r.seed), version: m.gameVersion });
  const a = createAgent({ layers: m.layers, params: m.params, actionRepeat: m.actionRepeat, gameVersion: m.gameVersion });
  let run = null;
  do {
    const k = a.act(g);
    if (k.down && !g.dino.jumping) {
      if (!run) {
        const vis = g.obstacles.filter(o => o.x < 600 && obstacleBox(o).x + obstacleBox(o).w > s0.x).sort((p, q) => p.x - q.x);
        run = { startScore: g.score(), len: 0, mid: false, nearestAtOnset: vis[0] ? name(vis[0]) : 'none', nearestXAtOnset: vis[0] ? obstacleBox(vis[0]).x : null };
      }
      run.len++;
      for (const o of g.obstacles) if (PC[o.y] === 'mid' && o.t.name === 'ptero') { const b = obstacleBox(o); if (b.x < s0.x + s0.w && b.x + b.w > s0.x) run.mid = true; }
    } else run = null;
    g.step(k);
  } while (g.state === 'running' && g.distance < cap);
  const hit = g.obstacles.filter(o => { const b = obstacleBox(o); return b.x < 104 && b.x + b.w > 54; }).sort((p, q) => p.x - q.x)[0];
  res.push({ seed: r.seed, score: Math.min(g.score(), TARGET), crashed: g.state === 'over', killer: hit ? name(hit) : null, finalDuck: run });
}
const agg = {};
for (const x of res) { const k = `${x.finalDuck ? (x.finalDuck.mid ? 'mid-in-duck' : 'no-mid') : 'no-duck'} onset:${x.finalDuck ? x.finalDuck.nearestAtOnset : '-'}`; agg[k] = (agg[k] || 0) + 1; }
const lens = res.filter(x => x.finalDuck).map(x => x.finalDuck.len).sort((p, q) => p - q);
const summary = { n: res.length, agg, duckLen: { min: lens[0], median: lens[lens.length >> 1], max: lens[lens.length - 1] } };
if (opt('--out')) fs.writeFileSync(opt('--out'), JSON.stringify({ summary, res }, null, 1) + '\n');
console.log(JSON.stringify(summary, null, 1));
