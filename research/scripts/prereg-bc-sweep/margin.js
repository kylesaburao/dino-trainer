'use strict';
// Min top-2 logit margin over 600k frames of a model's own play (research/prereg-bc-sweep.md
// §6, prereg-bc-shapes.md rung 1). The decision log's original 600k-frame script is not in
// the repo, so this definition was fixed here, before any tier-B result existed:
//   - the model plays v2 with policy.js createAgent (JS forward pass, actionRepeat 1);
//   - start levels 0, 2000, 50000, 300000, 900000 (the bc-gen levels), 120,000 running
//     decision frames from each (600,000 in all);
//   - seeds 30,003,000, 30,003,001, ... (the tier-B scan range, not held out), a fresh seed
//     whenever a game ends (crash or 1,000,000 points) before the level's frame quota;
//   - per running frame, margin = largest logit - second-largest logit, in float64 with the
//     same weights; reported: min (with level, seed, score), 1st percentile, median.
// Usage: node margin.js --model <best.json|model.js> [--out result.json]
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '../../..');   // the checkout this script lives in (worktree-safe)
const Core = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const file = path.resolve(opt('--model'));
const m = file.endsWith('.json') ? JSON.parse(fs.readFileSync(file, 'utf8')) : require(file);
const LEVELS = [0, 2000, 50000, 300000, 900000], PER_LEVEL = 120000, SEED0 = 30003000, TARGET = 1000000;

const L = m.layers, p = m.params;
const bufs = L.map(n => new Float64Array(n));
function logits(x) {
  let input = x, k = 0;
  for (let l = 1; l < L.length; l++) {
    const nIn = L[l - 1], nOut = L[l], out = bufs[l], bBase = k + nOut * nIn, last = l === L.length - 1;
    for (let i = 0; i < nOut; i++) {
      let s = p[bBase + i];
      const row = k + i * nIn;
      for (let j = 0; j < nIn; j++) s += p[row + j] * input[j];
      out[i] = last ? s : Math.tanh(s);
    }
    input = out; k = bBase + nOut;
  }
  return input;
}

const x = new Float64Array(P.OBS_DIM);
const hist = [];
let min = Infinity, at = null, seed = SEED0, games = 0, crashes = 0, frames = 0;
const cap = TARGET / Core.CFG.scoreCoef;
for (const level of LEVELS) {
  let n = 0;
  while (n < PER_LEVEL) {
    const g = Core.createGame({ rng: Core.createRng(seed), version: m.gameVersion || 'v2', startScore: level || undefined });
    const agent = P.createAgent(m);
    games++;
    do {
      if (g.state === 'running' && n < PER_LEVEL) {
        const o = Array.from(logits(P.observe(g, x))).sort((a, b) => b - a);
        const mg = o[0] - o[1];
        hist.push(mg);
        if (mg < min) { min = mg; at = { level, seed, score: g.score() }; }
        n++;
      }
      g.step(agent.act(g));
    } while ((g.state === 'running' || g.state === 'idle') && g.distance < cap && n < PER_LEVEL);
    if (g.state === 'over') crashes++;
    seed++;
  }
  frames += n;
}
hist.sort((a, b) => a - b);
const r = { model: file, layers: L.join('-'), frames, games, crashes, minMargin: min, at, p01: hist[Math.floor(hist.length * 0.01)], median: hist[hist.length >> 1] };
console.log(JSON.stringify(r));
if (opt('--out')) fs.writeFileSync(path.resolve(opt('--out')), JSON.stringify(r) + '\n');
