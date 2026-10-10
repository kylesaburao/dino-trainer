'use strict';
// How often does the shipped model land with the jump key still held (so its next
// "jump" output does nothing until it outputs a non-jump first)? And how often does
// it output DOWN-then-JUMP while airborne (the transition that re-holds the key)?
// Also: hidden-unit saturation and argmax margin of the shipped net, per zone.
// usage: node keyheld.js
const R = '/Users/kyle/Desktop/dino-game/';
const Core = require(R + 'core.js');
const P = require(R + 'policy.js');
const model = require(R + 'model.js');
const { createGame, createRng, CFG } = Core;

const L = model.layers, p = model.params;
function forward(x) {           // returns { pre: [layer1, layer2], out }
  let input = x, k = 0; const pre = [];
  for (let l = 1; l < L.length; l++) {
    const nI = L[l - 1], nO = L[l], o = new Float64Array(nO), bb = k + nO * nI;
    for (let i = 0; i < nO; i++) { let s = p[bb + i]; for (let j = 0; j < nI; j++) s += p[k + i * nI + j] * input[j]; o[i] = s; }
    k = bb + nO;
    if (l < L.length - 1) { pre.push(o); input = o.map(Math.tanh); } else return { pre, out: o };
  }
}

function zone(name, level, window, seeds) {
  let landings = 0, landHeld = 0, heldFrames = 0, frames = 0, sat = 0, satN = 0, margins = [], crashes = 0;
  const x = new Float64Array(P.OBS_DIM);
  for (let s = 0; s < seeds; s++) {
    const g = createGame({ rng: createRng(90000 + s), version: 'v2', startScore: level || undefined });
    const ag = P.createAgent(model);
    const cap = (level + window) / CFG.scoreCoef;
    let wasJ = false, f = 0;
    do {
      const a = ag.act(g);
      if (g.state === 'running' && (f++ % 7 === 0)) {
        const r = forward(P.observe(g, x));
        for (const layer of r.pre) for (const v of layer) { satN++; if (Math.abs(v) > 2) sat++; }
        const o = [...r.out].sort((a, b) => b - a); margins.push(o[0] - o[1]);
      }
      g.step(a);
      frames++;
      if (wasJ && !g.dino.jumping) { landings++; if (g.keys.jump) landHeld++; }
      if (!g.dino.jumping && g.keys.jump && g.state === 'running') heldFrames++;
      wasJ = g.dino.jumping;
    } while (g.state === 'running' && g.distance < cap);
    if (g.state === 'over') crashes++;
  }
  margins.sort((a, b) => a - b);
  console.log(JSON.stringify({ name, level, window, seeds, crashes, landings, landHeldFrac: +(landHeld / landings).toFixed(4),
    groundedKeyHeldFrac: +(heldFrames / frames).toFixed(4), hiddenSatFrac: +(sat / satN).toFixed(3),
    marginMedian: +margins[margins.length >> 1].toFixed(3), marginP05: +margins[Math.floor(margins.length * 0.05)].toFixed(3) }));
}

zone('early', 0, 2000, 200);
zone('2k', 2000, 10000, 60);
zone('50k', 50000, 10000, 60);
zone('300k', 300000, 10000, 60);
zone('late', 970000, 10000, 60);
