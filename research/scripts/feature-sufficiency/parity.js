// Check: a base-variant clone run through policy.createAgent gives the same actions as cloneB's loop.
const R = '/Users/kyle/Desktop/dino-game/';
const P = require(R + 'policy.js'), C = require(R + 'core.js');
const m = JSON.parse(require('fs').readFileSync(process.argv[2]));
const agent = P.createAgent({ format: 'dino-mlp-v1', layers: m.layers, params: m.params, actionRepeat: 1 });
const decide = P.createNet(m.layers, Float64Array.from(m.params));
const x = new Float64Array(44); let diff = 0, n = 0;
const g = C.createGame({ rng: C.createRng(777), version: 'v2', startScore: 300000 });
for (let f = 0; f < 20000 && g.state !== 'over'; f++) {
  const a = agent.act(g);
  if (g.state === 'running') { const b = P.ACTIONS[decide(P.observe(g, x))]; if (a !== b) diff++; n++; }
  g.step(a);
}
console.log('frames', n, 'action mismatches', diff);
