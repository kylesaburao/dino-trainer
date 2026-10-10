// Shipped model: action frequencies per zone (how often it ever uses DOWN), and
// the length of each jump-key hold that starts a jump (frames the key stays down).
const R = '/Users/kyle/Desktop/dino-game/';
const C = require(R + 'core.js'), P = require(R + 'policy.js'), model = require(R + 'model.js');
for (const [L, span] of [[0, 2000], [50000, 10000], [300000, 10000], [970000, 10000]]) {
  const cnt = [0, 0, 0], holds = {}; let airDown = 0;
  for (let s = 0; s < 40; s++) {
    const g = C.createGame({ rng: C.createRng(70000 + s), version: 'v2', startScore: L || undefined });
    const ag = P.createAgent(model); const cap = (L + span) / C.CFG.scoreCoef; let h = -1;
    do {
      const a = ag.act(g); const i = P.ACTIONS.indexOf(a);
      if (g.state === 'running') { cnt[i]++; if (i === 2 && g.dino.jumping) airDown++; }
      const wasJ = g.dino.jumping; g.step(a);
      if (!wasJ && g.dino.jumping) h = 1; else if (h > 0) { if (i === 1) h++; else { holds[h] = (holds[h] || 0) + 1; h = -1; } }
    } while (g.state === 'running' && g.distance < cap);
  }
  const tot = cnt.reduce((a, b) => a + b);
  const hs = Object.entries(holds).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${k}f:${v}`).join(' ');
  console.log(`level ${L}: none ${(cnt[0] / tot).toFixed(3)} jump ${(cnt[1] / tot).toFixed(3)} down ${(cnt[2] / tot).toFixed(4)} (airborne down frames ${airDown}) | jump-key hold lengths (top): ${hs}`);
}
