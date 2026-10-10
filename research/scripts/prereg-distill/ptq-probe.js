'use strict';
// Premise probe for research/prereg-distill.md (quantization arms; static, seconds).
// For a float model and its post-training-quantized copies (trainer/bc.js quantizeParams,
// the rule bc-quantize uses), play the FLOAT model on rules v2 from the bc-gen start
// levels and count, per running frame, whether each quantized copy would choose a
// different action (decision disagreement on the float model's own states), plus the
// float model's min top-2 margin on the frames where a copy disagrees.
// Usage: node ptq-probe.js --model <best.json> [--frames 20000] [--seed 30003000] [--scheme row]
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..', '..');
const Core = require(path.join(ROOT, 'core.js'));
const P = require(path.join(ROOT, 'policy.js'));
const bc = require(path.join(ROOT, 'trainer', 'bc.js'));
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const m = JSON.parse(fs.readFileSync(path.resolve(opt('--model')), 'utf8'));
const PER_LEVEL = Number(opt('--frames', 20000)), SEED0 = Number(opt('--seed', 30003000)), scheme = opt('--scheme', 'row');
const LEVELS = [0, 2000, 50000, 300000, 900000];
const BITS = [8, 6, 5, 4, 3, 2];

function logitsFn(layers, params) {
  const one = { X: new Float64Array(P.OBS_DIM), n: 1, dim: P.OBS_DIM };
  return (x) => { one.X.set(x); return bc.teacherLogits({ layers, params }, one); };
}
const top2 = (z) => { const s = Array.from(z).sort((a, b) => b - a); return s[0] - s[1]; };
const argmax = (z) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };

const floatZ = logitsFn(m.layers, m.params);
const copies = BITS.map(bits => {
  const q = bc.quantizeParams(m.layers, m.params, { bits, scheme });
  const levels = bc.checkQuantized(m.layers, q.params, { bits, scheme, scales: q.scales });
  return { bits, levels, z: logitsFn(m.layers, q.params), disagree: 0, byClass: [0, 0, 0], minMarginAtFlip: Infinity };
});
const agent = P.createAgent(m);
const x = new Float64Array(P.OBS_DIM);
let frames = 0, seed = SEED0, minMargin = Infinity;
const cls = [0, 0, 0];
for (const level of LEVELS) {
  let n = 0;
  while (n < PER_LEVEL) {
    const g = Core.createGame({ rng: Core.createRng(seed++), version: m.gameVersion || 'v2', startScore: level || undefined });
    while (n < PER_LEVEL && (g.state === 'idle' || g.state === 'running') && g.score() < 1000000) {
      if (g.state === 'running') {
        P.observe(g, x);
        const zf = floatZ(x), a = argmax(zf), mg = top2(zf);
        minMargin = Math.min(minMargin, mg);
        cls[a]++;
        for (const c of copies) if (argmax(c.z(x)) !== a) { c.disagree++; c.byClass[a]++; c.minMarginAtFlip = Math.min(c.minMarginAtFlip, mg); }
        n++; frames++;
      }
      g.step(agent.act(g));
    }
  }
}
console.log(JSON.stringify({ model: opt('--model'), layers: m.layers, scheme, frames, floatActions: { none: cls[0], jump: cls[1], duck: cls[2] }, floatMinMargin: minMargin,
  copies: copies.map(c => ({ bits: c.bits, levels: c.levels, disagree: c.disagree, rate: c.disagree / frames, onFloatAction: { none: c.byClass[0], jump: c.byClass[1], duck: c.byClass[2] },
    floatMinMarginAtFlip: Number.isFinite(c.minMarginAtFlip) ? c.minMarginAtFlip : null })) }, null, 1));
