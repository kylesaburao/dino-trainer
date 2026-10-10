'use strict';
// The sweep grid (research/prereg-bc-sweep.md §2) and its exact size/cost formulas.
// MACs per decision = 44w + (d-1)w^2 + 3w (weights only); parameters = MACs + d*w + 3 (biases).
// Training FLOPs = 6 x MACs x samples processed (2 forward + 4 backward per MAC: a standard
// approximation). Samples per epoch = floor(rows / batch) * batch (the remainder is dropped).
const DEPTHS = [1, 2, 4, 8, 16, 32, 64];
const WIDTHS = [8, 16, 32, 64];
const ROWS = 2096268, BATCH = 1024, EPOCHS = 15;
const SAMPLES_PER_EPOCH = Math.floor(ROWS / BATCH) * BATCH;
const SHIPPED_MACS = 2528;

const macs = (d, w) => 44 * w + (d - 1) * w * w + 3 * w;
const params = (d, w) => macs(d, w) + d * w + 3;
const flops = (d, w, epochs) => 6 * macs(d, w) * SAMPLES_PER_EPOCH * epochs;
const name = (d, w, seed = 1) => `d${d}-w${w}-s${seed}`;

const configs = [];
for (const w of WIDTHS) for (const d of DEPTHS) {
  const m = macs(d, w);
  configs.push({ d, w, macs: m, params: params(d, w), tierBEligible: m <= 2 * SHIPPED_MACS, s2Candidate: m < SHIPPED_MACS });
}

module.exports = { DEPTHS, WIDTHS, ROWS, BATCH, EPOCHS, SAMPLES_PER_EPOCH, SHIPPED_MACS, macs, params, flops, name, configs };

if (require.main === module) {
  for (const c of configs) console.log(`${name(c.d, c.w)}\tMACs ${c.macs}\tparams ${c.params}\ttrainFLOPs ${flops(c.d, c.w, EPOCHS).toExponential(3)}` +
    `${c.tierBEligible ? '\ttierB-eligible' : ''}${c.s2Candidate ? '\tS2' : ''}`);
}
