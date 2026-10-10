'use strict';
// Per-generation training policies as pure functions (no I/O, no workers), so
// test/rules.test.js can check them directly: best-model selection, the held-out
// stop check, the stall rule, sigma annealing, (gated) window adaptation,
// geometric-mean fitness, folded validation, and the Clopper-Pearson bound used to
// report certification evals.
const { median, trainSeed, startScores, rampMask, episodeCaps } = require('./env.js');

// One generation's draws from the trainer RNG: popSize/2 noise seeds, then the episode
// seeds, shared by every candidate or (c.pairEpisodes) drawn per antithetic pair; both
// members of a pair always play the same episodes (common random numbers). With the
// curriculum (cur = config with the current window, or null) each seed set also gets
// its start levels and episode ends, a pure function of the seeds (no RNG draws).
// Returns { noiseSeeds, pairEp: [{ seeds, starts, caps[, ramp] }] } (starts/caps null
// without the curriculum; ramp, the natural-ramp episode mask, only when some episode
// starts on the ramp); pairs that share seeds share one object.
function drawGeneration(rng, c, cur) {
  const noiseSeeds = Array.from({ length: c.popSize / 2 }, () => Math.floor(rng() * 4294967296));
  const drawSeeds = () => Array.from({ length: c.episodes }, () => trainSeed(rng));
  const episodes = (seeds) => {
    const starts = cur ? startScores(seeds, cur) : null;
    const ep = { seeds, starts, caps: starts && episodeCaps(starts, cur) };
    const ramp = cur && rampMask(seeds, cur);
    if (ramp) ep.ramp = ramp;
    return ep;
  };
  if (c.pairEpisodes) return { noiseSeeds, pairEp: noiseSeeds.map(() => episodes(drawSeeds())) };
  const shared = episodes(drawSeeds());
  return { noiseSeeds, pairEp: noiseSeeds.map(() => shared) };
}

// Validation summary. count = seeds that reached the target; meanLog = mean of ln(1 + score).
function valSummary(scores, target) {
  return {
    median: median(scores), min: Math.min(...scores), max: Math.max(...scores),
    count: scores.filter(x => x >= target).length,
    meanLog: scores.reduce((a, x) => a + Math.log1p(x), 0) / scores.length,
  };
}

// Does validation summary s replace the current best ({ valMedian, valCount, valMeanLog })?
//   median: strictly higher median (the original rule).
//   count:  lexicographically higher (count, meanLog). A saturated key (every one of
//           nSeeds passed, so meanLog is maximal too) can never increase, which would
//           freeze best; so at saturation a tie also replaces it (prefer the newest θ).
function isBetter(s, best, mode, nSeeds) {
  if (!best) return true;
  if (mode === 'median') return s.median > best.valMedian;
  if (mode !== 'count') throw new Error(`unknown --select ${mode}`);
  const count = best.valCount ?? -1, meanLog = best.valMeanLog ?? -Infinity;
  if (s.count !== count) return s.count > count;
  if (s.count === nSeeds) return s.meanLog >= meanLog;
  return s.meanLog > meanLog;
}

// --stop-rule median | allpass | allpass:N  ->  { kind, seeds } (seeds = held-out seeds played)
function parseStopRule(text) {
  const m = /^(median|allpass)(?::(\d+))?$/.exec(String(text));
  if (!m || (m[1] === 'median' && m[2])) throw new Error(`--stop-rule must be median or allpass[:N], got ${text}`);
  return { kind: m[1], seeds: m[2] ? Number(m[2]) : 25 };
}

// Has this exact best (same params, same target, same held-out seed count) already
// been held-out tested? Re-scoring identical params on identical seeds is
// deterministic, so a retest can never change the outcome.
const alreadyTested = (best, heldOut, target, seeds) =>
  !!(best && heldOut && heldOut.generation === best.generation && heldOut.target === target && heldOut.seeds === seeds);

// Should this generation run a held-out check of best? val = this generation's
// validation summary (null if none ran), nVal = validation seeds it used.
function wantHeldOut(rule, val, best, heldOut, target, nVal) {
  if (!val || !best) return false;
  const triggered = rule.kind === 'median' ? val.median >= target : val.count === nVal;
  return triggered && !alreadyTested(best, heldOut, target, rule.seeds);
}

// allpass held-out check: play seeds in rounds of roundSize, stopping after the first
// round with any failure. score(seedsChunk) -> Promise<scores>.
async function sequentialAllPass(score, seeds, roundSize, target) {
  const scores = [];
  for (let i = 0; i < seeds.length; i += roundSize) {
    const chunk = seeds.slice(i, i + roundSize);
    const s = await score(chunk);
    scores.push(...s);
    if (s.some(x => x < target)) break;
  }
  const failAt = scores.findIndex(x => x < target);
  const passed = scores.filter(x => x >= target).length;
  return {
    played: scores.length, passed, pass: failAt < 0 && scores.length === seeds.length,
    firstFailSeed: failAt < 0 ? null : seeds[failAt], scores,
  };
}

// Antithetic pairs whose two fitness values are equal (pair k = fitness[2k], fitness[2k+1]).
function tiedPairs(fitness) {
  let n = 0;
  for (let k = 0; 2 * k + 1 < fitness.length; k++) if (fitness[2 * k] === fitness[2 * k + 1]) n++;
  return n;
}

// Stall rule: if more than half the pairs tie, skip this generation's Adam step and
// widen sigma for the next one (x1.5, at most 4x base); otherwise back to base sigma.
const STALL_GROW = 1.5, STALL_MAX = 4;
function stallUpdate(stallMult, tiedFrac) {
  if (tiedFrac > 0.5) return { skip: true, stallMult: Math.min(stallMult * STALL_GROW, STALL_MAX) };
  return { skip: false, stallMult: 1 };
}

// Sigma anneal: when >25% of start-0 candidate episodes crash below 200 points but the
// centre survives the opening (latest validation min >= 200), sigma *= 0.8 (floor
// sigmaMin), at most once every 10 generations. Returns the new { sigma, annealedAt }.
const ANNEAL = { below: 200, frac: 0.25, factor: 0.8, cooldown: 10 };
function annealUpdate({ sigma, annealedAt }, { zeroStartScores, valMin, generation, sigmaMin }) {
  const unchanged = { sigma, annealedAt };
  if (!zeroStartScores.length || valMin == null || valMin < ANNEAL.below || sigma <= sigmaMin) return unchanged;
  if (annealedAt != null && generation - annealedAt < ANNEAL.cooldown) return unchanged;
  const crashed = zeroStartScores.filter(x => x < ANNEAL.below).length / zeroStartScores.length;
  if (crashed <= ANNEAL.frac) return unchanged;
  return { sigma: Math.max(sigma * ANNEAL.factor, sigmaMin), annealedAt: generation };
}

// Window adaptation: keep the late-window pass rate between 30% and 60%.
const WINDOW_MIN = 5000;
function adaptWindow(window, passRate, target) {
  const lo = Math.min(WINDOW_MIN, target);
  if (passRate > 0.6) return Math.min(target, Math.round(window * 1.5));
  if (passRate < 0.3) return Math.max(lo, Math.round(window / 1.5));
  return window;
}

// Gated window adaptation, decided every few generations on pooled counts
// acc = { zeroPassed, zeroTotal, latePassed, lateTotal }: while start-0 episodes mostly
// fail (pass rate < 30%) the window shrinks, however the late windows go; only once the
// opening is learned (> 60%, or no start-0 episodes) does the late pass rate move it.
// A shrink never grows a window that is already below adaptWindow's floor.
function gatedAdaptWindow(window, acc, target) {
  const shrink = () => Math.min(window, adaptWindow(window, 0, target));
  const zeroRate = acc.zeroTotal > 0 ? acc.zeroPassed / acc.zeroTotal : null;
  if (zeroRate !== null && zeroRate < 0.3) return shrink();
  if ((zeroRate === null || zeroRate > 0.6) && acc.lateTotal > 0) {
    const lateRate = acc.latePassed / acc.lateTotal;
    return lateRate < 0.3 ? shrink() : adaptWindow(window, lateRate, target);
  }
  return window;
}
const emptyAdaptAcc = () => ({ zeroPassed: 0, zeroTotal: 0, latePassed: 0, lateTotal: 0 });

// Geometric-mean curriculum fitness over one candidate's episodes: exp(mean ln max(f_j,
// 1/span_j)). A failing stage pulls it towards 0 (dF/df_j is proportional to F/f_j), so
// the opening cannot be traded for late-window progress. The floor 1/span_j is one point.
function geomeanFitness(fs, spans) {
  let s = 0;
  for (let j = 0; j < fs.length; j++) s += Math.log(Math.max(fs[j], 1 / spans[j]));
  return Math.exp(s / fs.length);
}

// Folded validation: the generation's pool results are popSize training candidates
// followed by the validation games of the pre-update θ.
const splitFolded = (results, popSize) => ({ train: results.slice(0, popSize), val: results.slice(popSize) });

// Two-sided 95% Clopper-Pearson interval for k successes in n trials.
function clopperPearson(k, n, alpha = 0.05) {
  const lgamma = (x) => {   // Lanczos
    const c = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1;
    let a = 0.99999999999980993;
    for (let i = 0; i < 8; i++) a += c[i] / (x + i + 1);
    const t = x + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  };
  const logC = (j) => lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1);
  const pmf = (j, p) => (p <= 0 ? (j === 0 ? 1 : 0) : p >= 1 ? (j === n ? 1 : 0) : Math.exp(logC(j) + j * Math.log(p) + (n - j) * Math.log1p(-p)));
  const cdf = (j, p) => { let s = 0; for (let i = 0; i <= j; i++) s += pmf(i, p); return s; };   // P(X <= j)
  const bisect = (f) => { let lo = 0, hi = 1; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (f(m)) lo = m; else hi = m; } return (lo + hi) / 2; };
  // lower: P(X >= k | p) = alpha/2 ; upper: P(X <= k | p) = alpha/2
  const lower = k === 0 ? 0 : bisect((p) => 1 - cdf(k - 1, p) < alpha / 2);
  const upper = k === n ? 1 : bisect((p) => cdf(k, p) > alpha / 2);
  return { lower, upper };
}

module.exports = {
  drawGeneration, valSummary, isBetter, parseStopRule, alreadyTested, wantHeldOut, sequentialAllPass,
  tiedPairs, stallUpdate, STALL_MAX, annealUpdate, adaptWindow, gatedAdaptWindow, emptyAdaptAcc, geomeanFitness, splitFolded, clopperPearson,
};
