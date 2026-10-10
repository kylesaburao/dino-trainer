'use strict';
// Summarise a teacher-duck survey JSONL (research/prereg-teacher-duck.md §4–§5).
// Usage: node research/scripts/prereg-teacher-duck/analyze.js <survey.jsonl> [--json]
// Totals; crash rate with the one-sided 95% Clopper-Pearson upper bound; DOWN-held share of
// frames; share of obstacles ducked; mid pteros encountered / ducked (game-side); and the
// solver's fallback share when it exposes counters (solverStats: midEncountered, midDucked,
// fallback).
const fs = require('fs');
const path = require('path');
const { clopperPearson } = require(path.join(__dirname, '..', '..', '..', 'trainer', 'rules.js'));

const file = process.argv[2];
const rows = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const sum = (k) => rows.reduce((a, r) => a + (r[k] || 0), 0);
const n = rows.length, crashes = rows.filter(r => r.crashed);
const upper = n ? clopperPearson(crashes.length, n, 0.10).upper : null;   // two-sided 90% upper = one-sided 95%
const stats = rows.map(r => r.solverStats).filter(Boolean);
const ssum = (k) => stats.reduce((a, s) => a + (s[k] || 0), 0);
const seeds = rows.map(r => r.seed);
const out = {
  file: path.basename(file), seeds: n, seedRange: n ? [Math.min(...seeds), Math.max(...seeds)] : null,
  crashes: crashes.length, crashRate: crashes.length / n, crashRateUpper95oneSided: upper,
  reachedCap: rows.filter(r => !r.crashed).length,
  frames: sum('frames'), downShare: sum('downFrames') / sum('frames'),
  obstaclesPassed: sum('obstaclesPassed'), obstaclesDuckedShare: sum('obstaclesDucked') / sum('obstaclesPassed'),
  midEncountered: sum('midEncountered'), midPassed: sum('midPassed'), midDucked: sum('midDucked'),
  midDuckedShare: sum('midPassed') ? sum('midDucked') / sum('midPassed') : null,
  solver: stats.length ? {
    seedsWithStats: stats.length, midEncountered: ssum('midEncountered'), midDucked: ssum('midDucked'), fallback: ssum('fallback'),
    fallbackShare: ssum('midEncountered') ? ssum('fallback') / ssum('midEncountered') : null,
  } : null,
  crashList: crashes.map(r => ({ seed: r.seed, score: r.score, onScreen: r.onScreen, lastFrame: r.last30 && r.last30[r.last30.length - 1] })),
};
if (process.argv.includes('--json')) { console.log(JSON.stringify(out, null, 1)); process.exit(0); }
console.log(`${out.file}: ${n} seeds ${out.seedRange?.join('–')}`);
console.log(`crashes ${out.crashes} (rate ${out.crashRate.toFixed(5)}, one-sided 95% CP upper ${upper?.toFixed(5)}); reached cap ${out.reachedCap}`);
console.log(`frames ${(out.frames / 1e9).toFixed(2)}B; DOWN held ${(out.downShare * 100).toFixed(3)}% of frames; obstacles ducked ${(out.obstaclesDuckedShare * 100).toFixed(3)}% of ${out.obstaclesPassed}`);
console.log(`mid pteros (game-side): encountered ${out.midEncountered}, passed ${out.midPassed}, ducked ${out.midDucked}${out.midDuckedShare != null ? ` (${(out.midDuckedShare * 100).toFixed(2)}%)` : ''}`);
if (out.solver) console.log(`solver counters: mid encountered ${out.solver.midEncountered}, ducked ${out.solver.midDucked}, fallback ${out.solver.fallback} (${out.solver.fallbackShare != null ? (out.solver.fallbackShare * 100).toFixed(2) + '%' : '-'})`);
else console.log('solver counters: none exposed');
for (const c of out.crashList) console.log(`CRASH seed ${c.seed} score ${c.score}: ${JSON.stringify(c.onScreen)}`);
