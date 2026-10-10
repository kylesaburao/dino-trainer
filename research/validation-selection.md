# Validation and selection design for late-training ES checkpoints

Compiled 2026-10-10, branch wt/mac. No project code was changed and no trainer was run.

Labels: **V** = read at the source this session (abstract or text actually returned). **V-prior** = verified in an earlier note (`es-practice.md`, `rare-failure-robustness.md`) and not re-read now. **S** = search or summary only. **U** = unverified, textbook knowledge or recollection. **D** = derived here (arithmetic or small simulation; script in scratchpad, not committed; Monte Carlo figures are +/-0.001). **INF** = my inference about this project.

Relevant code, read this session: `trainer/rules.js` (`isBetter`, `valSummary`, `clopperPearson`, `sequentialAllPass`), `trainer/cli.js` (`recordValidation`, `validate`, `--fold-validation`), `trainer/env.js` (`validationSeeds(n)` extends the 25-seed set with the same formula; held-out seeds are a disjoint range). This note extends `rare-failure-robustness.md` section 5 (which already gives the winner's-curse table, Bonferroni table and Hoeffding widths); it does not repeat them.

## 1. Summary

1. **The fixed validation seeds make "noise" a selection-overfit problem, not a resampling problem.** [INF, from code] Validation seeds are fixed and the game is deterministic, so a given theta always scores the same. Every candidate already sees the same seeds (common random numbers, section 7). What varies is the seed *sample*: best-so-far is a maximum over every generation ever validated, all on the same 36 games, so the bias grows with the number of looks. A 12/36 to 32/36 swing is also not binomial noise at p about 0.97 (at p = 0.7, n = 36 the sd is 2.7 games; a 20-game range is about 7 sd). INF: most of that swing is real theta jitter between iterates, which favours averaging (section 5) as well as selection.
2. **At p about 0.97 a 36-seed screen cannot rank anything.** [D] P(36/36) = 0.33 at p = 0.97 and 0.70 at p = 0.99, so most good checkpoints tie. `isBetter` then resolves ties by newest (`select count`, full count) or by `meanLog` (which is constant when every game is capped), i.e. essentially arbitrarily.
3. **Separating p = 0.97 from 0.99 costs about 300-500 games per candidate** (test against a known threshold, section 4) and about 540-1050 for a head-to-head comparison of two unknowns. That is 10-30x the current 36.
4. **Successive halving or a screen-then-confirm design fits the budget.** [D] With 40 checkpoints of true p about 0.97 +/- 0.01, a 12-seed screen followed by 100 fresh seeds on the top 8 costs 1,280 games (single-stage 36 seeds: 1,440) and picks a winner whose true p is 0.978 versus 0.974 for single-stage (observed 1.000 for the single-stage winner: a +2.6 point winner's curse). The gain is real but small (about 0.4 point, a 15% cut of the failure rate), and the chance of picking the actual best of 40 stays near 10%.
5. **Report from fresh seeds only.** Any stage whose seeds took part in a selection is biased upward. Rank on stage-2 results alone (or a lower confidence bound on them), never on pooled stage-1 + stage-2 counts.
6. **Weight averaging is the only option with a plausible effect larger than selection, and the only one with no ES evidence.** Recorded as an experiment, not a recommendation.

## 2. Two-stage validation, racing, successive halving, elite re-evaluation

- V-prior (Such et al. 2017, arXiv 1712.06567, SI): the top 10 of each generation are re-evaluated on 30 additional episodes before the elite is named; those frames count as training cost. This is a two-stage design with a fixed k = 10 and n2 = 30. A re-fetch this session returned an unreadable compressed PDF, so the wording was not re-read.
- V (Maron and Moore, Hoeffding Races, NeurIPS 1993; abstract page read in the earlier note): keep Hoeffding intervals per model, eliminate a model once its best possible error is worse than the best model's worst possible error, union bound for the confidence. INF: for Bernoulli outcomes with p near 0.97 the Hoeffding width (0.226 at n = 36, `rare-failure-robustness.md` 5.1) is about 8x the actual sd (0.028), so a Hoeffding race eliminates nothing before several hundred games. Use exact binomial or Wilson/Clopper-Pearson intervals in the race.
- U/S (Birattari et al., F-Race, GECCO 2002; Friedman-test racing): not fetched. F-Race ranks candidates across the *same* instances (blocks), which is the CRN structure used here. Not needed given binary outcomes: with discordant-pair counts, a sign test (McNemar) is the Friedman test for two candidates.
- S (Heidrich-Meisner and Igel, ICML 2009): Hoeffding and Bernstein races inside CMA-ES to set episodes per candidate (abstract only, from the earlier note).
- U (Mnih, Szepesvari, Audibert, Empirical Bernstein Stopping, ICML 2008): an attempt to read the PDF this session returned unreadable binary. Claim carried from the earlier note (S): Bernstein bounds beat Hoeffding because they use the sample variance. For p near 0.97 the variance term is about 0.03, so the gain is large, but a binomial interval is simpler.
- V (Jamieson and Talwalkar 2015, arXiv 1502.07943; abstract): hyperparameter optimisation as non-stochastic best-arm identification; "comparable test accuracies an order of magnitude faster than baseline methods". The abstract does not name successive halving; INF: that is the algorithm they analyse (recollection, U). V (Li et al., Hyperband, arXiv 1603.06560; abstract): random search with adaptive resource allocation and early stopping, "over an order-of-magnitude" speedup claimed. Caveat INF: these papers' arms are *monotone training curves* where a cheap budget predicts a larger one. Here the "budget" is seeds, and a seed's outcome is an i.i.d.-like Bernoulli draw, so the stage-1 signal is an unbiased but very noisy estimate of the same quantity, which is the classic stochastic best-arm setting.
- V (Even-Dar, Mannor, Mansour, JMLR 7:1079-1105, 2006; abstract): action elimination with confidence intervals; about O((n/eps^2) log(1/delta)) pulls find an eps-optimal arm among n. V (Kaufmann, Cappe, Garivier, arXiv 1407.4443; abstract): fixed-confidence and fixed-budget complexity, matching algorithms for Bernoulli two-armed bandits, improved sequential stopping rules. INF: for n = 40 checkpoints and eps = 0.01 the Even-Dar order is 40/0.0001 x ln(1/delta) = 400,000 x ln(20) games times constants. Resolving the best of 40 within 1 point is unaffordable. The problem is to find *a* checkpoint close to the best, not the best.
- INF (design): the cheap move is an elimination that is one-sided and aimed at the failure mode. A checkpoint that fails any of its first m games on a *hard-seed* set is dropped. `sequentialAllPass` in `rules.js` already does this for held-out seeds.

## 3. Winner's curse and its fixes

- S/U (Smith and Winkler 2006, "The Optimizer's Curse"; abstract and correction notice via search, `rare-failure-robustness.md` R13): picking the top of unbiased noisy estimates yields a realised value below the estimate. Bayesian shrinkage proposed as remedy.
- D (this note, Monte Carlo, 4,000 trials, K = 40 checkpoints, true p about 0.97, sd 0.01 across checkpoints, independent): winner of 36 seeds has observed 1.000 and true 0.974. Note the true mean of the 40 is 0.970, so the selected winner gains only +0.4 point over a random pick, while the report claims +3 points. Uniform 36 seeds: 0.973; 12/top-8/100: 0.978; 36/top-4/100: 0.979; 6/top-10/100: 0.978; oracle best about 0.990 (the clipped distribution makes the oracle approximate).

Fixes, in the order of cost:

1. **Holdout re-evaluation (fresh seeds).** Select on stage 1, then rank the finalists on stage 2 results alone. Unbiased for the finalist's true p; bias then comes only from choosing among the k finalists on stage 2, which is a max over k = 4-8 instead of 40. A third untouched set (held-out) certifies. This is the Such et al. design.
2. **Lower confidence bound.** Clopper-Pearson one-sided 95% lower bound on p for a clean sweep: 0.920 at 36/36, 0.970 at 100/100, 0.990 at 300/300 (D: 0.05^(1/n)). Ranking by LCB penalises small n and makes unequal-n comparison fair. With equal n it equals ranking by count, so it only matters when n differs between candidates (racing, early abort) or across stages.
3. **Shrinkage.** Empirical-Bayes posterior mean (k + a)/(n + a + b), with Beta(a, b) fitted to the pass counts of all checkpoints (U, textbook, Efron and Morris; not fetched). It is monotone in k for fixed n, so it changes the ranking only when n differs. Add only if unequal n is used. INF: for equal n it is a reporting tool, not a selection tool.
4. **Do not pool.** Pooling stage-1 and stage-2 counts re-introduces the bias of the stage-1 selection into the estimate.

## 4. Budget arithmetic

Notation: failure rate q = 1 - p; p0 = 0.97, p1 = 0.99; one game costs c (about 1.9M frames for a pass, less for a failure that ends early, so cost per pass is the upper bound).

**A. Test one candidate against a threshold** (reject "p <= 0.97", accept "p >= 0.99"), normal approximation:
n = [ z_a sqrt(p0 q0) + z_b sqrt(p1 q1) ]^2 / (p1 - p0)^2.
- alpha = 0.05, beta = 0.05: z = 1.645 each; sqrt(0.97 x 0.03) = 0.1706, sqrt(0.99 x 0.01) = 0.0995; sum x 1.645 = 0.4442; squared 0.1973; / 0.0004 = **494 games**.
- alpha = 0.05, beta = 0.20 (z_b = 0.842): **332**. alpha = beta = 0.10 (z = 1.282): **300** (D).
- Exact SPRT (Wald, U for the textbook form) with these p0, p1: a pass adds ln(0.99/0.97) = +0.0204, a failure -1.0986 (toward p0). Boundary ln(0.95/0.05) = 2.944. A clean run needs **144 passes** to accept p1; each failure costs 54 further passes. Average sample number about 319 under p1 and 224 under p0 (D). So the sequential test is no cheaper than the fixed-n test in expectation for a good candidate; it saves mostly on bad ones, which are rejected after about 25-30 games if failures are at 3% (one failure at game 10, say, plus slow recovery).

**B. Compare two unknown candidates** (p = 0.97 versus 0.99), delta = 0.02, one-sided alpha = 0.05, beta = 0.05, (z_a + z_b)^2 = 10.82:
- Independent samples: n per candidate = 10.82 x (0.0291 + 0.0099) / 0.0004 = **1,055**.
- Paired on the same seeds (CRN) with discordant fraction d: n = 10.82 x d / delta^2. d = 0.02 (the minimum possible: only the better candidate's wins): **541**; d = 0.03: **811**; d = 0.05: 1,352. (alpha = 0.05, beta = 0.20, (z_a + z_b)^2 = 6.18: independent 603; paired d = 0.02: 309; d = 0.03: 463.) All D, normal approximation, which is rough at these rates.
- Reading: pairing helps only if the failures sit on shared hard seeds (d close to delta). It cannot beat the Bernoulli floor of 0.02 x 10.82 / 0.0004 = 541 games.

**C. What the screen/confirm design buys per 1,440 games** (the current 36-seed spend over 40 looks): D simulation in section 3 above. A screen of 12 seeds has sd 0.049 on p, 5x the checkpoint spread of 0.01, so stage 1 mostly eliminates *bad* checkpoints (those with p near 0.9) and cannot rank the good ones. Stage 2 with 100 seeds (sd 0.017) still cannot resolve a 0.01 spread. The 0.4-point improvement comes from dropping obvious losers and from not trusting an all-pass tie.

**D. Time horizon.** If late training runs 1,000 generations with fold validation at 36 seeds, that is 36,000 games. One 300-seed certification of a single finalist is under 1% of that. INF: spending on a few high-n confirmations of 3-5 finalists (about 100-300 each) is cheap next to continuous 36-seed validation, which suggests validating less often and with more seeds, rather than every generation.

## 5. Checkpoint averaging and EMA of theta

- V-prior (Izmailov et al. SWA, arXiv 1803.05407): averaging SGD iterates under cyclical or high constant lr gives flatter solutions, one network. V-prior (Nikishin et al. 2018 workshop): SWA in A2C/DDPG stabilises; 3 repeats; CrazyClimber got *worse* (147,030 to 139,752). V-prior (Wortsman et al. 2022, model soups): uniform soup can lose to the best single model; greedy soup (add a model only if held-out accuracy does not fall) is no worse than the best single model on the set used to build it.
- U (Polyak and Juditsky 1992; Ruppert): iterate averaging attains the optimal asymptotic rate for stochastic approximation in the convex, decreasing-step regime. Not fetched.
- U (negative): no paper on Polyak/EMA/SWA of ES iterates found in two earlier searches (`rare-failure-robustness.md` 6.1); not re-searched now. Karpathy's "A Recipe for Training Neural Networks" (V, fetched this session) does not mention weight averaging or EMA. It says ensembles "are a pretty much guaranteed way to gain 2% of accuracy on anything", early stopping on validation loss, and not to rely on smoothing noisy curves. Ensembles do not ship (CSS network page).
- INF (ES specifics): with Adam at lr 0.03 the iterate moves a bounded step (about lr per coordinate) every generation, and the gradient estimate has noise from 128 x 4 episodes. The iterate is therefore a random walk around a ridge, plausible with the large jitter in point 1. An EMA with horizon H generations averages that jitter away and costs one extra vector. Ranking by validation then also becomes more stable, since the EMA moves slowly. The risk is the Nikishin result: an average of two different solutions can sit between basins. For a tanh MLP with no normalisation layers there is no BN recompute. Greedy-soup rule applies: accept an average only if it scores at least as well as the best member on fresh seeds.
- INF: EMA does not touch the training loop if it is kept as a separate vector, so default behaviour stays bit-identical.

## 6. Pooled "late-phase" protocol for a fixed budget (INF)

Three seed sets, all disjoint from training, from `env.js`: screen S (validation range, first 36), confirm C (validation range, next 100-300), certify H (held-out range, offset past the stop-check seeds). A run keeps a top-k pool by screen result (distinct generations, at least 10 apart so they are not near-duplicates), re-scores the pool on C at the end (or when the pool changes and budget allows), ranks by C alone (count, then LCB), and certifies the single winner on H with `sequentialAllPass`. Total late spend per 1,000 generations: 36,000 (screen) + k x |C| (about 1,000-2,000) + |H| (about 300).

## 7. Common random numbers

- U (Law and Kelton, Simulation Modeling and Analysis; Glasserman and Yao): CRN reduces Var(X - Y) = 2s^2(1 - rho) when the two estimators are positively correlated across the shared random inputs. Not fetched.
- INF for this problem: the current validation already uses CRN (same fixed seeds for every checkpoint) and a stage-2 set should as well. The variance benefit shows up only in comparisons (section 4B: from 1,055 down to 541-811 per candidate), not in single-candidate estimates. Success correlation across checkpoints of one run is likely high (shared hard seeds), so a CRN gain is expected, but it is not measured.
- Cost of CRN: all candidates share the same hard seeds, so selection learns *those* seeds. This is the overfitting in point 1. The fix is not to give up CRN but to change the seed set between stages and keep a certification set unseen.
- Cheapest measure of rho: replay two existing late checkpoints (adjacent saved thetas) on the same 100 fresh seeds and count discordant pairs (static probe, no training; Section 8, experiment 1).

## 8. Decisions this changes

All flags are described only; none is implemented. Each defaults so that trainer output is bit-identical to today (checked with `node test/bit-identity.js <old> <new>`; the core hash is untouched because only `trainer/` changes).

**Premise checks first (seconds-scale, no trainer):**
1. From `runs/v2-es-d2w32-s1-t1m/metrics.*`, tabulate per-generation validation counts in the last 300 generations: fraction of generations at 36/36, and lag-1 autocorrelation of the counts. Tests point 1 of section 1: if counts are strongly autocorrelated and range widely, jitter is real theta movement (supports EMA); if they hover near 33-36 with little autocorrelation, they are sampling noise (supports confirm stage).
2. Replay 2-3 saved thetas (best.json at different generations if available) on 100 fresh validation-range seeds; report counts and discordant pairs (gives rho and d for section 4B). If a decisive test cannot be run because only one theta exists, the pool design needs the checkpoint-dump flag first (see B).

**A. `--val-confirm K:N` (default off).**
- Keep a pool of up to K candidates (params plus stage-1 result, at least `--val-confirm-gap` generations apart), then re-score them on N seeds from the validation range beyond `--val-seeds` (that range extends with the same formula and is disjoint from held-out), and write the winner to `best.json`. Pool size K = 4-8, N = 100-300. Run at the end of the run, on SIGINT, and every M generations. Rank by stage 2 only.
- Default: no pool, `recordValidation` as it is today.
- Cheapest experiment: offline, no new training. Take the saved late-run thetas (needs a dump; otherwise resume the shipped run for 200 generations in a scratch directory once CPUs are free, dumping theta every 10 generations to a separate file), score all (about 20) on the 36-seed screen and on 300 fresh seeds, and compare the screen's top-1 to its rank by fresh-seed count. Positive control: include one deliberately worse checkpoint (an early one at p about 0.8); the screen must rank it last. Decision: adopt if the top-1 by screen is not in the top 25% on fresh seeds in at least 2 of 3 resumptions (that is, the screen is not already enough).

**B. `--select lcb` (default `median`, so unchanged).**
- Rank by the Clopper-Pearson lower bound `clopperPearson`-style on stage-2 count and n; ties by `meanLog`. Equivalent to `count` when n is equal; only matters with unequal n (early abort, racing).
- Cheapest experiment: unit test in `test/trainer-rules.test.js` on synthetic counts; no run needed. Real effect only visible together with A.

**C. `--val-race P` (default off).** Early abort of stage-2 candidates by `sequentialAllPass` or an SPRT with p0 = 0.97, p1 = 0.99 (boundaries from section 4A), for pool members that clearly lose, to save games.
- Cheapest experiment: simulate the stopping rule on the replayed per-seed outcomes from premise check 2 (no new games). Report games saved and the number of wrongly dropped candidates against the fixed-n answer.

**D. `--ema-theta H` and `--val-ema` (default off).**
- Maintain EMA theta with decay 1 - 1/H (H about 20-50) as a separate vector excluded from the training state hash, and validate it as an additional candidate in the same validation call. Accept it into best.json only under the greedy rule: its stage-2 count is at least that of the best raw candidate.
- Cheapest experiment: offline, from the same dumped late thetas as A: average the last 5, 10, 20 and score the averages versus the members on 300 fresh seeds with CRN. Positive control: averaging two checkpoints from different runs (different basins) should score worse; if it does not, the test cannot detect harm. Null scope: one run, one phase, one commit.

**E. Seed-count rule for `--stop-rule`.** No flag change; document that `allpass:N` should use N from `rare-failure-robustness.md` section 5.3 for the number of finalists K, and that certification must start at a held-out offset past the stop-check seeds.

**Not recommended:** a Hoeffding race (too loose, section 2); pooling stage counts (biased); ranking at 12 seeds only (sd 0.049, 5x the checkpoint spread).

## 9. References

| # | Reference | URL | Status |
|---|---|---|---|
| R1 | Petroski Such et al., Deep Neuroevolution, arXiv:1712.06567 | https://arxiv.org/abs/1712.06567 | V-prior (top-10, 30 episodes); PDF re-fetch this session unreadable |
| R2 | Maron, Moore, Hoeffding Races, NeurIPS 1993 | https://papers.neurips.cc/paper_files/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html | V-prior |
| R3 | Jamieson, Talwalkar, Non-stochastic Best Arm Identification and Hyperparameter Optimization, arXiv:1502.07943 | https://arxiv.org/abs/1502.07943 | V (abstract) |
| R4 | Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, Hyperband, arXiv:1603.06560 | https://arxiv.org/abs/1603.06560 | V (abstract) |
| R5 | Even-Dar, Mannor, Mansour, Action Elimination and Stopping Conditions, JMLR 7:1079-1105, 2006 | https://www.jmlr.org/papers/v7/evendar06a.html | V (abstract) |
| R6 | Kaufmann, Cappe, Garivier, On the Complexity of Best Arm Identification, arXiv:1407.4443 | https://arxiv.org/abs/1407.4443 | V (abstract) |
| R7 | Mnih, Szepesvari, Audibert, Empirical Bernstein Stopping, ICML 2008 | https://icml.cc/Conferences/2008/papers/523.pdf | U (PDF unreadable via fetch; the fetched file itself was not identified) |
| R8 | Heidrich-Meisner, Igel, Hoeffding and Bernstein Races, ICML 2009 | https://mlanthology.org/icml/2009/heidrichmeisner2009icml-hoeffding | S (abstract only, prior note) |
| R9 | Birattari, Stutzle, Paquete, Varrentrapp, A Racing Algorithm for Configuring Metaheuristics (F-Race), GECCO 2002 | not fetched | U |
| R10 | Smith, Winkler, The Optimizer's Curse, Management Science 52(3), 2006; correction 2007 | https://fds.duke.edu/db/aas/stat/faculty/bob.winkler/publications/278471 | S (prior note) |
| R11 | Izmailov et al., Averaging Weights Leads to Wider Optima (SWA), arXiv:1803.05407 | https://arxiv.org/abs/1803.05407 | V-prior (partial) |
| R12 | Nikishin et al., Improving Stability in Deep RL with Weight Averaging, UDL 2018 | https://www.gatsby.ucl.ac.uk/~balaji/udl-camera-ready/UDL-24.pdf | V-prior |
| R13 | Wortsman et al., Model soups, 2022 | https://arxiv.org/abs/2203.05482 | V-prior (via earlier note) |
| R14 | Polyak, Juditsky, Acceleration of Stochastic Approximation by Averaging, SIAM J. Control Optim. 30(4), 1992 | not fetched | U |
| R15 | Karpathy, A Recipe for Training Neural Networks, 2019 | https://karpathy.github.io/2019/04/25/recipe/ | V (via summarising fetch; ensembles, early stopping, no EMA mention) |
| R16 | Law, Kelton, Simulation Modeling and Analysis (CRN chapter); Glasserman, Yao 1992 | not fetched | U |
| R17 | Wald, Sequential Analysis (SPRT); Oxford lecture notes | https://www.stats.ox.ac.uk/~steffen/teaching/bs2siMT04/si14bw.pdf | S (prior note) |

Computations in sections 1 to 4 (D) come from a throwaway Node script in the session scratchpad (normal-approximation sample sizes, SPRT steps, Clopper-Pearson all-pass bounds, a 4,000-trial screen/confirm simulation with independent true p, sd 0.01 around 0.97, clipped to [0.9, 0.999], unseeded Math.random). It is not committed; re-derive before relying on third-decimal values.
