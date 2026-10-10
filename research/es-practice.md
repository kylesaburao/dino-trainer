# ES and neuroevolution practice for small policy networks

Compiled 2026-10-10, branch wt/mac. No project code was changed and no trainer was run.

Labels used throughout:
- **STATED (source)**: read in the source text or code (paper PDF text extracted locally, raw repository file, or a fetched page). Quotes are short and verbatim only where marked with quotation marks.
- **STATED (via summary)**: the page was fetched through a summarising tool, not as raw text. The fact is as reported by that tool.
- **INFERENCE**: my reasoning about this project. Not from any source.
- **UNVERIFIED**: could not be confirmed from source text in this session.

Project facts (plateau ~54 points, escape ~776 generations, σ≈0.1, Adam lr 0.03, 44 features, population 128 x 4 episodes) come from the task brief and `HANDOFF.md` 4.2-4.5. I did not re-measure them. This note does not repeat `research/karpathy-training-techniques.md` (VBN, weight-decay rationale, the duplicate-feature width argument, saturation diagnostics) or the literature list in `HANDOFF.md` 4.5. It cites them where a new fact bears on them.

Code was read from: openai/evolution-strategies-starter and modestyachts/ARS (both `master`, accessed 2026-10-10, no SHA recorded), RobertTLange/evosax at `d4b4127708cb2cd0cc906b910ddcd091013153a9` (main on 2026-10-10), google/evojax at `acf5252475e4fbf47e0e3f7994c03de11dd3c4ca`, hardmaru/estool (`master`). Paper PDFs were downloaded and text-extracted locally (macOS PDFKit), then grepped.

---

## 1. Summary: the decision-relevant takeaways

1. **Tie handling differs between reference implementations, and the split matches HANDOFF 4.2 exactly.**
   - STATED (es-starter `compute_ranks`; estool `compute_ranks`; EvoJAX PGPE `process_scores`): `ranks[x.argsort()] = arange(n)`. Tied returns receive distinct consecutive ranks, in an order the sort decides. This is the behaviour the shipped trainer has, and HANDOFF found its index-tie random walk was what escaped the plateau.
   - STATED (evosax at d4b4127, `_rank_fitness`): `jax.scipy.stats.rankdata`, whose documented default is `method='average'` (midranks). HANDOFF measured midranks with coupled L2 as not escaping by gen 900 (seed 1), while midranks with decoupled decay 0.1 escaped at about the baseline time; the brief reports midranks with decoupled decay 0.01 not escaped by gen 400 on 3 seeds. So midranks alone are not ruled out.
   - STATED (ARS code): no ranks. Pair differences are scaled by the reward standard deviation, with optional top-b directions.
   - INFERENCE: ties are not a bug in the references. Whether they hurt or help depends on the phase. At the plateau, random tie-breaking is accidental exploration. Near the cap (many candidates scoring identically), the same randomness is pure noise.
   - Caveat: numpy's default `argsort` is not stable, so "ordered by index" is an implementation detail. Do not rely on it.

2. **No source solves "most antithetic pairs tie" directly. The closest published rule is CMA-ES's flat-fitness response.**
   - STATED (Hansen tutorial, appendix B.4 and the listing, lines 92-96): `if arfitness(1) == arfitness(ceil(0.7*lambda)) sigma = sigma * exp(0.2+cs/damps)`. That is, when the best value equals the 70th-percentile value, step size is multiplied by exp(0.2+cs/damps).
   - STATED (same page): "observation of a flat fitness should be rather a termination criterion".
   - INFERENCE: a stagnation-triggered σ escalation keyed on the tied fraction is the cheapest plateau experiment in this note. Lehman et al. (item 3) give the supporting reason: larger σ crosses flat regions.

3. **Larger σ crosses zero-fitness gaps, smaller σ gets stuck.**
   - STATED (Lehman et al. 2018, section 3 landscapes): in "Gradient Gap", σ = 0.18 traverses a zero-fitness non-differentiable gap while lower σ and finite differences cannot. In "Fleeting Peaks", σ = 0.16 ignores the local optima, σ = 0.048 hops between them and sticks on the last, σ = 0.002 stays at the first "indefinitely".
   - These are 2-D toy landscapes. The paper says ES optimises the *distribution*, not the point.

4. **NSRA-ES's adaptive rule is reusable without a behaviour characterisation.**
   - STATED (Conti et al. 2018, SI 6.5): w starts at 1.0. If the best episodic reward has not improved in t_w = 50 generations, w drops by δ_w = 0.05. When a new best appears, w rises by δ_w.
   - STATED (Conti, Atari discussion): on Seaquest, t_w = 50 and δ_w = 0.05 were too slow for 200 generations, so NSRA-ES stayed in ES's local optimum. With t_w = 10 it matched NSR-ES.
   - INFERENCE: "no new best for N generations means shift toward exploration" can drive σ or population instead of a novelty weight. N must be small against a 900-generation budget.

5. **Every reference implementation freezes observation statistics at evaluation and updates them only from training rollouts.**
   - STATED: es-starter (eval rollouts carry `ob_sum=None`), ARS code (`update_filter=False` when `evaluate`), EvoJAX (`if not test` guard), Such et al. (statistics from 10,000 random policies, then fixed).
   - The only effect size found is qualitative: STATED (ARS paper, section 3.2) "We were not able to train a linear policy for this task [Humanoid] without the normalization of the states".
   - INFERENCE: the project's 44 features already sit near unit scale (|x| up to 2.3), so the expected gain is small. The risk is the opposite: rarely-active features get amplified by std normalisation (see section 2).

6. **The only source-stated σ-scales-with-size rule is ClipUp's fixed-radius σ.**
   - STATED (ClipUp, section 4): r = ‖σ0‖ = q·v_max with q in 10-20, default 15. This is a norm over the whole parameter vector.
   - INFERENCE: if σ is uniform across coordinates, per-coordinate σ0 = r/√d. A constant radius therefore shrinks per-coordinate σ as the network grows.
   - STATED (Salimans 2017, section 3.2): duplicating features leaves ES unchanged if σ and lr are both halved. NES (Wierstra et al., Table 1) scales λ = 4+⌊3 log d⌋ and the σ learning rate with d.
   - No library default (evosax, EvoJAX, estool) scales σ with network width. They use per-task constants.

7. **Practical σ and lr schedules are small and multiplicative.**
   - STATED (evosax paper, Table 4, Brax tasks, OpenAI-ES): lr init 0.01-0.02, decay 0.999, final 0.001; σ init 0.05-0.1, decay 0.999, final 0.01; Adam; centred ranks; population 256.
   - STATED (estool OpenES defaults): σ 0.1 decayed by 0.999 to a floor of 0.01; lr 0.01 decayed by 0.9999 to 0.001.
   - STATED (Salimans 2017): "we did not see benefit from adapting σ during training".
   - STATED (Such et al.): the GA annealed σ to 0.001 after 1,000 generations, "which resulted in a small performance boost at the end of training".

8. **Restart triggers in the literature are mostly numerical-stagnation tests, and the population-doubling lever is the part that transfers.**
   - STATED (Auger and Hansen 2005): IPOP doubles the population per restart (1.5-5 judged reasonable; 2-3 similar on Rastrigin).
   - STATED (Hansen tutorial B.3): EqualFunValues (range of best values over the last 10+⌈30n/λ⌉ generations is zero) and Stagnation (median of the newest 30% of a 20% history window is not better than the oldest 30%).
   - STATED (Loshchilov et al. 2012): BIPOP interleaves doubling restarts with small restarts, λ_small = λ_default·(½·λ_large/λ_default)^(U²), σ0_small = σ0_default·10^(−2U), always choosing the regime that has used fewer evaluations.
   - I found no ES-for-RL paper that applies a restart rule to OpenAI-ES. The seed-portfolio evidence I did find is ARS's 100-seed timing table.

9. **Late-phase noise control has direct precedents.**
   - STATED (Such et al.): the top 10 individuals of each generation are re-evaluated on 30 additional episodes before the elite is named; those frames are counted as training cost.
   - STATED (Hardmaru, via summary): fitness is "the average of 16 random rollouts" so lucky terrains do not carry an agent. He notes the data cost, "16x worse".
   - INFERENCE: for "all held-out seeds reach 1M", selecting the best checkpoint by a few validation seeds is the weak step, not the gradient.

10. **ARS removed a survival bonus because it created a stand-still local optimum.**
    - STATED (ARS paper, section 4.2): the bonus "discourage[s] the exploration of policies that cause falling early" and makes ARS find policies that "stand still for a thousand timesteps". The authors subtract it during training and evaluate with the default reward.
    - INFERENCE: the project forbids reward shaping (the capped score is the reward). This is the nearest precedent for a training-only change that leaves evaluation unmodified, which the project allows in principle (HANDOFF/AGENTS: "Any curriculum is training-only"). It is out of scope here and listed only for completeness.

11. **Small-network game results use small, fixed σ and modest populations, and report overfitting to the evaluation distribution as the pitfall.**
    - STATED (estool config): SlimeVolley uses 12 inputs, hidden [20, 20] tanh, 3 outputs. Example command: CMA-ES default optimiser, `--sigma_init 0.5 -e 8`, 8 workers x 4 trials = population 32 (via `train.py` arithmetic).
    - STATED (slimevolleygym TRAINING.md, via summary): earlier agents "had somewhat overfit to a particular opponent's playing style".
    - Hobbyist Chrome-dino GA repos exist (5-8-2 and 6-10-3 networks, population 100, mutation 0.1, σ 0.5; or 3-input population 12) but report no tuning data (STATED via summary). I did not find a Flappy Bird source worth citing (UNVERIFIED snippets only).

12. **Output-layer initialisation in the official ES code is deliberately near-zero.**
    - STATED (es-starter `policies.py`, `tf_util.py`): hidden layers use `normc_initializer(1.0)`, the output layer `normc_initializer(0.01)`, biases zero. `normc` rescales each *column* of the weight matrix (one unit's incoming weights) to a given L2 norm.
    - INFERENCE: with a 0.01-norm output layer the argmax is decided by small input-dependent logit differences rather than by a bias. HANDOFF measured the project's init policy as 52% none / 47% duck. This is a cheap init-level experiment (table row 7).

---

## 2. Q1: observation normalisation in ES and ARS

### 2.1 OpenAI es-starter (the Salimans MuJoCo setup)
STATED (es-starter `es_distributed/es.py`, `policies.py`, `tf_util.py`, `optimizers.py`, `configurations/humanoid.json`, `master`):
- `RunningStat(shape, eps)` keeps `sum`, `sumsq` (initialised to `eps`) and `count` (= `eps`). Master creates it with `eps=1e-2`. Mean is `sum/count`. Std is `sqrt(max(sumsq/count - mean², 1e-2))`, i.e. a variance floor of 1e-2 (std at least 0.1).
- Initial state: mean 0, std 1, so the first generation sees raw observations.
- Network input: `tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0)`.
- Statistics are collected only on a random fraction of rollouts: `rs.rand() < calc_obstat_prob`, with `calc_obstat_prob` 0.01 in `humanoid.json`. Workers sum observations (`obs.sum(axis=0)`, squares, count) and the master merges them with `ob_stat.increment(...)` for the current task only.
- After each update the master calls `policy.set_ob_stat(ob_stat.mean, ob_stat.std)`. Workers set the same statistics before each task. So within one generation every perturbed policy (including its antithetic partner) sees identical normalisation, and statistics are the cumulative history from the start of training (no window or decay).
- Evaluation: eval rollouts call `policy.rollout(env)` with no observation saving and return `ob_sum=None`, so they never update the statistics. They use the statistics supplied with the task, i.e. frozen at the current generation's values.
- Humanoid config (STATED, `humanoid.json`): `noise_stdev` 0.02, `l2coeff` 0.005, Adam stepsize 0.01, `centered_rank`, hidden dims [256, 256], tanh, `ac_noise_std` 0.01, `episodes_per_batch` 10000.
- Weight decay (STATED): the update is `optimizer.update(-g + l2coeff * theta)`, i.e. coupled L2 inside the Adam input, not decoupled.
- Adam (STATED, `optimizers.py`): bias-corrected, `a = stepsize*sqrt(1-β2^t)/(1-β1^t)`, `step = -a*m/(sqrt(v)+eps)`, β1 0.9, β2 0.999.
- Salimans et al. 2017 (paper text) does not describe observation normalisation for MuJoCo. Its Atari fix is VBN (see karpathy note, takeaway 4). I did not read the paper's supplement hyperparameter tables (UNVERIFIED; `humanoid.json` is the stated source of the numbers above).

### 2.2 ARS V2 (Mania, Guy, Recht 2018)
STATED (paper, Algorithm 2 and section 3.2; ARS code `filter.py`, `ars.py`, `policies.py`):
- V2 policies are `(M ± νδ) diag(Σ)^(-1/2) (x − µ)`. µ and Σ are "the mean and covariance of the 2NH(j+1) states encountered from the start of training"; only the diagonal is tracked. Footnote: a diagonal entry below 1e-8 is set to +∞ so 0/0 is treated as 0. The code uses the threshold 1e-7 (`self.std[self.std < 1e-7] = float("inf")`).
- Implementation: `MeanStdFilter` pushes every training observation into a Welford `RunningStat` (`rs`) and a per-worker `buffer`. `mean` and `std` used in `__call__` change only in `stats_increment()`, which the learner calls once per iteration after merging worker buffers. So statistics are frozen within an iteration.
- Evaluation: `do_rollouts(..., evaluate=True)` sets `self.policy.update_filter = False` ("so that evaluation rollouts are not used for updating state statistics"). The paper adds that evaluation states and rewards "were not used in any form during training".
- Why (STATED, section 3.2): it "is akin to data whitening", ensures "policies put equal weight on the different components of the states", and can be seen as non-isotropic exploration in parameter space (`M̃ = M diag(Σ)^(-1/2)`).
- Reported effect (STATED, section 3.2): Humanoid-v1 could not be trained by a linear policy without it, and "the measured sample complexity of ARS V2 is better on the other MuJoCo locomotion tasks as well". I did not find per-task ablation numbers in the text; the comparison is V1 vs V2 rows of Table 1.
- Limitation (STATED): V2 is impractical on the LQR problem because states grow exponentially for unstable policies.
- ARS also scales steps by the standard deviation σ_R of the rewards used in the update (section 3.1), and contrasts that with ES's ranks plus Adam, saying those "change the direction of the updates".
- Other ARS facts used later: survival bonus subtracted in training; 100-seed timing in Tables 6-7; hyperparameters in Table 9 (for example Humanoid: α 0.02, ν 0.0075, N 230, b 230).

### 2.3 Other uses
- STATED (Such et al. 2017, humanoid locomotion SI): "we normalize each dimension of the input by subtracting its mean and dividing by its standard deviation, which are computed from executing 10,000 random policies" - a static, pre-computed normaliser.
- STATED (EvoJAX `obs_norm.py`, `sim_mgr.py`): `ObsNormalizer` clip 5.0, std clipped to [1e-6, 1e6]. Statistics update only `if not test`. The trainer's `normalize_obs` defaults to False.
- STATED (ClipUp, section 2): "We normalize observations using the running statistics over all the observations received by all the agents until the current iteration." Table 1 lists ObNorm "No" for LunarLanderContinuous; the Humanoid and Walker tables list its use per environment (not verified line by line).
- STATED (Wong et al. 2024, linear policies, Atari RAM): they compute a rolling mean and variance during each run and standardise each observation. For Pendulum, "By applying state normalization, ARS-v2 performs better" than V1.

### 2.4 What this means here
- INFERENCE: all references normalise *because observation scales differ by orders of magnitude* (MuJoCo velocities vs positions). The project's features are hand-crafted and mostly O(1), so a large benefit is not expected.
- INFERENCE (risk): with a cumulative running std, rarely-active features (for example pterodactyl features that are zero for the first ~580 points) have a tiny std early on. es-starter floors the *variance* at 1e-2 (std 0.1), so such a feature could be amplified up to 10x and then clipped at ±5. ARS divides by `std + 1e-8` after sending tiny stds to infinity. Either choice changes what an early policy does the moment a rare feature appears.
- INFERENCE (non-stationarity): cumulative statistics weight states by episode length. In this project episode length ranges from ~54 points to ~1.9M frames, so late-run statistics are dominated by late-game states, and the effective first-layer function drifts while θ is held fixed.
- INFERENCE (a safe form): a static per-feature affine map (mean and std from a fixed batch of observations, as in Such et al.) can be folded into W1 and b1 *outside* `policy.js`, so decisions and the core hash are unchanged. This only works if the optimiser acts on the normalised parametrisation and the fold is applied when handing weights to the unchanged policy code. Whether that also holds for `model.js` export and the CSS page needs checking and is not verified.

---

## 3. Q2: plateau and deceptive-reward escape

### 3.1 Such et al. 2017 (Deep GA)
STATED (arXiv 1712.06567 text, SI Table 2 and SI 7):
- Algorithm: truncation selection with population N, top T parents, offspring `θ' = θ + σε` with uniform parent choice, one unmodified elite, no crossover. The elite is chosen by re-evaluating the top 10 on 30 additional episodes.
- Table 2: Humanoid Locomotion N = 12,500+1, σ = 0.00224, T = 625. Image Hard Maze N = 20,000+1, σ = 0.005, T = 61. Atari N = 1,000+1, σ = 0.002, T = 20. "Many of the unusual numbers were found via preliminary hyperparameter searches in other domains."
- Initialisation: Xavier (zero biases, weights N(0, 1/N_in)). For humanoid they also tried the normalised-columns initialisation of Salimans et al. and found them "qualitatively similar".
- Humanoid details: σ annealed to 0.001 after 1,000 generations; fitness is a mean over 5 episodes; 5 independent runs, median reported.
- Novelty search (GA-NS): archive insertion probability p = 0.01, novelty = average distance to k = 25 nearest neighbours in population plus archive, replacing fitness. On the deceptive Image Hard Maze (the GA and ES fail; GA-NS solves it).
- Near-origin observation: the GA outperforms DQN/ES/A3C in few generations "suggests that many high-quality policies exist near the origin". Random search beat DQN on 3/13, A3C 6/13 and ES on 3/13 Atari games, and "never" the GA. INFERENCE: for a plateau like this project's, sampling many random initialisations (cheap with action-repeat 54-point episodes) is a legitimate baseline, not only a restart tool.
- GA vs ES in deception: the Uber blog (STATED via summary) says ES "converges quickly to a local optimum" in an oxygen task and "never learns to escape the deceptive trap".

### 3.2 Conti et al. 2018 (NS-ES, NSR-ES, NSRA-ES)
STATED (arXiv 1712.06560 text):
- NS-ES: novelty N(θ, A) = mean L2 distance to the k nearest behaviour characterisations in archive A. The archive is fixed during an iteration and only the current θ_t's characterisation is added at the end ("adding those for each sample would inflate the archive and slow the nearest-neighbors computation").
- Meta-population of M agents; each iteration one agent is chosen with probability proportional to its novelty (equation 1: P(θ_m) = N(θ_m, A) / Σ_j N(θ_j, A)) and updated.
- NSR-ES averages reward and novelty gradients after rank-normalising each. NSRA-ES adapts the weight w: w = 1.0 initially (reward only); if best reward has not increased in t_w = 50 generations, w ← max(0, w − δ_w) with δ_w = 0.05; on a new best, w increases by δ_w.
- Humanoid and Atari settings: Adam lr 1e-2, σ = 0.02, n = 10,000 (humanoid) and 5,000 (Atari) samples per generation, M = 5 (humanoid) and 3 (Atari), k = 10, T = 600-800 (humanoid) and 150-300 (Atari). "chosen through an informal hyperparameter search".
- Results (SI Table 2, deceptive humanoid): median final reward ES 5.3, NS-ES 16.5, NSR-ES 31.2, NSRA-ES 37.1. On the non-deceptive isotropic problem ES 8098.5, NSRA-ES 7923.1, NSR-ES 6756.9, NS-ES 6010.5. The text says NS-ES/NSR-ES cost performance where there is no deception and that NSRA-ES "does not pay a cost".
- Seaquest lesson (see takeaway 4): t_w 50 is too slow for 200 generations; every-10 matched NSR-ES.
- The behaviour characterisation is domain-specific (humanoid: final x,y; Atari: RAM state). The projects' analogue would be something like final distance or obstacle-phase statistics. Whether one exists that does not leak the future is unverified. INFERENCE: anti-cheat rules constrain observations to x < 600, but a behaviour characterisation used only for training selection is not an observation.

### 3.3 Lehman et al. 2018, "ES is more than just a traditional finite-difference approximator"
STATED (arXiv 1712.06568 text):
- ES optimises the expected reward of a Gaussian *population*, so σ changes what is optimised. High σ moves the mean to the centre of a donut-shaped hole (σ 0.16), low σ approaches finite-difference behaviour (σ 0.002).
- Landscape results: see takeaway 3.
- Robustness study: ES is trained with σ = 0.02 (copied from Salimans for performance, not for robustness), Adam step 0.01, 10,000 rollouts per iteration; GA needed σ = 0.00224. High-σ perturbations of ES champions retain performance better than those of the GA or TRPO (Mann-Whitney p < 0.01).
- INFERENCE: this supports using a larger σ during the plateau, with a cost: the robustness argument also says a large-σ solution is a flatter, more perturbation-tolerant solution, which may help the "all held-out seeds reach 1M" goal.

### 3.4 Uber AI Labs blogs
- STATED (via summary, https://www.uber.com/blog/deep-neuroevolution/): ES and GA are stymied by deceptive problems, novelty search addresses it; no hyperparameter values in the blog; random search "substantially outperforms DQN, A3C, and ES" on some games, "never outperforms the GA".
- The NS-ES blog post (eng.uber.com summary) I saw only as a search snippet; it mentions NS-ES and NSR-ES but not NSRA-ES. UNVERIFIED for any detail beyond that.

### 3.5 Cheapest to bolt onto an existing OpenAI-ES loop
Ranking is INFERENCE; the facts in each line are STATED above.
1. **Stagnation-triggered σ escalation** (CMA flat-fitness rule plus NSRA-style trigger). No new data structures; one scalar in the per-generation config.
2. **Population or episode increase on stagnation** (IPOP idea, Hansen's `exp(0.2 + cs/damps)`).
3. **Re-initialisation portfolio** (take the best of several short random starts; Such et al. random-search observation).
4. **NSR-ES with a trivial behaviour characterisation**: needs an archive, a kNN novelty term, a rank-normalised mix and a meta-population. Most code, and no guarantee that a good characterisation exists here.

---

## 4. Q3: fitness shaping and ties

### 4.1 What each implementation does (STATED unless marked)
| Implementation | Shaping | Ties |
|---|---|---|
| es-starter `compute_centered_ranks` | rank in [0, n), divided by (n−1), minus 0.5 | `argsort`: distinct ranks, order set by the sort |
| es-starter other modes | `'sign'`: `np.sign(rews).sum()` per rollout; `'centered_sign_rank'`: ranks of those | `'sign'` is a transform of per-*timestep* rewards, not of the return. Needs a per-step reward stream |
| estool `OpenES` | same `compute_centered_ranks`, then `reward += −weight_decay·mean(θ²)` (penalty on the shaped reward), then z-scoring | `argsort` |
| EvoJAX PGPE `process_scores` | `ranks[x.argsort()] = arange; ranks/ranks.max() − 0.5` | `argsort` |
| EvoJAX OpenES | wraps old evosax `FitnessShaper(centered_rank=True, w_decay=...)` | old implementation not read (UNVERIFIED) |
| evosax (d4b4127) `centered_rank_fitness_shaping_fn` | `rankdata(...) − 1`, divided by (n−1), minus 0.5; NaN fitness ranked as worst | midranks (jax `rankdata` default `'average'`, STATED from the JAX docs: ties "both get rank 3.5" in the example) |
| evosax `weights_fitness_shaping_fn` | CMA-style weights by rank; all-NaN returns zero weights | uses the same `rankdata` |
| ARS | no ranks; `rewards /= np.std(rewards)`; optional top-b by `max(r+, r−)` | selection by percentile threshold (`>=`) keeps all tied directions at the cut |
| NES (Wierstra et al.) | utility on sorted fitness | the paper discusses utility choice ("not been crucial … as long as it is monotonous and based on ranks") but I found no tie discussion (UNVERIFIED beyond a grep of the text) |
| CMA-ES (Hansen) | rank-based weights | equal values handled by the flat-fitness rule (takeaway 2) |

Notes:
- The evosax commit history for `fitness_shaping.py` has "fix: rank failed evaluations consistently" dated 2026-08-17 (GitHub API). Earlier versions of evosax may have used a different rank implementation (UNVERIFIED; the tags I tried did not resolve).
- es-starter's gradient uses `proc_returns[:,0] − proc_returns[:,1]` (the antithetic difference of shaped returns), then `/ returns_n2.size`. `compute_centered_ranks` ranks all 2N rollouts jointly (rank spacing 1/(2N−1)). A pair contributes zero only if both members receive the same shaped value. With distinct ranks for tied returns, the two members of a tied pair land at arbitrary positions inside the block of m tied rollouts, so their rank difference can be anywhere up to ±(m−1)/(2N−1). When most of the population ties (HANDOFF: 56-61 of 64 pairs tied, so m is about 120 of 128), that is close to ±1, not a small nudge. This is the "index-tie random walk" in HANDOFF 4.2, restated against the code (INFERENCE for the magnitude: it follows from the code, I did not simulate it). It also means modes (a) and (c) in section 4.2 differ by a large amount, not marginally.
- No source discusses discrete or quantised *returns* with ties. The nearest items are discretised *actions* (Salimans: 10 bins for hopping/swimming, which "forced the actions to be non-smooth with respect to input observations and parameter perturbations"), and Hansen's flat-fitness remark. The search I ran for tie-handling literature found a library doc saying ties are broken by index (not a paper) and a sign-of-pair-difference scheme in a recent paper (snippet only, UNVERIFIED).

### 4.2 Inference for this project
- INFERENCE: three tie policies are testable with one config flag each: (a) distinct ranks (current), (b) midranks (HANDOFF's variant), (c) pair-sign weights `sign(F+ − F−)` with ties exactly zero (my construction; ARS-like in using only the pair difference; not a published ES setting I verified).
- INFERENCE: at the top of training the situation reverses. When most of 128 candidates score the cap on all 4 episodes, (a) assigns pseudo-random ranks among truly equal candidates and so injects noise in the update, while (b) and (c) give zero. A tie policy that switches on the tied fraction is therefore plausible, and a plain A/B of (a) vs (b) over a whole run mixes two opposite effects.
- Hardmaru (via summary): shaping is "very useful for RL tasks if the objective function is non-deterministic" and "can sometimes slow down the time it takes to find a good solution". Ties not discussed.

---

## 5. Q4: σ and lr schedules, ClipUp, size-dependence

### 5.1 Defaults and schedules (STATED from code or paper tables)
| Source | σ | lr / optimiser | Population | Schedule |
|---|---|---|---|---|
| es-starter `humanoid.json` | 0.02 fixed | Adam 0.01 | ~10,000 episodes per batch | none (adaptive timestep limit only) |
| Conti et al. Atari / humanoid | 0.02 | Adam 0.01 | 5,000 / 10,000 | none |
| estool `OpenES` | init 0.1, decay 0.999, limit 0.01 | lr 0.01, decay 0.9999, limit 0.001, Adam | 256 | multiplicative |
| estool `train.py` defaults | `--sigma_init 0.10`, `--sigma_decay 0.999` | same | num_worker x num_worker_trial | per-optimiser limits (0.02 in train.py) |
| EvoJAX `OpenES` | init 0.01, decay 0.999, limit 0.001 | lrate 0.01, decay 0.999, limit 0.001, Adam β1 0.99, β2 0.999 | user | multiplicative |
| evosax paper Table 4 (Brax Ant, Fetch, HalfCheetah, Humanoid) | init 0.05, 0.05, 0.075, 0.1 (column order assumed); decay 0.999; final 0.01 | lr init 0.01, 0.02, 0.01, 0.02; decay 0.999; final 0.001; Adam; centred rank | 256 | multiplicative |
| evosax `Open_ES` class defaults | std schedule `constant(1.0)` | `optax.sgd(1e-3)` | user | placeholders, not tuned (INFERENCE) |
| EvoJAX/evosax PGPE | init_stdev 0.1 (EvoJAX), `std_init` 1.0 (evosax) | center lr 0.15, `std_lr` 0.1, max change 20% (both); ClipUp `learning_rate 0.01, max_velocity 0.02` default in evosax | user | σ learned per coordinate |
| ARS Table 9 | ν 0.0075-0.03 per task | α 0.01-0.03 SGD on std-scaled step | N 1-230 directions | none |
| Such et al. GA | 0.002-0.005 | n/a | 1,000-20,000 | humanoid: anneal to 0.001 after 1,000 gens |

Notes: the evosax paper's hyperparameter table text extraction is partly scrambled (hidden units, layers and column alignment in Table 2), so the architecture reading from that table is uncertain. The σ/lr rows above read cleanly in column order but the mapping to task names is by order of appearance (assumed, not labelled).
- STATED (Salimans 2017): fixed σ, no adaptive σ. STATED (Such et al.): annealed σ gave "a small performance boost".

### 5.2 PGPE, adaptive σ and ClipUp (Toklu et al. 2020)
STATED (arXiv 2008.02387 text; evosax `optimizer.py`; EvoJAX `pgpe.py`):
- PGPE learns a per-coordinate σ vector with learning rate Ω (0.1) and clips each σ update to 20% of its current value. Ranks in [−0.5, 0.5] and Adam were adopted from Ha's earlier work.
- ClipUp (Algorithm 2): normalise the gradient to length α, add to momentum-0.9 velocity, clip the velocity's norm to v_max. Heuristic: v_max is the one tuned value, α = v_max/2, m = 0.9, radius r = ‖σ0‖ = q·v_max with q ≈ 10-20 (default 15; 18 for Humanoid-v2).
- Reported advantage (ClipUp, section 5.1): insensitive to reward-scale (LunarLander Table 2: ClipUp 269.95 with ranks, 270.15 raw, 262.98 x1000, while Adam at α 0.15 fell to 77.52% raw and 54.68% x1000 of its ranked score).
- v_max used (Tables 1, 3-5): LunarLanderContinuous 0.3 (linear, 16 variables), Walker2d-v2 1.5e-2 (linear, 108 variables, λ 100 up to 800), Humanoid-v2 1.5e-2 (linear, 6,409 variables, λ 200 up to 3,200), HumanoidBullet 0.15 (one hidden layer of 64, 3,985 variables, λ 10,000 up to 80,000).
- STATED (ClipUp, HumanoidBullet section): as in the MuJoCo runs the default alive bonus was removed, and the trajectory length bound was cut from 1000 to 200 timesteps "since the hardest part of the task is starting a forward gait". INFERENCE: a shortened-episode phase early in training is a published precedent for the plateau problem, but the project's rule that evaluation uses the unmodified game still holds for the held-out metric; whether a training-only cap is allowed is the project's call.
- PGPE+ClipUp also uses an adaptive population (add λ more solutions until a timestep budget T is reached or λ_max) and the running-statistics observation normalisation (section 2.3).
- v_max values in these tables differ by 20x across problems, so the heuristic fixes *ratios* (α, r relative to v_max), not an absolute size rule across networks.
- INFERENCE (arithmetic, not measured): for the 44-32-32-3 net d = 44·32+32 + 32·32+32 + 32·3+3 = 2,595 parameters (assuming biases on every layer). σ = 0.1 gives r = 0.1·√2595 ≈ 5.1 and a ClipUp v_max of r/15 ≈ 0.34 per generation. Adam at lr 0.03 can move each coordinate by up to ~0.03 per step, a vector step up to ~0.03·√2595 ≈ 1.5. HANDOFF measured mean |m|/√v of 0.22-0.31, so the real step is smaller, but still several times the ClipUp default.

### 5.3 Does anything scale σ with network size?
- STATED, yes in formula form: ClipUp's r = ‖σ0‖ (norm over all coordinates); NES default learning rates and λ vs d (Wierstra et al., Table 1: λ = 4+⌊3 log d⌋; SNES η_σ = (3+log d)/(5√d)); Salimans duplicate-feature argument (karpathy note takeaway 3).
- STATED, no for libraries: evosax/EvoJAX/estool apply one σ schedule per task regardless of hidden size (Brax tasks use the same MLP with σ varying 0.05-0.1 by *task*, not by size).
- INFERENCE: the width/depth sweep should treat σ0 as a function of the parametrisation (fan-in or total d), and test one of those rules against a fixed-σ control.

---

## 6. Q5: restarts and portfolios

### 6.1 IPOP and BIPOP
STATED (Auger and Hansen 2005 text; Hansen tutorial B.3; Loshchilov et al. 2012):
- IPOP: restart when any default stopping criterion fires, double λ each time, default everything else; each restart starts from a point sampled uniformly in the search box with σ0 = (B−A)/2. Doubling was chosen "to conduct a larger number of restarts per run".
- Stopping criteria (the paper's list and the tutorial's, which agree on the numerical ones): equalfunvalhist/EqualFunValues (range of best values over the last 10+⌈30n/λ⌉ generations is zero), TolFun 1e-12 on that range plus the current generation, TolX 1e-12·σ0 (σ and evolution path), NoEffectAxis (adding 0.1σ along a principal axis leaves the mean unchanged), NoEffectCoord (adding 0.2σ in a coordinate leaves it unchanged), ConditionCov > 1e14. The tutorial adds Stagnation (a history of best and median fitness over the last 20%, at least 120+30n/λ and at most 20,000 iterations; stop if the median of the newest 30% is not better than the median of the oldest 30% in both histories) and TolXUp (σ·max(diag D) grows by more than 1e4).
- The tutorial advises that flat fitness "should be rather a termination criterion and consequently lead to a reconsideration of the objective function formulation".
- BIPOP (Loshchilov et al. 2012, restating Hansen 2009; the primary BIPOP paper was not read, UNVERIFIED): large regime = IPOP; small regime draws λ_small = λ_default·(½·λ_large/λ_default)^(U[0,1]²) and σ0_small = σ0_default·10^(−2U[0,1]); the regime with fewer function evaluations so far is chosen next.
- evosax `restart.py` (STATED): `spread_cond` (max−min < 1e-12), `fitness_std_cond` (std ≤ atol + 0.001·|mean|), `generation_cond`, IPOP/BIPOP controllers with `min_num_gens` 50 and `population_size_multiplier` 2. Its BIPOP small regime applies only to CMA-ES (`_default_small_population_params` returns other params unchanged).
- Alternative result (STATED, Auger and Hansen abstract text): on CEC 2005, IPOP was significantly better than a local restart strategy in 29 of 60 cases, worse in 4. The local-restart strategy keeps visiting different optima (small populations), IPOP converges more reliably (large).

### 6.2 ES-for-RL work using restarts or racing
- STATED (NES, Wierstra et al., section 3.4): restarts are a standard mitigation for local optima. I did not extract the specific restart rule.
- Heidrich-Meisner and Igel 2009 (ICML): Hoeffding and empirical-Bernstein races choose the number of episodes per candidate in CMA-ES, "just accurate enough to rank candidate policies". Only the abstract was verified; the race's confidence level and episode limits are UNVERIFIED.
- STATED (ARS paper, Tables 6-7): over 100 random seeds on Humanoid, ARS V2 reached reward 6000 in a 10th/25th/50th/75th percentile of 10/13/21/45 minutes on a 48-CPU instance (16/23/41/96 and 21/28/57/144 on 18-CPU instances). The seed spread is a 4x range between the 10th and 75th percentile. INFERENCE: the project's three-seed plateau escape times are a small sample of a heavy-tailed distribution; a portfolio of short starts with early-kill is a sound fit to that shape.
- STATED (Conti): the meta-population (M = 3-5) is itself a portfolio that reallocates updates to the most novel member. In the deceptive humanoid some lineages "get stuck in the deceptive trap, incentivizing other policies" to move elsewhere.
- I found no source applying IPOP/BIPOP triggers to OpenAI-ES (UNVERIFIED negative: a web search only, and my reading is limited).

### 6.3 Concrete trigger candidates for this project (INFERENCE, adapting the rules above)
- Plateau trigger: tie fraction (pairs whose two returns are equal) at or above 0.7 for 10 consecutive generations (echoing the 70th-percentile test), or no new best validation score in t_w = 10 (NSRA lesson) to 25 generations.
- Late-run trigger: a validation score at cap with no held-out crash for N consecutive checks switches the run to the confirmation protocol rather than restarting.
- Restart action (IPOP analogue): population x2 (128 to 256) with the same σ; BIPOP analogue: draw σ0 log-uniformly over [0.1, 0.5] at the base population. Budget accounting by evaluations (population x episodes x generations), as BIPOP does.

---

## 7. Q6: small-network ES successes on simple games

| Source | Network | Algorithm and settings | Pitfalls reported |
|---|---|---|---|
| estool SlimeVolley config (`config.py`, README, STATED) | 12 inputs, [20, 20] tanh, 3 outputs | CMA-ES is the default optimiser; example `train.py slimevolley -n 8 -e 8 -t 4 --sigma_init 0.5` (population 32, 8 episodes per candidate) | none in the repo; slimevolleygym TRAINING.md (via summary): "somewhat overfit to a particular opponent's playing style"; GA ran 500,000 tournaments "only a few hours"; CMA-ES "a few thousand generations" |
| estool cartpole_swingup (README, STATED) | 5 inputs, [10, 0] hidden | CMA, `--sigma_init 1.0`, `-e 1` | README: after 400 generations the average over 32 trials is "over 900" |
| Hardmaru, Evolving Stable Strategies (via summary) | 2 hidden layers tanh, widths not given | PEPG (population 256 for Ant), CMA-ES, OpenAI-ES; fitness is the average of 16 rollouts | lucky easy maps; averaging narrows the train/test gap but costs 16x data; objective-driven search exploits design flaws |
| Hardmaru, Visual Guide (via summary) | n/a | Rule of thumb: CMA-ES below ~1,000 parameters (patient up to ~10K), PEPG when there are "several thousand" | OpenAI-ES on 100-D Rastrigin needed an annealing schedule; shaping "can sometimes slow down" |
| Salimans 2017 (paper text) | 2 x 64 tanh MLP (MuJoCo) | fixed σ, ranks, weight decay, Adam, population in the thousands | perturbed Atari policies always taking one action; discretised actions helped on hopping and swimming |
| ClipUp HumanoidBullet (paper text) | 1 hidden layer of 64 tanh | PGPE+ClipUp, λ 10,000 | none specific |
| Cuccu et al. 2019 "Playing Atari with Six Neurons" (text partly read) | 6-18 neurons, recurrent, fed by a learned compressor | XNES with fitness shaping | the paper stresses that the O(p³) covariance update limits XNES to a few hundred dimensions (per its section 3.2 text) |
| Wong et al. 2024 (text partly read) | linear policies on Atari RAM | CSA-ES, CMA-ES, sep-CMA-ES, ARS | state normalisation was used for ES; linear policies matched deep RL on several MuJoCo tasks |
| dusanerdeljan/neuroevolution-dino (via summary) | cactus net 5-8-2, bird net 6-10-3 | GA, population 100, elitism 0.2, mutation rate 0.1, std 0.5, crossover | none reported |
| IAMDinosaur (via summary) | 3 inputs, one output with three states | GA, population 12, fitness = cacti jumped | the dino "drifts" to the right; README workaround |
| Flappy Bird repos | 4-6-1 and others | population 50-100, mutation 0.1 | snippets only: UNVERIFIED, none cited as evidence |

Observations:
- INFERENCE: nobody in this list reports a 1M-scale episode cap or an all-seeds criterion; the closest analogues to "late rare crashes" are the averaging protocols (Hardmaru's 16 rollouts, Such et al.'s 30-episode re-evaluation, Conti et al.'s "mean over ~30 evaluations" for reported champions).
- STATED (Salimans, section 2.2): "it is crucial that Gaussian perturbation vectors occasionally lead to new individuals θ + σε with better return". That is the plateau problem stated in one sentence.

---

## 8. Mapping table: technique to project problem to code change to cheap experiment

All experiments: 900 generations at most, hidden 32,32, seeds 1-3, baseline = the current shipped config (HANDOFF: escape at ~776 on seed 1) and, where relevant, the midrank plus decoupled decay 0.01 config. "Escape" means the first generation at which the validation median exceeds 100 points (the 54-point plateau is below that); report per seed. Add one tie-fraction column to every run (pairs with equal return, per generation).

| # | Technique | Problem | Code change (ES loop with ranks, Adam, per-generation config) | Cheap experiment |
|---|---|---|---|---|
| 1 | CMA flat-fitness σ bump (Hansen listing) + Lehman high-σ gap crossing | Plateau (1) | Per generation compute `tieFrac`. If `tieFrac ≥ 0.7` for 10 consecutive generations, set `σ ← σ·exp(0.2)` per further generation (cap at 4σ0); when `tieFrac < 0.5` decay `σ ← σ·0.97` back to σ0. σ lives in the per-generation config, and the noise regenerated by workers uses the current σ. | 3 seeds, 900 gens, midrank + decay 0.01. Compare escape generation with the 776 baseline and the 400-gen non-escape of midrank + decay. Record σ trajectory and ‖θ‖. |
| 2 | NSRA-style stagnation trigger without novelty (Conti) | Plateau (1) | Track best validation score. After `t_w` generations without improvement, `σ ← σ·(1+δ)`; on improvement, `σ ← σ/(1+δ)`. Use `t_w` 10 and 25, δ 0.1. | 3 seeds x 2 settings, 900 gens each (6 runs), midrank + decay 0.01. Escape generation per run. |
| 3 | Tie policy as a flag: distinct ranks vs midrank vs pair-sign | Plateau (1), late noise (3) | A `tie` config key selects the shaping function: argsort ranks, midranks, or `w_i = sign(F+_i − F−_i)` with equal pairs exactly zero (no rank at all). Everything downstream (Adam) is unchanged. | 3 seeds, 900 gens, each tie mode with decoupled decay 0.01. HANDOFF only measured two modes on seed 1. Escape generation and tieFrac trajectory. |
| 4 | IPOP population doubling on stagnation | Plateau (1) | Per-generation config key `popsize` with a controller: if the plateau trigger fires (tieFrac ≥ 0.7 for 25 generations), double 128 to 256 (keep σ, Adam state reset or kept as a second variant). | 3 seeds, 900 *generation-equivalents* (count evaluations, not generations). Escape in evaluations vs baseline. |
| 5 | BIPOP-style σ0 draw on restart | Plateau (1), seed variance | After 150 generations with validation < 80, re-initialise θ and Adam, draw σ0 log-uniform in [0.1, 0.5], and keep the evaluation budget accounted. | 3 seeds, total budget 900 generations including the dropped start. Escape in total evaluations. |
| 6 | Ranked success of short random starts (Such et al. near-origin observation) | Plateau (1) | Sample 64-256 random θ (scale as init), score each over 4 seeds, start ES from the best 1-3. No loop change, a pre-step in `train`. | 3 seeds: report the best random-start score distribution first (no training). Then 900 gens from the best start vs baseline. |
| 7 | Near-zero output layer init (es-starter `normc(0.01)` output, `normc(1.0)` hidden) | Plateau (1); init policy is 52% none / 47% duck | A different init for θ0 only: scale each unit's incoming weight vector to the target norm; output layer norm 0.01, biases 0. No loop change, `policy.js` untouched. | 3 seeds, 900 gens. Report the init action histogram and the escape generation. |
| 8 | Per-layer σ from fan-in / ClipUp-style fixed radius | σ/lr transfer (2) | σ becomes a per-parameter-group vector: `σ_l = σ0·√(f_ref/fan_in_l)` (rule A), or `σ_i = r/√d` with r = ‖σ0‖ (rule B). Equal to the baseline when the network equals the reference one. | On 32,32 only: rule A with `f_ref` = 32 vs the fixed-σ control (3 seeds, 900 gens). A width test (16,16 vs 64,64) is outside the stated bound and is a follow-up. |
| 9 | ClipUp optimiser replacing Adam | σ/lr transfer (2) | Replace the Adam step by: normalise g to length α, `v ← 0.9 v + α g/‖g‖`, clip ‖v‖ to v_max, `θ ← θ + v`. With σ=0.1 and d=2,595 the heuristic gives v_max≈0.34, α≈0.17 (INFERENCE arithmetic). | 3 seeds, 900 gens, v_max 0.34 and 1.0. Compare escape generation and ‖θ‖ with Adam lr 0.03. Note the optimiser has one step-size knob, so results transfer more easily to other widths if the claim holds. |
| 10 | Static observation normalisation folded outside `policy.js` | Plateau (1) (minor); width sweep | Compute per-feature mean/std from a fixed batch of random-policy rollouts (Such et al.), floor std at 0.1 (es-starter variance floor 1e-2), clip ±5. Train in the normalised parametrisation, fold into W1/b1 when handing weights to the unchanged policy. | 3 seeds, 900 gens. Expected small effect (INFERENCE); run last. Check first that every feature has a non-degenerate batch std. |
| 11 | Elite re-evaluation (Such et al.) before naming "best" | Late rare crashes (3) | In the validation step, re-evaluate the top K=5-10 candidates (by validation) on extra seeds drawn from a disjoint range, then pick the best by the combined mean/min. Only the selection changes. | Resume the shipped 32,32 run (or a checkpoint of the most recent late-phase run) for 200-300 generations, 3 seeds of resumption noise. Count held-out seeds below cap at the end. |
| 12 | Averaged-rollout fitness with more episodes in the late phase (Hardmaru's 16, Such's 30) | Late rare crashes (3) | Per-generation config `episodes` rising from 4 to 8-16 when the mean score is at cap; ranks over means. | Resume as in row 11; count crashes among held-out seeds, and report evaluations per generation. |

---

## 9. Open questions

1. Does the argsort tie policy really help only at the plateau? The HANDOFF counterfactual used one seed (seed 1). Row 3 gives three. A mechanism check would log how many antithetic pairs get a non-zero weight due solely to tie-breaking.
2. What is the tie fraction trajectory in the baseline run, and does it ever stay below 0.7? If it does not reach the trigger, rows 1 and 4 are inert.
3. Is there a behaviour characterisation available to NS-ES that does not use observations the anti-cheat rules forbid (nothing about future obstacles)? Candidates (jump timing against obstacle phase) were not tested.
4. Does a larger σ also harm late convergence? Lehman et al. link high σ to robustness, Such et al. to a "small performance boost" from annealing downward. Both can be true at different phases. A σ schedule (cosine or the estool multiplicative decay) over the late phase is untested here.
5. I did not read Heidrich-Meisner and Igel's race rule (abstract only), Hansen's BIPOP paper (second-hand), the Salimans supplement hyperparameter tables, the older evosax rank implementation, or the NES paper's restart text. Each could contain a more specific rule than the secondary descriptions above.
6. ClipUp reports reward-scale insensitivity for rank or raw rewards. With ranks the gradient scale is already bounded, so it is not clear that the advantage over Adam carries over. Row 9 checks only the plateau, not the late phase.
7. Whether static normalisation can be folded without touching `model.js` export, the CSS page and the policy's bit-identical decision test was not checked.
8. A constant-radius σ (rule B) and fan-in scaling (rule A) disagree for the first layer (fan-in 44) vs the others (32). Which of them is right for a tanh MLP with 44 hand-crafted inputs of mixed scale is not decided by any source.

---

## 10. References

Access date for all URLs: 2026-10-10. "V" means read as source text or code in this session (PDF text extracted locally, raw repository files, or the fetched page in full). "S" means fetched through a summarising tool, so facts are as reported by that tool. "U" means unverified (abstract, search snippet or second-hand only).

| # | Reference | URL | Status |
|---|---|---|---|
| R1 | Salimans, Ho, Chen, Sidor, Sutskever, Evolution Strategies as a Scalable Alternative to Reinforcement Learning, arXiv:1703.03864 (v2) | https://arxiv.org/abs/1703.03864 | V (main text; supplement hyperparameters not read) |
| R2 | openai/evolution-strategies-starter (`es_distributed/es.py`, `policies.py`, `optimizers.py`, `tf_util.py`, `configurations/humanoid.json`, README), master | https://github.com/openai/evolution-strategies-starter | V |
| R3 | Mania, Guy, Recht, Simple random search provides a competitive approach to reinforcement learning (ARS), arXiv:1803.07055 | https://arxiv.org/abs/1803.07055 | V |
| R4 | modestyachts/ARS (`code/ars.py`, `filter.py`, `policies.py`, `optimizers.py`), master | https://github.com/modestyachts/ARS | V |
| R5 | Petroski Such et al., Deep Neuroevolution, arXiv:1712.06567 | https://arxiv.org/abs/1712.06567 | V |
| R6 | Conti et al., Improving Exploration in Evolution Strategies ... Novelty-Seeking Agents, arXiv:1712.06560 | https://arxiv.org/abs/1712.06560 | V |
| R7 | Lehman, Chen, Clune, Stanley, ES Is More Than Just a Traditional Finite-Difference Approximator, arXiv:1712.06568 | https://arxiv.org/abs/1712.06568 | V |
| R8 | Toklu, Liskowski, Srivastava, ClipUp, arXiv:2008.02387 | https://arxiv.org/abs/2008.02387 | V |
| R9 | RobertTLange/evosax at d4b4127708cb2cd0cc906b910ddcd091013153a9 (`core/fitness_shaping.py`, `core/restart.py`, `core/optimizer.py`, `algorithms/distribution_based/open_es.py`, `pgpe.py`) | https://github.com/RobertTLange/evosax | V |
| R10 | google/evojax at acf5252475e4fbf47e0e3f7994c03de11dd3c4ca (`algo/open_es.py`, `algo/pgpe.py`, `obs_norm.py`, `sim_mgr.py`) | https://github.com/google/evojax | V |
| R11 | Lange, evosax: JAX-based Evolution Strategies, arXiv:2212.04180 (hyperparameter table text partly scrambled) | https://arxiv.org/abs/2212.04180 | V (partial) |
| R12 | Tang, Tian, Ha, EvoJAX, arXiv:2202.05008 | https://arxiv.org/abs/2202.05008 | V (little relevant content) |
| R13 | Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772 (v2) | https://arxiv.org/abs/1604.00772 | V |
| R14 | Auger, Hansen, A Restart CMA Evolution Strategy With Increasing Population Size, CEC 2005 | https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC05/auger05ARCMA.pdf | V |
| R15 | Loshchilov, Schoenauer, Sebag, Alternative Restart Strategies for CMA-ES, arXiv:1207.0206 (BIPOP formula, secondary) | https://arxiv.org/abs/1207.0206 | V |
| R16 | Wierstra et al., Natural Evolution Strategies, arXiv:1106.4487 (v1 text read; JMLR 2014) | https://arxiv.org/abs/1106.4487 | V |
| R17 | hardmaru/estool (`es.py`, `config.py`, `train.py`, README), master | https://github.com/hardmaru/estool | V |
| R18 | JAX documentation, `jax.scipy.stats.rankdata` | https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.rankdata.html | S |
| R19 | Wong, de Nobel, Back, Plaat, Kononova, Solving Deep RL Tasks with Evolution Strategies and Linear Policy Networks, arXiv:2402.06912 | https://arxiv.org/abs/2402.06912 | V (partial) |
| R20 | Cuccu, Togelius, Cudre-Mauroux, Playing Atari with Six Neurons, arXiv:1806.01363 | https://arxiv.org/abs/1806.01363 | V (partial) |
| R21 | Hardmaru, A Visual Guide to Evolution Strategies | https://blog.otoro.net/2017/10/29/visual-evolution-strategies/ | S |
| R22 | Hardmaru, Evolving Stable Strategies | https://blog.otoro.net/2017/11/12/evolving-stable-strategies/ | S |
| R23 | hardmaru/slimevolleygym README and TRAINING.md | https://github.com/hardmaru/slimevolleygym | S |
| R24 | Uber Engineering, Welcoming the Era of Deep Neuroevolution | https://www.uber.com/blog/deep-neuroevolution/ | S |
| R25 | dusanerdeljan/neuroevolution-dino README | https://github.com/dusanerdeljan/neuroevolution-dino | S |
| R26 | NikoZHAI/IAMDinosaur README | https://github.com/NikoZHAI/IAMDinosaur | S |
| R27 | Uber, Improving exploration in ES ... (blog summary of R6) | https://www.uber.com/blog/research/improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents | U (search snippet only) |
| R28 | Heidrich-Meisner, Igel, Hoeffding and Bernstein Races for Selecting Policies in Evolutionary Direct Policy Search, ICML 2009 | https://dl.acm.org/doi/pdf/10.1145/1553374.1553426 | U (abstract via mlanthology only) |
| R29 | Hansen, Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed, GECCO 2009 | https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00382093 | U (search listing only; BIPOP formula taken from R15) |
| R30 | Flappy Bird neuroevolution repositories (for example frenzzy/flappylearning) | https://github.com/frenzzy/flappylearning | U (search snippets only) |
| R31 | Sehnke et al., Parameter-exploring policy gradients, Neural Networks 2010 (cited via R8 and R9 only) | https://link.springer.com/chapter/10.1007/978-3-540-87536-9_40 | U (not read) |

Counts: V (read as source text or code) = 19: R1-R17, R19, R20 (R11, R19 and R20 only partly read). S (fetched via summary) = 7: R18, R21-R26. U (unverified) = 5: R27-R31. Total 31.

Local project sources used: `research/karpathy-training-techniques.md` (sections 1 and 2.9) and `HANDOFF.md` sections 4.2-4.5 (not counted above).
