# Robustness of students distilled from a privileged teacher: what the literature says, and what to do after the clone scan

Scope: the shipped 44-32-32-3 tanh MLP, behaviour-cloned from `test/solver.js` (which reads the hidden gap and the fairness lookahead) and playing only from the 44 legal features. The scan in `research/prereg-bc-robustness.md` (2,000 paired games per clone arm) asks whether it has rare failure modes the solver lacks. This note reads the literature for that question and ends with a mapping onto the prereg §5 readings.

Labels. **STATED**: read in the source (abstract or page) this session. **INFERENCE**: my reasoning from stated facts. **[D]**: computed here, parameters given. **UNVERIFIED**: I could not fetch or read the source text, or the claim is from memory. Fetch limits: the arXiv PDFs came back as compressed streams, so theorem statements for Ross et al. and the DART internals are not read first-hand; I list what the abstracts and search snippets support, and mark the rest. I did not run any code against the repo.

---

## 1. Summary

1. **The classic worst-case bound says plain BC compounds quadratically, and DAgger removes that.** BC's loss over horizon T grows like T^2 times the per-step error eps (Ross and Bagnell 2010 and Ross, Gordon and Bagnell 2011; the T^2 form is **UNVERIFIED** first-hand, from memory and from the DAgger paper's framing). DAgger's finite-sample theorem is J(pi_hat) <= J(pi*) + u T eps_N + O(1), where u is how much one mistake can cost (STATED via a search snippet of Theorem 3.2, [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)). The bound needs a non-trivial u: if one mistake can be fatal and unrecoverable, u ~ T and DAgger gains nothing over BC in the worst case (INFERENCE from the form of the bound).
2. **For this game the worst case is vacuous in both directions, so the bound cannot decide anything.** T is millions of frames, and the clone passed 150/150 and 352/352 full games. The bound is an upper envelope; the scan measures the realised rate. Spencer et al. report that plain BC does well on common benchmarks, which they call too easy to show real error compounding (STATED, [arXiv 2102.02872](https://arxiv.org/abs/2102.02872)). Our game is closer to that "easy" regime: a deterministic, fully-observed-in-features task, with a teacher that is a function of state.
3. **"Learning by Cheating" is our exact setup, and it reports a student that beat the prior state of the art, 100% on the original CARLA tasks, with infractions cut by about an order of magnitude** (STATED, [arXiv 1912.12294](https://arxiv.org/abs/1912.12294)). Whether its student training used on-policy relabelling (DAgger-like) is not in the abstract (**UNVERIFIED**, from memory I believe it did).
4. **The documented failure mode of privileged distillation is the "imitation gap": teacher actions that depend on information the student lacks get averaged.** ADVISOR (Weihs et al., NeurIPS 2021): hidden teacher information is "marginalized" during imitation, giving weak students (STATED, [arXiv 2007.12173](https://arxiv.org/abs/2007.12173)). Kim et al. 2025 add that the asymmetry causes distribution shift, and propose querying the teacher only when needed, resetting from recovery states (STATED, [arXiv 2505.09546](https://arxiv.org/abs/2505.09546)). For us, probe B (`feature-sufficiency.md`) found the 44 features sufficient on average, but sufficiency "on average over visited states" does not exclude rare states where the hidden gap is ambiguous. That is the fatal-frame mechanism in §3.
5. **There is no standard name or test, in what I found, for "first teacher-student disagreement"; the prereg's oracle-disagreement trace is our own construct built from DAgger's labelling step.** I add one stronger test (§4: back-up-and-rescue) that separates drift from a fatal frame causally rather than by timing. (Literature search for a named diagnostic: nothing found, **UNVERIFIED negative**.)
6. **At our scale plain Monte Carlo is cheaper than importance splitting.** At 3.4 games/s (prereg §3), a rate of 1e-4 needs about 30,000 games for a rule-of-three bound, about 2.5 h [D: 3/1e-4/3.4/3600]. Splitting only pays below about 1e-5 (24 h of plain play). The scan's 2,000 games bound 0.18% two-sided, 0.15% one-sided [D].
7. **Pick next steps by mechanism, not by count.** Drift calls for relabelling (DAgger) or noise (DART); a fatal-frame error calls for more or better data at that state and a closer look at feature ambiguity, not for a new training method.

---

## 2. The literature, one item at a time

### 2.1 DAgger and its guarantee
- Setting: the learner's own policy shifts the state distribution, which breaks the i.i.d. assumption of supervised learning. DAgger iterates: roll out the current policy, have the expert label the visited states, aggregate all data, retrain. The paper frames it as a no-regret online learner and guarantees good performance "under the distribution of observations it induces" (STATED, [abstract](https://arxiv.org/abs/1011.0686)).
- Bound (Theorem 3.2 as quoted in a search snippet): if N is O~(uT), some iterate has J <= J(pi*) + u T eps_N + O(1). eps_N is the best-in-class loss on the aggregated distribution. Theorem 3.3 gives a finite-sample version with an Azuma-Hoeffding term (both via search snippet; the exact constants are **UNVERIFIED**).
- Read it for us. The bound is linear in T only when u is O(1), meaning mistakes are recoverable. A wrong action at the fatal frame of a dino game is not recoverable (u large for that frame). That is why the bound cannot promise to fix a fatal-frame error; it can only promise that drift, which is recoverable, is covered.
- Costs: DAgger needs expert queries at every state the learner visits. Ours is the cheap case: the solver is deterministic, free, and can be run on any state the clone reaches. The prereg notes the one engineering risk, driving `createSolver` from a mid-game state.

### 2.2 When relabelling helps, and when plain BC suffices
- INFERENCE, from the bound's structure and the above: relabelling helps when (a) the student's errors are small but not zero, (b) they move it into a region the demonstrations did not cover, and (c) the teacher's label there differs from what the student's generalisation gives. It does not help when the errors are in-distribution, because the aggregated data adds the same states again.
- Evidence that BC alone can suffice: Spencer et al. found plain BC works on the common benchmarks, which they call too easy (STATED, [arXiv 2102.02872](https://arxiv.org/abs/2102.02872)); probe B here (352 games, 0 crashes, 2.1M frames); the shipped clone 150/150.
- Evidence that it can fail: DART reports BC needs recovery data in a physical grasping task, with DART averaging a 62% gain over BC (STATED, [arXiv 1703.09327](https://arxiv.org/abs/1703.09327)). That task is stochastic with a physical robot; its relevance to a deterministic game is **INFERENCE**.
- Our setting has an unusual advantage: the demonstrations are generated in the very simulator we evaluate in, so the state coverage can be audited rather than assumed. The solver's own trajectories cover the states a perfect player visits, and the clone's trajectories nearly coincide with them whenever it matches the solver.

### 2.3 DART noise injection
- Laskey et al. (CoRL 2017) inject noise into the supervisor's actions while collecting demonstrations, so the data includes recoveries; the noise level is tuned to approximate the error of the trained policy; it is off-policy, so the supervisor is queried less than in DAgger (STATED, [arXiv 1703.09327](https://arxiv.org/abs/1703.09327)). Reported: up to 3x less compute than DAgger on Humanoid; supervisor reward lost about 5% during data collection, against much more for DAgger's policies (STATED, same source).
- Read it for us: noise injection is a cheap broadening of state coverage that needs no student rollouts, and the "noise matched to the learner's error" rule is something we can measure: the clone's action error rate is its disagreement with the solver (training loss 0.0044 Python, 0.0056 Node; prereg §2).
- A caution (INFERENCE): the solver's noisy rollouts visit states where the solver itself is perturbed. If the solver can fail there (it reads exact hidden state but its safe region is finite), those demonstrations contain solver deaths. `bc-gen` counts solver crashes, 0 so far, so the check is free.

### 2.4 Learning by Cheating and privileged distillation in RL
- Chen et al. 2019: stage 1 trains an agent with ground-truth layout and traffic positions; stage 2 distils it into a vision-only student with no privileged input (STATED, [arXiv 1912.12294](https://arxiv.org/abs/1912.12294)). The student there has a genuine information deficit; ours has a much smaller one, since the 44 features already contain the lookahead the model needs (probe B).
- Robot locomotion uses the same recipe: RMA trains a base policy with privileged environment parameters, then an adaptation module estimates them from history, trained in simulation and deployed without fine-tuning (STATED, [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)). I could not confirm from the abstract of the Lee et al. 2020 quadruped paper ([arXiv 2010.11251](https://arxiv.org/abs/2010.11251)) that it uses a teacher-student scheme (**UNVERIFIED**; I recall it does).
- Reported student failure modes (STATED where cited):
  - the imitation gap from marginalised hidden information (ADVISOR, [2007.12173](https://arxiv.org/abs/2007.12173));
  - distribution shift from information asymmetry, and a student that cannot follow the teacher into states it cannot perceive (Kim et al., [2505.09546](https://arxiv.org/abs/2505.09546));
  - a teacher trained without regard for the student gives demonstrations the student cannot realise; a 2024 paper trains the teacher jointly with the student to fix this (a search snippet only, [arXiv 2412.09149](https://arxiv.org/abs/2412.09149), **UNVERIFIED** beyond the snippet).
- Read it for us: three distinct failure classes, each with a different diagnostic.
  1. *Hidden-information ambiguity* (imitation gap): the solver's action at a state depends on `o.gap` or the fairness lookahead that the 44 features do not determine. Diagnostic: two states with identical 44 features and different solver actions. This is checkable offline on the demo set, in seconds. **I did not find that this check is already recorded in `feature-sufficiency.md`** (I only grepped for "probe B", "covariate" and "dagger"), so it should be treated as open.
  2. *Covariate shift*: the student leaves the teacher's state distribution.
  3. *Capacity or optimisation error*: the same state, a wrong output, because the fit is imperfect. The CSS evaluator's top-2 margin (2.7e-4 smallest) shows some states sit near decision boundaries.

### 2.5 Safety-aware DAgger variants (brief)
- HG-DAgger learns a risk threshold from model uncertainty to decide when the novice can act (STATED, [arXiv 1810.02890](https://arxiv.org/abs/1810.02890)); DropoutDAgger uses a dropout-based Bayesian novice for a confidence signal (STATED, [arXiv 1709.06166](https://arxiv.org/abs/1709.06166)). Both are about safe data collection with a human or a risky expert in the loop. Our oracle is a free simulator query, so we do not need their gating. What carries over is the uncertainty signal as a failure predictor, e.g. low top-2 logit margin as a flag for frames to relabel.

### 2.6 Karpathy
- I did not fetch a primary Karpathy source for this topic. The repo's `research/karpathy-training-techniques.md` covers training practice. A search for his Autopilot "data engine" returned only secondary descriptions (a trigger system, shadow mode logging disagreements between the model and the driver) (STATED from secondary sources, not primary; **UNVERIFIED** as to his own wording). The shared idea is useful as a framing: log the frames where the student's decision differs from the oracle's, then add them to the training set. That is DAgger by another name.

---

## 3. Telling drift from in-distribution error

What the literature gives is the mechanism, not a standard test (**UNVERIFIED negative**, see summary item 5). Using DAgger's own framing, a failure is covariate shift if the student reaches states with low density under the teacher's distribution and errs there; it is an in-distribution error if the state is one the teacher's data already covers and the student errs anyway.

The three tests, in cost order, all on a failed seed:

1. **First disagreement (prereg §4).** Replay the clone, query the solver on the clone's state at every frame, log the frames where the actions differ. A first disagreement long before death and a growing run of disagreements suggests drift. Disagreements only at the last frames suggests a local error. A weakness (INFERENCE): the solver and the clone may differ harmlessly on many frames (both actions safe), so early disagreements are expected and are weak evidence on their own. Report the rate of disagreement along the trajectory, compared with the rate on the solver's own trajectory for the same seed.
2. **Back-up-and-rescue (new).** For a failed seed, rerun the game replaying the clone's recorded actions up to frame k, then hand control to the solver. Find the largest back-up n = (death frame - k) from which the solver still survives. The game is deterministic, so the replay is exact. Small n (a few frames) means a fatal frame: the clone was already doomed by a very recent decision. Large n (hundreds of frames) means drift: the state was recoverable long before and the clone failed to recover. It is a causal test and costs one game per probe (about 0.3 s each; a bisection is under 20 probes). This is DAgger's recoverability constant u measured directly (INFERENCE: u is the quantity the bound depends on).
3. **Feature-ambiguity check.** From the death state, search the demo set for states with near-identical 44 features and a different solver action (nearest-neighbour in feature space). Close neighbours with opposite labels are the imitation gap (§2.4 class 1); no neighbours at all means the state is out of the demo distribution (class 2).

Decision table (INFERENCE):

| Rescue back-up n | Neighbour in demos? | Reading |
|---|---|---|
| large | no | drift into unseen states: DAgger target |
| small | yes, same label | capacity or optimisation error: more steps, wider net, or noise |
| small | yes, conflicting label | imitation gap: features ambiguous at this state; relabel does not help, a feature or a different teacher is needed |
| small | no | local out-of-distribution state; add data around it (DART-style) |

---

## 4. Estimating very low failure rates

- **Plain Monte Carlo with exact bounds is the baseline.** The rule of three, 3/N one-sided. Two-sided Clopper-Pearson [D] for the scan: 0/2,000 gives an upper bound of 0.184%; 0/4,000 gives 0.092%; 1/2,000 gives 0.0013%-0.278%; 2/2,000 gives 0.012%-0.361%; 3/2,000 gives 0.031%-0.438%; 3/4,000 gives 0.015%-0.219%. The prereg quotes one-sided 0.15% for 0/2,000, which is consistent.
- **Cost of going deeper [D].** At 3.4 games/s: 1e-4 needs about 30,000 games (2.5 h); 1e-5 needs about 300,000 (24 h). That is within reach on this CPU-only machine, so go with plain Monte Carlo until rates under about 1e-5 matter.
- **Importance and adversarial evaluation.** Uesato et al. adapt rare-event estimation to agent evaluation: concentrate tests on adversarially chosen conditions while keeping the failure estimate unbiased, with a continuation method that learns failure modes from related, less robust agents; they report finding failures and estimating rates orders of magnitude faster than vanilla Monte Carlo on humanoid control and driving (STATED, [arXiv 1812.01647](https://arxiv.org/abs/1812.01647)). `rare-failure-robustness.md` already discusses this paper (R1) and its caveats: importance sampling gives unbiased estimates, not a zero-failure certificate.
- **Multilevel splitting.** Adaptive multilevel splitting (Cerou and Guyader) clones trajectories that reach progressively higher levels of a score function toward the rare event; the adaptive version matches the precision of a fixed-level scheme with optimally placed levels, and has a central limit theorem (STATED, [arXiv 1408.6366](https://arxiv.org/abs/1408.6366), from the search summary, not the paper text). It needs a score that rises toward failure and the ability to copy a state mid-run.
  - Candidate scores for this game (INFERENCE): the minimum clearance to the next obstacle; or the negative top-2 logit margin; or the count of recent solver-clone disagreements. Which of these actually predicts death is unknown, and a score uncorrelated with death makes splitting no better than plain Monte Carlo.
  - The games are seeded and deterministic, so a state can be reproduced by replaying actions from the seed, which stands in for a state copy (INFERENCE, subject to the repo's RNG-sharing note: clouds, stars and blink share the game RNG, so branching trajectories after a split do not stay comparable unless the later RNG stream is either fixed or deliberately redrawn; I did not check what `createGame` exposes).
- **Cheaper stand-ins that need no splitting (INFERENCE, all untested):**
  - *Noise-stress curve.* Flip the clone's action with probability e per decision, and play the clone and the solver at the same e. Each curve is a failure rate against e. If the clone's curve rises earlier than the solver's, it is fragile near e = 0 in a way the clean games cannot see. Failures at e > 0 are plentiful, so this needs hundreds of games, not thousands.
  - *Teacher-prefix handoff.* Run the solver to a late score (say 900k, where v2 is hardest) and let the clone take over, many times from different seeds, for a short window. It concentrates evaluation on the late hazard that ES also showed (`rare-failure-robustness.md`). The conditional hazard differs from the full-game hazard, because the state at handoff is the teacher's, not the clone's.
  - *Margin watch.* The smallest top-2 margin over 600k frames is already recorded. Score each failing seed by the margin at its fatal frame.

---

## 5. What I could not verify

- The exact statements of Ross et al.'s Theorems 2.1-3.3 (only the abstract, a search snippet and the section outline were readable).
- The T^2 eps bound for BC and any constants, taken from memory.
- Whether Learning by Cheating used DAgger-style relabelling, and the teacher-student scheme in Lee et al. 2020.
- DART's noise-selection procedure beyond "approximates the learner's error", and any theorem relating it to DAgger.
- A Karpathy primary source on this topic.
- Whether the repo has the 44-feature ambiguity check already; the grep was narrow.
- Whether `createSolver` and `createGame` can be driven from a replayed state and handed over mid-game (prereg §4 flags the same risk).

---

## 6. Which decision this changes

The prereg (§1) asks D1: is DAgger or noise injection the next clone-research experiment, or is that branch pruned? Its §5 readings map to the following. The shared tools: the solver can label any state the clone reaches; play is about 3.4 games/s on 18 workers; retraining is CPU-only (Python numpy or the Node backend). I have not timed a retrain, so I give no retrain cost; the demo set is 2.1M frames.

**What the literature adds to the readings.** It does not move the thresholds. It changes what to do inside each reading: the counts say whether a problem exists; the §3 tests say which fix applies. Do the three §3 tests on every failure before choosing a method.

### 6.1 Zero failures (H0)
- Prereg reading: prune robustness work; shapes and the pipeline (D2) next.
- Literature check: consistent. BC does well when errors are small and mistakes recoverable (Spencer et al., [2102.02872](https://arxiv.org/abs/2102.02872)), and the clone's 0/4,000 bound is under 0.092% two-sided [D].
- Cheap additions, each minutes (all untested, INFERENCE):
  1. **Noise-stress curve, clone vs solver** (§4). It tests whether the clone is fragile in a way clean games cannot show. A flat match means the prune stands with more confidence; an early rise gives a concrete reopen trigger without waiting for a field failure. Pre-register it with the positive control (e = large kills both).
  2. **Feature-ambiguity count on the demo set** (§2.4 class 1): how many near-identical-feature pairs carry different solver actions. This is seconds and is the premise for smaller shapes, where margins thin; do it before the shape sweep.
  3. If shapes shrink, rerun the scan at that shape. Robustness is the reopen trigger already listed in prereg §6.

### 6.2 One or two failures (low-rate failure)
- Prereg reading: diagnose each; robustness stays open at low priority; targeted probe first.
- Do, in order, per failing seed (each is minutes):
  1. Confirm the solver passes (arm S), else H2.
  2. Run the three §3 tests: disagreement trace, back-up-and-rescue, feature-ambiguity neighbour.
  3. **Retrain-seed check.** C2 and N1 already give two further models on the same seeds. A failure shared across all clones points to the state (ambiguity or a hard zone); a failure in one only points to that model's fit. The prereg already reads "C1 = 0 and C2 > 0" as seed-dependent fragility.
- By diagnosis:
  - *Drift (large back-up n, no demo neighbour)*: one targeted DAgger round, not a program. Roll out the clone on, say, the failing seed and a few hundred neighbouring seeds, label the visited states with the solver, add to the demos, retrain, and re-run the scan's 2,000 paired seeds. Success is the failure fixed with no new ones; the scan's power cannot see a rate change from 1/2000, so judge the fix on the failing seed and on the disagreement rate, not on the 2,000.
  - *Fatal-frame, no conflicting neighbours*: DART-style augmentation near that state (solver demos with action noise around the death zone) or simply more demo seeds. Do not run DAgger: u is large at a fatal frame and the aggregated data adds the same state.
  - *Fatal-frame, conflicting neighbours (imitation gap)*: no training method fixes it. Check whether one more legal feature disambiguates it; the repo's rule is that `observe()` changes need a clear gain (memory: policy.js features), and the bar is one real failure traced to ambiguous features.

### 6.3 Three or more failures, solver passes (H1 supported)
- Prereg reading: pre-register a DAgger or noise-injection experiment targeting this failure set.
- Pre-register with the §3 table first; the choice between arms follows the diagnoses.
  - **Majority drift**: DAgger arm. Each round: 200-500 clone rollouts on fresh seeds, solver-label every visited state (or only states where the clone's margin is low or its action differs, to save size), aggregate, retrain from the previous weights, 3-5 rounds. Positive control: the round-0 clone must fail on the target seeds. Compare with the DART arm (noise level matched to the clone's measured disagreement rate, §2.3) at equal data size, since DART needs no clone rollouts. Both arms vs. an equal-size plain-BC arm (more solver seeds) to separate the effect of state coverage from the effect of data volume, which is the confound the literature rarely controls (INFERENCE).
  - **Majority fatal-frame, no ambiguity**: the same two arms with data concentrated at the death zones (pterodactyl heights and speeds from `tools/wall-diag.js` classes), not DAgger rounds.
  - **Majority ambiguity**: stop training experiments; report to the user that the feature set has a blind spot, with the pairs.
- Evaluate on a fresh scan range with the same paired design; count the original failure seeds separately from new ones so a fix of old failures at the cost of new ones is visible. Held-out offsets from 325 are reserved for a ship claim.

### 6.4 Any seed where the solver also fails (H2)
Unchanged: stop clone work and report the core issue first. The literature has nothing to add.

### 6.5 Drift versus fatal-frame, in one line
Drift: relabel (DAgger) where the clone goes. Fatal frame: add data or noise at that state; if the labels conflict, the features are the problem. The rescue test (§3, test 2) is the cheapest way to tell which, and it is what the prereg's trace should be extended with before the scan's failures are read.

### 6.6 One-line effect on D1 and D2
- **D1:** no change to the thresholds. If the scan shows any failure, run the §3 tests before choosing between DAgger and DART; if it shows none, run the noise-stress curve before declaring the branch pruned.
- **D2 (Node vs Python):** the literature does not bear on it. The Node clone's higher loss (0.0056 vs 0.0044) matters only through its disagreement rate, which the same trace measures, so add the clone's disagreement rate on the solver's own trajectories to the N1 vs C1 comparison.
