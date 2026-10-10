# What the research found, and would it have sped up the project? (2026-10-10, mac-3)

Written at the user's request: "Elaborate everything this repository and claude has researched and found. I would like to know if this automated research actually turned up anything 'useful' that could've sped up development when this project was created."

Sources: a sourced inventory of every research note, handoff, experiment and commit, compiled read-only against main `da74010`. Its key claims were spot-checked against git:
- `2da34de` 10-09 16:40;
- `bee6b90` 17:17;
- `57125a5` 18:05;
- `0e62018` 10-10 09:54;
- TrainingPrompt.md (`c58f203`), lines 108–111 and 155.

Times are PDT.

## 1. Short answer
**Yes, one finding would have sped the project up enormously, and it was available on day one: behaviour-clone the solver.** Almost nothing else would have.

- **The ES-only phase** ran from the first ES run (10-09 17:19) to the first cloning attempt (10-10 about 08:42). That is about 15 h of wall clock, about 7.5 h of active work, 42 commits and about 40 experiment runs. It never produced a model better than the one ES found in its first 74 minutes (24/25 to 1M).
- **From the first cloning attempt to shipping took about 1.2 h.** Training the clone took about 20 s, and both probes used about 25 CPU-minutes.
- **The clone is clearly better:**
  - 150/150 on fresh held-out seeds, against 143/150 for ES (paired McNemar p = 0.008);
  - 0 crashes in 2,000 extra games for each of three clones, against 18/500 for ES.

**Everything needed for this existed at project creation:**
1. **The game shipped with a solver.** `requirement()`/`placeFairly()` were in the first commit, and the spec itself says "(they are a solver)" (TrainingPrompt.md:111).
2. **A solver-driven "cheat" controller was written during scaffolding** (`test/record-golden.js`, `65051c3`), one second before the ES trainer commit. The exact `test/solver.js` existed 48 minutes later (`57125a5`).
3. **The spec forbade solver output only "as observations or as reward shaping"** (lines 108–111), not as training labels. It made ES only a "Suggested default (switch only with a reason recorded in PROGRESS.md)" (line 155).
4. **Behaviour cloning from a privileged teacher is a standard, published technique:** "Learning by Cheating", 2019, and DAgger, 2011. A 15-minute necessary-condition test would have shown that the 44 features suffice. Even a 44-8-8-3 net trained on 50k frames reaches 1M.

**Nobody considered it until the user's own hypothesis forced the test.** No document or commit message before 10-10 08:50 mentions cloning or imitation. The archived handoff lists demonstration-based methods only as *start-state curricula for ES*. Cloning arose when the user said "the input features as is may not be sufficient". The feature-sufficiency probe cloned the solver to test that claim, and the clone turned out to be the best model. **The decisive idea came from an empirical probe answering the user's question, not from the literature research.**

**Caveat on the counterfactual.** This saves time only if the goal was "an AI that reliably reaches 1M." If the goal was "neuroevolution learns the game", the clone doesn't answer it. On that goal ES still hasn't beaten its first model, and the second wall is still open research. The user's approval ("Shipping the clone is allowed") was needed for the pivot. At creation, the spec's ES default would have needed the same explicit sign-off.

## 2. Everything that was found, by value

### 2.1 Findings that changed what shipped or how fast things run
| Finding | How it was found | Effect |
|---|---|---|
| **The 44 features suffice. A solver clone plays to 1M** (352/352 probe games; then 150/150 certified, 0/6,000 scanned) | Empirical probe (`research/feature-sufficiency.md`) | **The ship** (`0e62018`), and `observe()` stays unchanged |
| **The shipped ES model's crashes are behavioural, not missing information.** Early crashes are premature jumps and a stuck jump key; late crashes come from never fast-dropping under fast, high pterodactyls | Empirical probe A | It explained the failures and pruned the feature-change branch without a retrain |
| **WASM SIMD forward pass: 1.65–1.7× faster, bit-identical on arm64 and x86** | Empirical prototype (`research/wasm-simd-mlp.md`) | `--wasm`; it also sped up the certification and the scans |
| **v2 had a fairness bug** (the solver crashed within a few hundred frames at level 1) | Writing `test/solver.js` on day one | Fixed before any v2 training |

### 2.2 Findings that explained the problem, but didn't produce a better model
- **The plateau is a seed lottery, not a tie problem.**
  - P(escape by gen 900) is about 0.25, and the Kaplan–Meier median is about 1,200 generations.
  - Racing 2–3 seeds to about 1,500 generations is the efficient lever.
  - This reframed hours of tie-mechanics work.
- **The second wall** (first pterodactyls, 500–700 points) causes most ES certification failures. The mechanism depends on the seed.
- **The 54-point plateau** is an obstacle-blind "bunny-hop" limit cycle. Index tie-breaking was what let ES escape it.
- **Trainer bugs** were found and fixed: a frozen-best deadlock and inert coupled weight decay.
- **Bounds that pruned whole directions** without runs:
  - GPU and Rust ports (episodes are serial, and the MLP is about 80% of the cost);
  - `actionRepeat` 2 (≤ 1.67×, and it breaks the solver guarantee);
  - early termination of episodes (≤ 8%).

### 2.3 Durable infrastructure (reused by the clone work)
- Disjoint training, validation and held-out seed ranges.
- `eval --seed-offset` with Clopper–Pearson intervals.
- The worker pool.
- `--wasm`.
- `test/bit-identity.js` (7 scenarios, about 9 s).
- Cross-architecture determinism, to 1e-9.

Any project that has to *certify* a model needs these, so this work was not wasted.

### 2.4 Process lessons (learned by making the mistakes)
From the retrospective (`handoff/2026-10-10T1635Z-retrospective.md`):
- pre-register each run;
- include a positive control inside the horizon;
- measure between-seed variance before comparing arms;
- check every premise in a brief against code or data;
- size certifications for the claim;
- prune with the cheapest discriminating test.

Applied from the start, these would have avoided most of the about 40 non-discriminating runs. Examples:
- 26 runs censored by a 400-generation horizon, while the known-good arm needed about 780 generations;
- a σ controller built on a premise the existing data already contradicted.

### 2.5 Research that produced nothing actionable
Of 13 research notes, **only 2 caused commits** (wasm-simd and feature-sufficiency). Both were empirical.

| Note | Outcome |
|---|---|
| Karpathy training techniques | Found that Karpathy has essentially no ES content. No action. The retrospective calls the topic "chosen by authority, not by the decision it serves" |
| Observation design | Its candidate features (tau, speed/22) changed nothing in probe B |
| ES practice | Its σ-on-ties family failed (0/14). The specific failing controller came from elsewhere |
| Width/depth transfer, rare-failure robustness, validation selection | No runs or code; overtaken by the clone |
| Demo-bootstrapped ES | Literature on the right idea, but it arrived after the probe was already running. **It predicted that plain BC would not be enough and ES fine-tuning would be needed; that was wrong** |
| Distillation robustness | Proposed diagnostics that were never exercised, because no clone has failed |

### 2.6 Dead ends (cost)
- Trainer flags W1–W6: Run A 46/75, Run B 0/75.
- The stall rule.
- The tie-rate σ controller: implementation plus 14 runs.
- The 400-generation proxy: 26 censored runs.
- The gated curriculum: merged, but never trained with during the ES phase. Its first test (Run B′) is running now, and it doesn't reach the 400–800 zone.

## 3. What would have sped up development, ranked
1. **"A solver exists. Try imitation first, as a 15-minute necessary-condition test, before designing an RL/ES pipeline."** It would have saved about 14 h of ES reliability work and given a better model. The only cost is the owner's decision on solver-derived labels, and the spec already allowed them.
2. **The pre-registration and positive-control discipline,** from the first experiment. It would have avoided most of the about 40 runs that couldn't discriminate between options.
3. **`--wasm` early.** It would have made every ES run about 1.7× faster. That speeds up the work but doesn't change the outcome.
4. **Pick research topics by the open decision, not by a source.** The decision-driven notes (WASM, feature sufficiency) paid off; the authority-driven ones didn't.

## 4. What is still open
- **ES as research:** Run B′ (dense late exposure) against hygiene is running on desktop-1. Natural-ramp starts are being built on desktop-2. Neither affects the shipped model.
- **Clone research:**
  - robustness is pruned at its scope (0/2,000 per clone);
  - the end-to-end pipeline check (`research/prereg-bc-pipeline-e2e.md`) is running;
  - next are smaller shapes, but only if a decision needs them (for example, CSS evaluator speed).
