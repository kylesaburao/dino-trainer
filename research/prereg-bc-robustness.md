# Pre-registration: does the solver clone have failure modes the solver lacks? (large-n robustness scan)

Status: **registered; advisor amendments applied (STOP threshold, arm order, oracle-disagreement trace, bound wording, --wasm).** Author: mac-3 (orchestrator), 2026-10-10, on main `d4a1224`.
Process: AGENTS.md "Research method"; retrospective `handoff/2026-10-10T1635Z-retrospective.md` §4 and §6.

## 1. Decision this informs
Clone research has priority on the Mac. Its candidate directions are:
- robustness work (DAgger relabelling, noise injection);
- smaller or different shapes;
- the Node vs Python backend;
- ES fine-tuning from a clone.

**Robustness work and ES fine-tuning only pay off if the clone fails somewhere.** The shipped clone's evidence is 150/150 + 25/25 + 25/25 on fresh full games. That bounds its per-game failure rate only below about 2.4% (95% CP upper bound on 0/150). That is too loose to say whether a covariate-shift failure mode exists at the 0.1–1% level, which is the level that matters for "reliably reach 1M".

The decision:
- **(D1)** Is a robustness method (DAgger or noise injection) the next clone-research experiment, or is that branch pruned at this scope?
- **(D2)** Is the Node backend good enough to be the default for clone-research runs, so the research needs no Python? Or does Python stay the reference?

This decision is also the decision log's own reopen trigger: "a failure mode appears in the clone that the solver doesn't have".

## 2. Hypotheses and what each predicts
- **H0 (no material covariate shift):** the clone's per-game failure rate is below about 0.15%. Prediction: 0 failures in 2,000 games.
- **H1 (covariate shift):** small action errors take the clone into states the solver never visits, where it fails. Prediction: ≥ 3 failures in 2,000 games, on seeds where the solver passes.
- **H2 (unfair game):** the solver also fails on some seeds, so the game, not the clone, is at fault. That would contradict the fairness model (a core bug). Prediction: the solver fails on a seed where the clone failed.

For the Node vs Python question, one more hypothesis:
- **H3 (Node backend is weaker):** the Node clone's higher final loss (0.0056 vs Python's 0.0044) means real play errors. Prediction: the Node clone fails more often than Python s1 on the same seeds.

## 3. Premise check (done, from existing data)
- **Throughput:** 150 full v2 games to 1M took 44 s on 18 workers (`runs/exp-bc-clones/cert-175-324-clone-s1.txt`), about 3.4 games/s. So 2,000 games take about 10 min per arm.
- **Seed hygiene:**
  - demonstrations use seeds 1–40;
  - validation uses [2^30, 2^31) and held-out uses [2^31, 2^32) (`trainer/env.js`);
  - the feature-sufficiency probes used 5000–9199, 100000–100199, 200000–200024 and 900000+.

  The scan uses **seeds 30,000,000–30,001,999**. They are in the ES training range, which is irrelevant to a supervised clone. They spend no held-out offsets (the next is still 325) and overlap no demonstrations.
- **Models exist:**
  - shipped `runs/bc-v2-h32x32/best.json` (Python, SEED=1);
  - replicate `runs/exp-bc-clones/model_base_s2.json` (Python, SEED=2);
  - Node clone `/Users/kyle/Desktop/dino-game-bc/runs/exp-bc-repro/node-s1/best.json` (control (c); 25/25 validation; trained on the same converted demos);
  - ES gen 970 `runs/v2-h32x32-1m/best.json`.
- **The solver is deterministic and its crashes are observable** (`bc-gen` counts them; 0 so far). It is replayed only on seeds where a clone failed.

## 4. Design
- **Game:** v2 full games from 0, target 1,000,000, the unmodified game. `--wasm` is allowed: AGENTS.md records it as bit-identical to JS on arm64 and x86.
- **Runner:** a new script in `research/scripts/prereg-bc-robustness/`, because `eval` only indexes held-out seeds. Outputs go to `runs/exp-bc-robust/` (gitignored). Copy the Node clone's `best.json` there, so the `dino-game-bc` worktree can be removed later.
- **Order (each arm gates the next):** P first, and read it before going on (STOP rule below). Then C1, which carries the shipped claim. Then N1. Then C2. Then S on the failing seeds.
- **Arms (paired: the same seeds in every arm):**

  | Arm | Model | Seeds |
  |---|---|---|
  | C1 | shipped clone, Python s1 | 30,000,000–30,001,999 (2,000) |
  | C2 | replicate clone, Python s2 | same 2,000 |
  | N1 | Node clone s1 | same 2,000 |
  | **P (positive control)** | ES gen 970 | the first 500 of them |
  | S (diagnostic) | `test/solver.js` | only seeds where any clone arm failed |

- **Positive control (P):** ES gen 970 has failed 10 of 225 fresh held-out games (offsets 100–324), about 4.4%. On 500 seeds it should fail about 22 times. **If P records fewer than 3 failures, the harness is not detecting crashes: STOP, and read nothing.** With 3–7 failures, proceed and note that the rate came in low (ES's 10/225 has a 95% CI of about 2.1–8%). Cross-check that two or three P failures die in the known zones (460–740 points, or ≥ 990k).
- **Power:**
  - With 0/2,000, the 95% CP upper bound on the failure rate is 0.15%.
  - A true rate of 0.3% gives ≥ 1 failure with probability 99.8%, and ≥ 3 failures with probability 94%.
- **For each failure, record:** seed, score at death, the killing obstacle (type; pterodactyl height class), speed, dino state, the actions over the last 30 frames, and whether the solver passes that seed (arm S). `tools/wall-diag.js` already classifies most of this; it reads a checkpoint θ, not `best.json` params, so adapt it.
- **Oracle-disagreement trace (separates H1 from a plain error):** for each clone failure, replay the clone's own trajectory on that seed. At every frame, query what the solver would do in that state, and log each frame where the clone's action differs, with its score.
  - The first disagreement long before the crash, then drift into states the solver never visits: **covariate shift** (H1 proper; the DAgger target).
  - The only disagreement is at, or a few frames before, the fatal obstacle: **an in-distribution action error**. The fix is more data or noise injection, not DAgger.
  - The solver must be queried from the clone's state. If `createSolver` can't be driven from a mid-game state the clone reached, report that rather than approximating.
- **Cost:** about 6,500 games, roughly 32 min on the Mac. Nothing else heavy runs on the Mac meanwhile.

## 5. Readings (fixed in advance)
**D1, from C1 and C2 together (4,000 games), counting only failures where S passes:**

| Result | Reading | Next |
|---|---|---|
| 0 failures | H0 at this scope. The shipped clone's rate is < 0.15% (0/2,000, from C1). The training procedure's rate, averaged over training seeds, is < 0.075% (0/4,000; C1 and C2 are two models on the same seeds, so this bounds the procedure, not each clone). **Robustness work is pruned (scope below).** | Clone research moves on to shapes and to the pipeline (D2); ES fine-tuning from a clone is not needed for reliability |
| 1–2 failures | A low-rate failure exists (about 0.025–0.05%). Diagnose each one; it does not justify a training method on its own. | Robustness stays OPEN at low priority; a targeted probe of the failure mechanism comes first |
| ≥ 3 failures, solver passes | **H1 supported.** | Pre-register a DAgger / noise-injection experiment next, with this failure set as its target |
| Any seed where the solver also fails | **H2:** the game itself can't be passed on that seed. | Stop clone work and report the core issue to the user first |

C1 alone carries the shipped claim. If C1 = 0 and C2 > 0 (or the reverse), the reading is seed-dependent fragility, and robustness stays OPEN.

**D2, from N1 against C1 (paired):**
- **Node acceptable as the research default:** N1 failures ≤ C1 failures + 2.
- **Node weaker (H3):** exact McNemar on the discordant seeds gives one-sided p < 0.05. Python stays the reference backend; open the loss gap (0.0056 vs 0.0044) as a question for the Node optimiser.
- **Otherwise:** inconclusive. Node may be used, but a clone headed for shipping is trained with Python.

**This scan is not a certification.** It uses non-held-out seeds. Any ship claim still needs fresh held-out offsets from 325.

## 6. Scope and reopen triggers
- **Scope:** v2; the 44 unchanged features; core hash `0f329d4e…`; 44-32-32-3 clones trained on seeds 1–40 demos; full games from 0 to 1M; the JS forward pass.
- **Reopen robustness (if pruned)** on any of:
  - a field failure in the browser;
  - a new game version or a changed `observe()`;
  - a smaller shape (whose margins are thinner);
  - the CSS evaluator diverging from JS.
