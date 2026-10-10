# Pre-registration: does the solver clone have failure modes the solver lacks? (large-n robustness scan)

Status: **run 2026-10-10; results at the end** (registered; advisor amendments applied: STOP threshold, arm order, oracle-disagreement trace, bound wording, --wasm). Author: mac-3 (orchestrator), 2026-10-10, on main `d4a1224`.
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

## Results (run 2026-10-10, mac-3 executor, branch `wt/bc-robust` from main `5f0edc8`)

Run exactly in the §4 order: P → STOP check → C1 → N1 → C2 → S + oracle trace. Scripts: `research/scripts/prereg-bc-robustness/` (`scan.js` runner on the trainer's worker pool, the same job path as `eval`; `diag.js` per-failure diagnosis, arm S and oracle trace; `analyze.js` statistics). Raw per-seed scores, logs and diagnosis JSON: `runs/exp-bc-robust/` (gitignored). Core hash `0f329d4e3493…` asserted by the runner. 18 workers, `--wasm`.

**Models checked before running** (all `dino-mlp-v1`, layers 44-32-32-3, `gameVersion` v2, `actionRepeat` 1, four distinct parameter vectors): C1 `runs/bc-v2-h32x32/best.json` (identical params to the shipped `model.js`); C2 `runs/exp-bc-clones/model_base_s2.json`; N1 `runs/exp-bc-robust/node-s1.json` (sha1 `490e5b6d…`, byte-identical copy of `dino-game-bc/runs/exp-bc-repro/node-s1/best.json`); P `runs/v2-h32x32-1m/best.json` (generation 970).

### Per-arm failure counts (seeds 30,000,000 + 0..N−1, v2, full games 0 → 1,000,000)

| Arm | Model | Failures / games | Two-sided 95% CP | One-sided 95% upper | Wall time |
|---|---|---|---|---|---|
| P | ES gen 970 | **18 / 500** | 2.15% – 5.63% | 5.29% | 88 s |
| C1 | shipped clone, Python s1 | **0 / 2,000** | 0 – 0.184% | 0.150% | 347 s |
| N1 | Node clone s1 | **0 / 2,000** | 0 – 0.184% | 0.150% | 348 s |
| C2 | replicate clone, Python s2 | **0 / 2,000** | 0 – 0.184% | 0.150% | 348 s |
| C1 + C2 pooled | (two models, same seeds) | 0 / 4,000 | 0 – 0.092% | 0.075% | — |
| S | `test/solver.js` | not run: no clone arm failed (S is defined on clone-failure seeds only) | — | — | — |

Every passing clone game ran the full 1,943,237 frames. The §1/§4/§5 bounds "0.15%" and "0.075%" are the one-sided 95% CP upper bounds; the two-sided 95% CP upper bounds are 0.184% and 0.092%.

**Forward-pass equivalence (scope §6 says "the JS forward pass"; the scans used `--wasm`).** (a) All 18 P failures were replayed single-threaded with the JS `policy.js` `createAgent`; every replay died at exactly the score the wasm scan recorded (`diag.js` asserts this). (b) The first 36 seeds of each clone arm were re-scanned with the JS forward pass: 36/36 identical (score, frames) per arm.

### P gate
**18 / 500 failures ≥ 3: the harness detects crashes; proceed** (not in the 3–7 "came in low" band; 3.6% is consistent with ES's 10/225). Known-zone cross-check: 10/18 died at 496–663 points (all `cactus-small`, the 460–740 opening zone) — the cross-check is met. The other 8 died at 968,764–983,418 (all `ptero-high` at speed 21.96, dino airborne, last 30 frames `jump×15 none×15`): late, but below the ≥ 990k zone the pre-registration names. Noted, not a gate issue. The solver passes all 18 P seeds.

### Per-failure table
**Clone arms (C1, N1, C2): no failures, so no rows; arm S and the oracle-disagreement trace had no clone seed to run on.**

For harness validation only, `diag.js` (classification, arm S, oracle trace) was run on all 18 P failures. ES is not a solver clone, so these trace verdicts are not readings of H1; they show that the trace runs and that the world is action-independent (the obstacle-list hash of the model's game and the solver's own game never differed before the crash, on all 18 seeds).

| Arm | Seed | Score | Killer (ptero class) | Speed | Dino state | Last 30 actions | Solver passes | First disagreement (frame @ score) | Effective disagreements | Drift (first dino-state diff frame) |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 30000078 | 510 | cactus-small | 8.77 | air | jump×30 | yes | 251 @ 38 | 73 | yes (252) |
| P | 30000101 | 496 | cactus-small | 8.70 | air | jump×30 | yes | 251 @ 38 | 59 | yes (252) |
| P | 30000121 | 569 | cactus-small | 9.03 | ground | none×2 duck×1 jump×8 duck×1 jump×18 | yes | 247 @ 37 | 83 | yes (248) |
| P | 30000254 | 544 | cactus-small | 8.92 | air | jump×30 | yes | 242 @ 37 | 95 | yes (243) |
| P | 30000290 | 525 | cactus-small | 8.83 | air | jump×30 | yes | 245 @ 37 | 46 | yes (246) |
| P | 30000311 | 511 | cactus-small | 8.77 | air | jump×30 | yes | 248 @ 37 | 57 | yes (249) |
| P | 30000344 | 663 | cactus-small | 9.44 | ground | none×6 duck×1 jump×9 duck×1 jump×13 | yes | 248 @ 37 | 54 | yes (249) |
| P | 30000367 | 544 | cactus-small | 8.92 | air | jump×30 | yes | 248 @ 37 | 95 | yes (249) |
| P | 30000476 | 592 | cactus-small | 9.13 | ground | none×3 duck×1 jump×8 duck×1 jump×17 | yes | 248 @ 37 | 76 | yes (249) |
| P | 30000491 | 528 | cactus-small | 8.85 | air | jump×30 | yes | 249 @ 38 | 44 | yes (250) |
| P | 30000092 | 972,527 | ptero-high | 21.96 | air | jump×15 none×15 | yes | 245 @ 37 | 212,171 | yes (246) |
| P | 30000114 | 973,083 | ptero-high | 21.96 | air | jump×15 none×15 | yes | 246 @ 37 | 214,258 | yes (247) |
| P | 30000129 | 977,275 | ptero-high | 21.97 | air | jump×15 none×15 | yes | 248 @ 38 | 212,870 | yes (249) |
| P | 30000149 | 968,764 | ptero-high | 21.96 | air | jump×15 none×15 | yes | 250 @ 38 | 213,024 | yes (251) |
| P | 30000162 | 983,418 | ptero-high | 21.98 | air | jump×15 none×15 | yes | 249 @ 38 | 214,568 | yes (250) |
| P | 30000185 | 970,541 | ptero-high | 21.96 | air | jump×15 none×15 | yes | 250 @ 38 | 212,816 | yes (251) |
| P | 30000341 | 975,714 | ptero-high | 21.97 | air | jump×15 none×15 | yes | 249 @ 38 | 213,134 | yes (250) |
| P | 30000382 | 980,656 | ptero-high | 21.97 | air | jump×15 none×15 | yes | 249 @ 38 | 217,228 | yes (250) |

Oracle-trace operationalisation (fixed in `diag.js` before any clone result was seen): a shadow `createSolver()` is queried every frame on the model's own game, only the model's keys are applied. The solver's only private state is `hold`, read only while airborne and set only on a takeoff it chose, so the query is **exact on ground frames**, exact on airborne frames when the shadow chose the same takeoff frame, and **undefined** on airborne frames of a takeoff the shadow did not choose (counted separately, never compared; for P these were 1.4k–1.9k frames per opening death and ~1.14M per late death). Ground frames where the jump key is already held and the solver says release are counted as no-op ("keyheld"), not effective. "Near-fatal window" = from the frame the killing obstacle first had x < 600 (observable), and at least the last 30 frames; "covariate shift" = an effective disagreement before that window; "in-distribution action error" = all effective disagreements inside it. Drift = the frame-aligned dino state (y, ducking) differs from the solver's own trajectory on the same seed.

### McNemar, N1 vs C1 (paired, 2,000 seeds)
Discordant pairs: b (N1 fails, C1 passes) = 0, c (C1 fails, N1 passes) = 0. The exact test is **undefined (no discordant pairs)**; it cannot reach p < 0.05.

### D1 reading (applied verbatim from §5)
C1 + C2 = 0 failures in 4,000 games (no failure, so no S condition to apply). Row "0 failures": **"H0 at this scope. The shipped clone's rate is < 0.15% (0/2,000, from C1). The training procedure's rate, averaged over training seeds, is < 0.075% (0/4,000; C1 and C2 are two models on the same seeds, so this bounds the procedure, not each clone). Robustness work is pruned (scope below)."** Next: **"Clone research moves on to shapes and to the pipeline (D2); ES fine-tuning from a clone is not needed for reliability."** (Bounds are one-sided 95% CP; two-sided 0.184% / 0.092%.) C1 = 0 and C2 = 0, so the seed-dependent-fragility clause does not apply. No H2 seed (the solver was never needed; it also passed all 18 P seeds).

### D2 reading (applied verbatim from §5)
N1 = 0 ≤ C1 + 2 = 2: **"Node acceptable as the research default."** H3 (McNemar one-sided p < 0.05) is not met: no discordant pairs.

**This scan is not a certification** (non-held-out seeds); any ship claim still needs fresh held-out offsets from 325.

### Register row (for the §6.4 pruned-branch register)
| Branch | Verdict | Scope of evidence | Power / validity caveat | Reopen if… |
|---|---|---|---|---|
| Clone robustness work (DAgger / noise injection) | **pruned at this scope** (D1); Node backend **acceptable as research default** (D2) | C1, C2, N1 each 0/2,000 v2 full games 0→1M, seeds 30,000,000–30,001,999, core `0f329d4e…`, 44-32-32-3 clones on seeds 1–40 demos, wasm forward pass (JS-equivalent, checked); positive control P 18/500 | One-sided 95% upper 0.15% per clone, 0.075% for the procedure; blind to rates ≲ 0.1%; non-held-out seeds; S and the oracle trace never exercised on a clone failure | a browser field failure; a new game version or changed `observe()`; a smaller shape; CSS evaluator diverging from JS |

### Deviations from the pre-registration
- None in arms, seeds, order, models or readings.
- `--wasm` used for every scan (allowed by §4); JS equivalence evidenced above, since §6 scopes the result to the JS forward pass.
- Arm S and the oracle trace were not run on any clone seed because none failed; they were exercised on the 18 P failures as a harness check only.
- The pre-registration's "< 0.15%" / "< 0.075%" are one-sided 95% CP bounds; the two-sided values (0.184% / 0.092%) are reported alongside.
- P's late failures fell at 968k–983k rather than ≥ 990k; the known-zone cross-check is met by the 10 opening deaths.
