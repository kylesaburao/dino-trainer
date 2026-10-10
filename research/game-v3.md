# Rules v3: superhuman past 100k

Status (2026-10-10): rules v3 are implemented on `wt/game-v3`. The budget probe, the solver unit tests and a 40 + 40 game solver pilot are done. The full solver survey and the evaluation of the two shipped clones on rules v3 are pre-registered below and **queued**; neither has been run.

Core hash after the change: `63fa2ab5…` (before: `0f329d4e…`).

## 1. Definition

The user's request: a variant of the game that is effectively impossible for humans past 100k points, that keeps the existing difficulty engine, and that can evaluate current models and train stronger ones.

The user's ruling on "impossible for humans" is a **reaction budget below 150 ms**:
- For each **must-act** obstacle, the reaction budget is the time from its first visibility on screen to the latest frame on which a valid action can still start.
  - First visibility: the obstacle has entered the canvas (x < W), after `placeFairly` has parked it.
  - Latest valid action: the most lenient legal action for that obstacle on its own. That is the latest takeoff over all release lengths whose arc clears it, or, for a mid pterodactyl, the latest duck start if that is later.
- Past 100k points, most must-act obstacles must have a budget below 150 ms (9 frames at 60 Hz), and the budget must tighten with score.
- Games must stay legal and possible: an exact solver must never crash.

Probe conventions (`research/scripts/game-v3/budget.js`):
- **Must-act** means the obstacle would hit a grounded, standing dino on some whole frame at the current speed.
- **Not must-act:**
  - overhead obstacles (high pterodactyls; the dino just stays grounded);
  - "tunnelled" obstacles, which a grounded dino skips between two frames. Collision is checked per frame, and a single small cactus's overlap span is 42 px, so above 42 px/frame some pass unseen. These are counted separately.
- The latest jump is `tau − a_r`, maximised over `RELEASE_FRAMES` whose window `b_r − a_r − δ ≥ 0`, with `tau`, `a`, `b` and `δ` as in `core.js requirement()`. The latest duck (mid pterodactyls only) is `tau`.
- Budgets are fractional frames from the first visible frame; ms = frames × 16.67.

## 2. What sets the budget

- **The jump arc barely depends on speed.** `jumpVelocity(s) = −10 − s/10` grows with speed, but `maxJumpY` and `dropVel` cut the rise at y < 30, so the airtime of the canonical (release 6) jump is 31–34 frames at every speed from 6 to 100 px/frame. A faster game gives a faster takeoff, not a longer jump.
- **So the budget depends only on the obstacle's speed.** It is about (W − dino front − box offset) / v − a, where a is the 2–4 frames the arc needs to clear the obstacle's height. That gives 9 frames at v ≈ 44 px/frame, against about 20 at v2's top speed of 22.
- **Gaps stop mattering.** Above about 25 px/frame, one airtime (about 32 frames) covers more track than the largest gap. The fairness model's spacing (`minGroundFrames` between airborne intervals), not `gapCoef`/`maxGapCoef`, then sets the obstacle density: at most one must-act obstacle per about 34 frames.
- **Pterodactyl speed spread** (±1.6) moves half the pterodactyls slightly faster. Pterodactyls are only about 40% of must-act obstacles, so the spread alone cannot reach "most".
- **Conclusion:** one lever is enough, the speed cap. Rules v3 changes nothing else in the v2 schedule.

**Fairness at high speed.** v2's `requirement()` places the canonical takeoff at the centre of the takeoff window. The window is long (`b − a − δ` ≈ 25 frames), so above about 33 px/frame the centre lies *before the obstacle appears*: at v = 60 it is about 15 frames before the obstacle reaches the dino, but the obstacle is only visible for about 8.
- The exact solver would then act on obstacles it cannot see. That is legal for a privileged teacher, but no student on the 44 legal features could imitate it.
- The fairness model only proves that the *centred* schedule is compatible, so a purely reactive schedule was not guaranteed to exist.

Rules v3 therefore makes one fairness change, v3-only and above 100k:
- An obstacle spawned with e > 0 is frozen with `o.late = true`.
- `requirement()` then anchors its takeoff at `tau − a − minWindow/2`, inside the last `minWindow` frames of the window, instead of at the centre. Everything else (release choice, `minWindow`, `minGround`, the pairwise checks, `placeFairly`) is unchanged, so the generated sequences remain provably compatible under the late schedule.
- In the probe, the canonical takeoff comes after the obstacle appears for 100% of must-act obstacles at every v3 level. At 1M its median is 4.6 frames (77 ms) after appearance.

## 3. The engine change and the chosen endpoints

`core.js`:
- `GAME_VERSIONS` gains `'v3'`. `difficulty()` gives v3 the v2 level d.
- `superLevel(version, score)` = e = log10(score / 100k) / log10(V3.endScore / 100k), clipped to [0, 1]. It is 0 on v1/v2 and at or below 100k.
- `rulesAt(d, r, e)` keeps every v2 value. Only when e > 0 is the speed cap set to `lerp(v2 cap at 100k = 19, V3.maxSpeed, e^V3.curve)`.
- Every `version === 'v2'` branch (committed airborne intervals, the takeoff-ahead push, per-frame rules) now reads `version !== 'v1'`, so v3 inherits all of v2's fairness machinery.
- Spawned obstacles carry `late`, and `game.superLevel` exposes e.

**Endpoints: `V3 = { startScore: 100000, endScore: 1000000, maxSpeed: 60, curve: 0.5 }`.**

| Score | v2 speed cap | v3 speed cap |
|---|---|---|
| 100k | 19.0 | 19.0 |
| 110k | 19.1 | 28.1 |
| 150k | 19.6 | 36.5 |
| 200k | 19.9 | 41.7 |
| 500k | 21.1 | 53.3 |
| 1M and above | 22.0 | 60.0 |

**Targets** (stated before choosing): v2 stays human-feasible (0% of budgets under 150 ms at every level), while v3 reaches ≥ 50% under 150 ms at 200k, ≥ 85% at 500k and ≥ 90% at 1M, with a budget median that falls monotonically with score. A front-loaded curve (`curve` 0.5) is needed. With a plain log-lerp (`curve` 1), 50% at 200k would need a cap above 100 px/frame at 1M, where most small cacti tunnel.

**Budget table** (`node research/scripts/game-v3/budget.js --games 12`): 12 solver games per cell, each 2,000 points from the start level, with the speed starting at that level's cap. The jump and duck solvers give identical rows because the obstacle sequence does not depend on the player while no crash occurs. 0 crashes in every cell.

| Rules | Start level | Speed at end | Must-act | Tunnelled | Budget median | p90 | Share < 150 ms |
|---|---|---|---|---|---|---|---|
| v2 | 100k | 19.0 | 1208 | 0 | 23.1 f / 386 ms | 26.3 f / 439 ms | 0.0% |
| v2 | 200k | 19.9 | 1171 | 0 | 22.0 f / 367 ms | 25.4 f / 424 ms | 0.0% |
| v2 | 500k | 21.1 | 1127 | 0 | 20.6 f / 343 ms | 24.2 f / 404 ms | 0.0% |
| v2 | 1M | 22.0 | 1092 | 0 | 19.6 f / 327 ms | 23.1 f / 386 ms | 0.0% |
| v3 | 100k (to 102k) | 22.8 | 1114 | 0 | 20.6 f / 344 ms | 24.3 f / 404 ms | 0.0% |
| v3 | 110k | 28.1 | 840 | 0 | 14.9 f / 249 ms | 17.6 f / 294 ms | 0.0% |
| v3 | 150k | 36.5 | 655 | 0 | 10.5 f / 175 ms | 13.3 f / 222 ms | 0.0% |
| v3 | 200k | 41.7 | 555 | 0 | 8.8 f / 147 ms | 11.9 f / 198 ms | **55.5%** |
| v3 | 500k | 53.3 | 446 | 5 | 6.7 f / 112 ms | 9.1 f / 151 ms | **87.7%** |
| v3 | 1M | 60.0 | 402 | 0 | 5.6 f / 93 ms | 8.4 f / 140 ms | **100.0%** |

(110k and 150k rows for v2: 0.0%, medians 23.0 f and 22.5 f.)

The bar is met from about 200k. Between 100k and about 180k the budget falls from 386 to about 150 ms but most obstacles still give more than 150 ms: see open choice 1.

Explored alternatives (6 games per cell, same probe, `--set`):

| Setting | 110k | 150k | 200k | 1M | Note |
|---|---|---|---|---|---|
| maxSpeed 60, curve 0.5 (**chosen**) | 0% | 0% | 55% | 100% | tunnelled ≈ 0 |
| maxSpeed 60, curve 0.25 | 16% | 80% | 82% | 100% | "most" from about 140k; very abrupt after 100k |
| maxSpeed 70, curve 0.5 | 0% | 47% | 80% | 100% | 12 tunnelled of 162 at 1M (7%) |

## 4. Rules v1 and v2 unchanged (proof)

- Rules v1: the existing goldens (`test/golden/`, from baseline `2da34de`) still pass, frame for frame (`test/parity.test.js`).
- Rules v2: before `core.js` was edited, commit `29d189b` added 7 v2 golden traces (`test/golden-v2/`, 36,000 frames). `test/record-golden-v2.js` recorded them from `core.js` and `test/solver.js` at `10cd0e0`. They cover the 1k onset, 30k (groups of 4 from 31.6k), 148k → 151k and 998k → 1M, with both solvers and random keys (crashes, restarts, fast-drops, ducks). `test/v2-parity.test.js` passes them all after the change, and `test/v3.test.js` replays the three traces that start below 100k on rules v3, frame for frame.
- Behaviour-cloning demos regenerate byte for byte: `bc-gen` with the default teacher gives sha256 `3a67bfd4e109d6bb…` (2,096,268 rows), and with `--teacher duck` gives `363872f61a785729…`, equal to `runs/exp-bc-duck/v2-demos-duck-s1-40`. `test/bc.test.js` and `test/net.test.js` pass.
- `node test/bit-identity.js <10cd0e0 worktree> <this tree> --core-changed`: IDENTICAL, 7 scenarios, 48,603 values (again after rebasing onto `b4624b3`). The new `--core-changed` flag reports the core hash instead of comparing it.
- **Core hash.** It changed (`0f329d4e…` → `63fa2ab5…`). That invalidates pre-v3 checkpoints (`--resume` needs `--force`) and pre-v3 demos (`bc-train` refuses them without `--force`). For rules v1/v2 work, `--force` is justified because the goldens, the regenerated demo hashes, `net.test.js` and bit-identity show identical games and decisions. Never reuse them for rules v3.

## 5. Solvers on rules v3

`test/solver.js`:
- **Lookahead.** The fixed `tau > 80` prune is now `tau > horizon(speed)` = max(80, longest airtime + 2). Every takeoff lies within one airtime before tau, and late anchors lie later still, so the prune is exact. Because the airtime is capped (§2), the horizon is 80 at every speed: the decisions are unchanged and it stays correct if the arc ever grows.
- **Newcomer bound.** The duck solver's `MAX_PTERO_OFFSET` now comes from `core.js` `MAX_PTERO_SPEED_OFFSET`: the largest pterodactyl offset any version draws, 1.6 on rules v2 and v3 (rules v3 does not change the spread).
- v1/v2 decisions are bit-identical (§4: demo hashes).

Results so far, all 0 crashes:
- `test/v3.test.js`: both solvers, 3 seeds each from 100k to 400k; plus 6 seeds × starts 300k, 600k, 1M × 20k points each; duck-solver `conflicts` 0.
- Pilot survey (not the pre-registered survey; seeds 50,000,000–50,000,039, 2 workers, about 8 s each): `survey.js --game v3 --start 100000 --cap 1000000`, 40 games per solver from 100k to 1M.

  | Solver | Crashes | Obstacles seen | Mid pterodactyls ducked |
  |---|---|---|---|
  | jump | 0/40 | 928,644 | 0 (it never ducks) |
  | duck | 0/40 | 928,644 | 134,341 of 134,546 (0 fallbacks) |

## 6. Pre-registered: full solver survey on rules v3

- **Decision.** Can rules v3 be declared solvable ("an exact solver never crashes") and serve as a training target with either teacher?
- **Hypotheses.**
  - H0: the late-anchored fairness model is exact on v3, so 0 crashes.
  - H1: a rare discretisation gap appears at v3 speeds, where `minWindow/2` gives about 1 frame of slack against the solver's ±0.5-frame rounding, and some games crash.
- **Premises (checked).** The probe shows 0 crashes in 288 short games. The unit tests and the 80-game pilot show 0 crashes. Late anchors keep every canonical takeoff after visibility.
- **Arms.**
  - `--solver jump` and `--solver duck`: N = 3,000 games each, from 100k to 1M, seeds 51,000,000–51,002,999, which are disjoint from the pilot.
  - Positive control: crashes are detectable on rules v3. Both shipped clones (v2 models on rules v3) crashed at 102.5k–103.7k on 6 of 6 held-out seeds (offsets 1000–1005; quick `runEpisode` check); the full evaluation is §7.
- **Commands** (2 workers, so as not to disturb the scan job on this Mac; about 0.35 s per game per worker, so about 9 min per arm):
  ```sh
  mkdir -p runs/exp-game-v3
  nohup node research/scripts/prereg-teacher-duck/survey.js --solver jump --game v3 --start 100000 --cap 1000000 --base 51000000 --n 3000 --workers 2 --out runs/exp-game-v3/survey-jump-51000000-3000.jsonl > runs/exp-game-v3/survey-jump-51000000-3000.log 2>&1 &
  nohup node research/scripts/prereg-teacher-duck/survey.js --solver duck --game v3 --start 100000 --cap 1000000 --base 51000000 --n 3000 --workers 2 --out runs/exp-game-v3/survey-duck-51000000-3000.jsonl > runs/exp-game-v3/survey-duck-51000000-3000.log 2>&1 &
  ```
  The dashboard's Scans section lists both files, with game v3.
- **Reading.**
  - 0/3,000 per arm gives a 95% CP upper bound of 0.12% per game on the crash rate. Accept H0, and rules v3 is solvable to that bound for that solver.
  - Any crash means H1. The row's `onScreen` and `last30` say why. Classify it with `research/scripts/prereg-teacher-duck/crash-classify.js` (adapted for `--game`), fix the fairness model (v3-only), and re-run with fresh seeds. Rules v3 must not be used as a training target before that.
  - Duck fallbacks above 1% would not block anything, but would be recorded as the newcomer bound biting at high speed.
- **Scope.** Start 100k only, at natural speed (no `startScore` past 100k). Starts above 1M are covered by the unit tests only.

## 7. Queued: the shipped clones on rules v3 (cross-rules evaluation)

- **Decision.** It gives the baseline for "train even stronger models": how far v2 models get on rules v3, and where.
- **Expectation.** Both fail just past 100k. Neither clone has seen a speed above 22, and the 44 features include speed. In a 6-seed quick check, the duck clone ("v2 model on rules v3") reached 102,500–103,317 and the jump clone 102,499–103,716. On rules v2 both reached 1M on the same seeds.
- **Commands** (queued for the orchestrator, after the stuck study; 2 workers; each game ends near 103k, so a few minutes):
  ```sh
  node trainer/cli.js eval --dir runs/exp-bc-duck/v2-bc-duck-d2w8-s1 --which best --game v3 --target 1000000 --seeds 150 --seed-offset 475 --workers 2 --wasm
  node trainer/cli.js eval --dir runs/v2-bc-jump-d2w32-s1 --which best --game v3 --target 1000000 --seeds 150 --seed-offset 475 --workers 2 --wasm
  ```
  `eval --game v3` warns "cross-rules evaluation: a v2 model on rules v3", and appends to the run's `evals.jsonl` with `gameVersion: 'v3'` and `modelGameVersion: 'v2'`.
- **Reading.** Report the score distribution (median, min, max) and the pass count, which is expected to be 0/150. Label it "v2 model on rules v3".

## 8. Open choices

1. **Onset.** The 150 ms bar is met from about 200k (55%), not right after 100k. `curve` 0.25 meets it from about 140k, but the speed then jumps from 19 to 38 within the first 10k points. A literal "most, right past 100k" would need a step in the cap at 100k. The user should choose.
2. **Tunnelling.** Above 42 px/frame a single small cactus can pass between two frames: 5 of 451 at 500k, none seen at 1M. It is legal (it only makes the game easier) and is excluded from must-act. Swept collision would be a physics change, so it was not made.
3. **Training for rules v3.** The default `bc-gen` levels (0, 2k, 50k, 300k, 900k) have no rows between 100k and 300k, where the speed changes fastest. A rules-v3 clone (`runs/v3-bc-…`) needs a level grid such as 100k, 120k, 150k, 200k, 300k, 500k, 1M. The `observe()` speed features are normalised by v1's cap (13) and not clipped, so at speed 60 feature 4 reaches 7.7, against at most 2.3 on rules v2. That is legal input, but a v3 student will see a much wider input range; `policy.js` stays unchanged.
4. **Above 1M** the rules stay at the 1M values (speed 60), as v2 stays at its own 1M values.
