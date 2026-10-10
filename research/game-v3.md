# Rules v3: superhuman past 100k

Status (2026-10-10): rules v3 are implemented on `wt/game-v3`. The budget probe, the solver unit tests and a 40 + 40 game solver pilot are done. The full solver survey and the assessment of the two shipped clones on rules v3 (`assess --game v3`) are pre-registered below. They are being run on branch `wt/exp-v3-survey` (from `fb1793c`); results go in §9.

Core hash after the change: `65f3546c…` with the step at 100k (`63fa2ab5…` for the superseded first proposal; before rules v3: `0f329d4e…`).

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
- `rulesAt(d, r, e)` keeps every v2 value. Only when e > 0 does it set the speed cap, to `lerp(V3.stepSpeed, V3.maxSpeed, e^V3.curve)`: a **step at 100k** (user ruling "Step at 100k", 2026-10-10), then a log-scale ramp.
- Every `version === 'v2'` branch (committed airborne intervals, the takeoff-ahead push, per-frame rules) now reads `version !== 'v1'`, so v3 inherits all of v2's fairness machinery.
- Spawned obstacles carry `late`, and `game.superLevel` exposes e.

**How the step is applied (obstacles already on the track at 100k).** The fairness model works in frames. A speed jump from 19 to 41 shrinks the frame gap between obstacles placed at the old speed by more than half, while the airtime (about 32 frames) stays the same. Two jump obstacles spaced for speed 19 would then demand overlapping jumps. So the speed is not raised under them:
1. Once e > 0 and the speed is still below `V3.stepSpeed` (`stepPending`), no new obstacle spawns.
2. The obstacles already placed (on screen or parked) run out at the old speed, under the old plan.
3. On the first frame with an empty track and the dino on the ground, the speed is set to the cap in one frame. Spawning resumes in the same update, so the next obstacle, already late-anchored, is placed at the new speed.

Results:
- The pause between e > 0 and the step is 1–48 frames (median 31; 80 games from 98k, both solvers), at most 0.8 s of empty track.
- `test/v3.test.js` checks it: 6 seeds × both solvers from 98k to 110k. The speed jumps from below 20 to at least 41, no pre-step obstacle is on the track at the step, the dino is grounded, the pause is under 120 frames, and there are no crashes.
- `startScore` games above 100k start at the cap and never pause.

**Endpoints: `V3 = { startScore: 100000, endScore: 1000000, stepSpeed: 41, maxSpeed: 60, curve: 1 }`.**

| Score | v2 speed cap | v3 speed cap |
|---|---|---|
| 100k | 19.0 | 19.0 |
| 101k | 19.0 | 41.1 |
| 110k | 19.1 | 41.8 |
| 150k | 19.6 | 44.3 |
| 200k | 19.9 | 46.7 |
| 500k | 21.1 | 54.3 |
| 1M and above | 22.0 | 60.0 |

**Targets** (from the ruling): v2 stays human-feasible (0% of budgets under 150 ms at every level). On v3, ≥ 50% of budgets are under 150 ms at 110k, rising with score to ≥ 90% at 1M. `stepSpeed` 41 is the round value that gives ≥ 50% at 110k with margin (59.9%), and already 51% at 101k.

**Budget table** (`node research/scripts/game-v3/budget.js --games 12`): 12 solver games per cell, each 2,000 points from the start level. A start level of 101k or more starts at that level's cap. The 99k row starts on v2 speed and crosses the step naturally. The jump and duck solvers give identical rows because the obstacle sequence does not depend on the player while no crash occurs. 0 crashes in every cell.

| Rules | Start level | Speed at end | Must-act | Tunnelled | Budget median | p90 | Share < 150 ms |
|---|---|---|---|---|---|---|---|
| v2 | 99k–150k | 19.0–19.6 | 1188–1216 | 0 | 386–375 ms | 439–429 ms | 0.0% |
| v2 | 200k | 19.9 | 1171 | 0 | 22.0 f / 367 ms | 25.4 f / 424 ms | 0.0% |
| v2 | 500k | 21.1 | 1127 | 0 | 20.6 f / 343 ms | 24.2 f / 404 ms | 0.0% |
| v2 | 1M | 22.0 | 1092 | 0 | 19.6 f / 327 ms | 23.1 f / 386 ms | 0.0% |
| v3 | 99k (crosses the step) | 41.1 | 893 | 0 | 21.3 f / 354 ms | 26.3 f / 439 ms | 18.7% |
| v3 | 101k | 41.2 | 557 | 0 | 8.9 f / 149 ms | 12.2 f / 203 ms | **51.2%** |
| v3 | 110k | 41.9 | 553 | 0 | 8.6 f / 144 ms | 11.8 f / 197 ms | **59.9%** |
| v3 | 150k | 44.5 | 516 | 1 | 7.8 f / 131 ms | 10.7 f / 178 ms | **73.1%** |
| v3 | 200k | 46.8 | 483 | 9 | 7.4 f / 124 ms | 10.4 f / 174 ms | **79.1%** |
| v3 | 500k | 54.3 | 434 | 6 | 6.6 f / 110 ms | 8.8 f / 147 ms | **92.6%** |
| v3 | 1M | 60.0 | 402 | 0 | 5.6 f / 93 ms | 8.4 f / 140 ms | **100.0%** |

The bar is met right after the step and tightens monotonically with score.

**Superseded first proposal** (no step; `maxSpeed` 60 with `curve` 0.5 from v2's 19, on commit b1a5529): 0% at 110k and 150k, 55.5% at 200k, 87.7% at 500k, 100% at 1M. The user ruled for the step instead.

## 4. Rules v1 and v2 unchanged (proof)

- Rules v1: the existing goldens (`test/golden/`, from baseline `2da34de`) still pass, frame for frame (`test/parity.test.js`).
- Rules v2: before `core.js` was edited, commit `417eaa8` added 7 v2 golden traces (`test/golden-v2/`, 36,000 frames). `test/record-golden-v2.js` recorded them from `core.js` and `test/solver.js` at `10cd0e0`. They cover the 1k onset, 30k (groups of 4 from 31.6k), 148k → 151k and 998k → 1M, with both solvers and random keys (crashes, restarts, fast-drops, ducks). `test/v2-parity.test.js` passes them all after the change, and `test/v3.test.js` replays the three traces that start below 100k on rules v3, frame for frame.
- Behaviour-cloning demos regenerate byte for byte: `bc-gen` with the default teacher gives sha256 `3a67bfd4e109d6bb…` (2,096,268 rows), and with `--teacher duck` gives `363872f61a785729…`, equal to `runs/exp-bc-duck/v2-demos-duck-s1-40`. `test/bc.test.js` and `test/net.test.js` pass.
- `node test/bit-identity.js <10cd0e0 worktree> <this tree> --core-changed`: IDENTICAL, 7 scenarios, 48,603 values, and again against `b4624b3`. After the step at 100k it was not re-run against `e22126a`: it refuses to start while another trainer is running. The step touches only rules v3 above 100k, which those small v2 trainings never reach. The new `--core-changed` flag reports the core hash instead of comparing it.
- **Core hash.** It changed (`0f329d4e…` → `65f3546c…`). That invalidates pre-v3 checkpoints (`--resume` needs `--force`) and pre-v3 demos (`bc-train` refuses them without `--force`). For rules v1/v2 work, `--force` is justified because the goldens, the regenerated demo hashes, `net.test.js` and bit-identity show identical games and decisions. Never reuse them for rules v3.

## 5. Solvers on rules v3

`test/solver.js`:
- **Lookahead.** The fixed `tau > 80` prune is now `tau > horizon(speed)` = max(80, longest airtime + 2). Every takeoff lies within one airtime before tau, and late anchors lie later still, so the prune is exact. Because the airtime is capped (§2), the horizon is 80 at every speed: the decisions are unchanged and it stays correct if the arc ever grows.
- **Newcomer bound.** The duck solver's `MAX_PTERO_OFFSET` now comes from `core.js` `MAX_PTERO_SPEED_OFFSET`: the largest pterodactyl offset any version draws, 1.6 on rules v2 and v3 (rules v3 does not change the spread).
- v1/v2 decisions are bit-identical (§4: demo hashes).

Results so far, all 0 crashes:
- `test/v3.test.js`: the step crossing (§3: 6 seeds × both solvers, 98k → 110k); both solvers, 3 seeds each from 100k to 400k; plus 6 seeds × starts 300k, 600k, 1M × 20k points each; duck-solver `conflicts` 0.
- Pilot survey (not the pre-registered survey; seeds 50,000,000–50,000,039, 2 workers, about 8 s each): `survey.js --game v3 --start 100000 --cap 1000000`, 40 games per solver from 100k to 1M. Every game starts at 100k on v2 speed and so takes the step. Re-run after the step ruling:

  | Solver | Crashes | Obstacles seen | Mid pterodactyls ducked |
  |---|---|---|---|
  | jump | 0/40 | 882,247 | 0 (it never ducks) |
  | duck | 0/40 | 882,247 | 127,795 of 127,981 (0 fallbacks) |

## 6. Pre-registered: full solver survey on rules v3

- **Decision.** Can rules v3 be declared solvable ("an exact solver never crashes") and serve as a training target with either teacher?
- **Hypotheses.**
  - H0: the late-anchored fairness model is exact on v3, so 0 crashes.
  - H1: a rare discretisation gap appears at v3 speeds or at the step, where `minWindow/2` gives about 1 frame of slack against the solver's ±0.5-frame rounding, and some games crash.
- **Premises (checked).** The probe shows 0 crashes in 336 short games, including the step crossing. The unit tests and the 80-game pilot show 0 crashes. Late anchors keep every canonical takeoff after visibility.
- **Arms.**
  - `--solver jump` and `--solver duck`: N = 3,000 games each, from 100k to 1M, seeds 51,000,000–51,002,999, which are disjoint from the pilot.
  - Positive control: crashes are detectable on rules v3. Both shipped clones (v2 models on rules v3) crashed at 100,036–100,358 on 6 of 6 held-out seeds, on the first obstacles after the step (offsets 1000–1005; quick `runEpisode` check); the full evaluation is §7.
- **Commands** (2 workers, so as not to disturb the scan job on this Mac; about 0.35 s per game per worker, so about 9 min per arm):
  ```sh
  mkdir -p runs/exp-game-v3
  nohup node research/scripts/prereg-teacher-duck/survey.js --solver jump --game v3 --start 100000 --cap 1000000 --base 51000000 --n 3000 --workers 2 --out runs/exp-game-v3/survey-jump-51000000-3000.jsonl > runs/exp-game-v3/survey-jump-51000000-3000.log 2>&1 &
  nohup node research/scripts/prereg-teacher-duck/survey.js --solver duck --game v3 --start 100000 --cap 1000000 --base 51000000 --n 3000 --workers 2 --out runs/exp-game-v3/survey-duck-51000000-3000.jsonl > runs/exp-game-v3/survey-duck-51000000-3000.log 2>&1 &
  ```
  The dashboard's Scans section lists both files, with game v3.
- **Reading.**
  - 0/3,000 per arm gives a 95% CP upper bound of 0.12% per game on the crash rate. Accept H0, and rules v3 is solvable to that bound for that solver.
  - Any crash means H1. The row's `onScreen` and `last30` say why. Classify it with `research/scripts/prereg-teacher-duck/crash-classify.js` (adapted for `--game`), fix the fairness model or the step (v3-only), and re-run with fresh seeds. Rules v3 must not be used as a training target before that.
  - Duck fallbacks above 1% would not block anything, but would be recorded as the newcomer bound biting at high speed.
- **Scope.** Start 100k only, at natural speed (no `startScore` past 100k). Starts above 1M are covered by the unit tests only.

## 7. Queued: the shipped clones on rules v3 (cross-rules assessment)

- **Decision.** It gives the baseline for "train even stronger models": how far v2 models get on rules v3, and where.
- **Expectation.** Both fail just past 100k. Neither clone has seen a speed above 22, and the 44 features include speed. In a 6-seed quick check (with the step), both clones ("v2 model on rules v3") crashed at the same scores, 100,036–100,358, the first obstacles after the step. On rules v2 both reached 1M on the same seeds.
- **Commands** (queued for the orchestrator, after the stuck study; 2 workers; each game ends near 100.3k, so a few minutes):
  ```sh
  node trainer/cli.js assess --dir runs/exp-bc-duck/v2-bc-duck-d2w8-s1 --which best --game v3 --target 1000000 --seeds 150 --seed-offset 475 --workers 2 --wasm
  node trainer/cli.js assess --dir runs/v2-bc-jump-d2w32-s1 --which best --game v3 --target 1000000 --seeds 150 --seed-offset 475 --workers 2 --wasm
  ```
  `assess --game v3` (formerly `eval`) warns "cross-rules assessment: a v2 model on rules v3", and appends to the run's `assess.jsonl` with `gameVersion: 'v3'` and `modelGameVersion: 'v2'`.
- **Reading.** Report the score distribution (median, min, max) and the pass count, which is expected to be 0/150. Label it "v2 model on rules v3".

## 8. Open choices

1. **Onset (decided: step at 100k).** The user ruled for a step. The cap jumps from 19 to 41 at 100k, after a pause of at most 0.8 s with no obstacles (§3).
2. **Tunnelling.** Above 42 px/frame a single small cactus can pass between two frames: 1–9 per 500 must-act obstacles from 150k to 500k, none seen at 1M. It is legal (it only makes the game easier) and is excluded from must-act. Swept collision would be a physics change, so it was not made.
3. **Training for rules v3.** The default `bc-gen` levels (0, 2k, 50k, 300k, 900k) have no rows between 100k and 300k, where the speed changes fastest. A rules-v3 clone (`runs/v3-bc-…`) needs a level grid such as 100k, 120k, 150k, 200k, 300k, 500k, 1M. The `observe()` speed features are normalised by v1's cap (13) and not clipped, so at speed 60 feature 4 reaches 7.7, against at most 2.3 on rules v2. That is legal input, but a v3 student will see a much wider input range; `policy.js` stays unchanged.
4. **Above 1M** the rules stay at the 1M values (speed 60), as v2 stays at its own 1M values.
