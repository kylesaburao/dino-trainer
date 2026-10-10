# Pre-registration: train stronger models for rules v3 (duck-teacher clones that survive the 100k step)

Status: **registered (pending orchestrator review), 2026-10-10; not run.** Author: a mac subagent, branch `wt/prereg-v3-clones` from main `63dcfa1` (core hash `65f3546c…`, rules v3 in `core.js`). Nothing here has been run. The numbers come from the repo (cited) or from closed-form integrals of the speed caps in `core.js` `rulesAt` (§9). Those integrals reproduce the survey's measured frame count to 0.002 %.

Context: AGENTS.md ("Research method", the naming scheme, "Rules v3", the shape cap, the dashboard invariant, `assess`), `research/game-v3.md` (§3, §8 item 3, §9), `research/prereg-teacher-duck-stuck.md` (screens and failure classes), `research/prereg-distill.md` §11a/§11b (clean count out of 10, tier-B cap, Stage 0).

**User's goal:** models that play rules v3 to 1M. Rules v3 are superhuman past 100k: reaction budgets under 150 ms for most must-act obstacles.

## 1. Decision
Can behaviour cloning of `teacher-duck` (`createDuckSolver`) on rules-v3 demonstrations produce a model that plays **rules v3 from 0 to 1M** reliably, using only the 44 unchanged `observe()` features? If so, at which of the registered shapes, and how often per training seed?

What the answer decides:
- whether a rules-v3 model (`runs/exp-v3-clones/v3-bc-duck-…`) can be put to the user as a candidate. Certification on fresh held-out offsets is a separate registration; so are shipping and any `?game=v3` default;
- which shape the v3 line continues with: small (1×6, cheap in CSS) or the teacher-size 2×32;
- whether the v3 pipeline needs more than the default procedure with a new level grid. For example: more demos near the step, DAgger, or a fix for the opening release modes. Each of these would be its own registration.

It certifies nothing and ships nothing. It never writes the root `model.js`/`model.css`.

## 2. What is known (premises)
- **P1, rules v3 is solvable by the teacher.** Duck solver: 0/3,000 crashes from 100k (with the step) to 1M, 0 fallbacks, 0 conflicts, every mid pterodactyl ducked (`game-v3.md` §9.1; `runs/exp-game-v3/survey-duck-51000000-3000.summary.json`). Two-sided 95 % CP upper bound 0.123 % per game.
- **P2, no v2 model survives the step.** Both shipped clones, as v2 models on rules v3, scored 0/150 on held-out offsets 475–624. Every game crashed between 100,016 and 101,555, and none below 100k (`game-v3.md` §9.2). **Offsets 475–624 are spent; the next fresh offset is 625.** This study spends none.
- **P3, the input range changes.** Feature 4 is `(speed − 6) / 7` (`policy.js` `observe`). On rules v2 it lies in [0, 2.29] (speed 6–22). On rules v3 it jumps from 1.86 to **5.0** at the step (speed 19 → 41) and rises to **7.71** at speed 60. The per-obstacle speed feature `(speed + offset) / 13` goes from at most about 1.8 to about 4.7. These are legal inputs and `policy.js` stays unchanged. A v2 model has never seen x4 > 2.29, which explains P2. In play, x4 never takes values in (1.86, 5.0) except on the single frame of the step, so no demonstrations are needed there.
- **P4, the speed is log-linear in score past 100k.** The cap is 41 + 19·log10(S/100k) (`V3`: `stepSpeed` 41, `maxSpeed` 60, `curve` 1). So x4 is linear in log S, and **levels spaced evenly in log score are spaced evenly in x4**.
- **P5, the step is demonstrated only from a start below 100k.** A `startScore` game at or above 100,001 starts at that score's cap and never pauses (`game-v3.md` §3). A start at 100,000 has e = 0, but it begins with an empty track, so it shows the step without obstacles placed at the old speed. **Only a start below 100k crosses the step the way a full game does:** obstacles placed at speed 19 run out, spawning pauses for 1–48 frames, then the speed jumps. In the budget probe, the 99k row is the one that crosses the step naturally.
- **P6, rules v3 equals rules v2 below 100k** (proven by `test/v2-parity.test.js` and `test/v3.test.js`). So the opening scan (0 → 2,000) plays identical games on either rules. On the v2 grid, the v2 clones learned 0 → 100k with no failure below 100k in 300 v3 games (P2).
- **P7, the opening release modes are a property of the procedure** (`prereg-teacher-duck-stuck.md` Results). On rules v2 with the default grid, clean seeds were: d2w32 4/20 (14/20 early release, 2/20 stuck), d2w8 8/20 (9/20 stuck) and 5/10 on seeds 1–10. Distillation Stage 0 (branch `wt/exp-distill2`, not yet on main): 1×6 10/10 clean and 1×4 10/10. 2×6 was 4/10, 2×4 5/10 and direct 2×8 5/10. Min margins of the tier-B 1×6 models: 1.56e-4 (s1) and **8.6e-5 (s2)**, close to the 5.8e-5 CSS-risk flag. 1×4: 3.3e-4 and 1.6e-4.
- **P8, frame counts on rules v3** (§9, integrals of the caps; the survey's 2,023,074,000 frames / 3,000 games = 674,358 per game from 100k to 1M, against an integral of 674,344):

  | Segment | Rules v2 | Rules v3 |
  |---|---|---|
  | 0 → 100k | ≈ 229k frames | the same (P6) |
  | 100k → 1M | 1,714,146 | **674,344** (+ the 1–48-frame pause) |
  | **Full game 0 → 1M** | **1,943,237** (measured, `runs/exp-bc-sweep/scans/d1-w32-s1.jsonl`) | **≈ 903,500 (0.465×)** |
  | 95k → 130k (post-step screen) | n/a | ≈ 10,545 (95k → 100k, speed 19) + 28,490 (100k → 130k) ≈ **39,000** |
  | 5,000 points from a level | ≈ 10,500 at speed 19 | 4,854 at 100k (speed 41), 4,272 at 200k, 3,682 at 500k, 3,381 at 900k |

  So **a full rules-v3 game is under half a rules-v2 game in frames** (about 1.04M frames fewer). Past the step, one must-act obstacle comes at most about every 34 frames (`game-v3.md` §2): about 18k must-act obstacles per game from 100k to 1M (from the budget table's 33–46 per 2,000 points; 65.6M obstacles of all types passed / 3,000 survey games ≈ 21.9k).

**Shape cap** (user order): at most 16 layers and at most 32 units per layer. Every arm below satisfies it.

## 3. Hypotheses and predictions
Per arm, the outcome is the **clean count out of 10** (§6). The modes are read from where the failures are.
- **H-learnable:** the 44 features plus the v3 grid suffice, and BC learns rules v3. Predicts at least one arm with clean ≥ 9/10, failures (if any) in the opening only, as on rules v2.
- **H-step:** the step is the weak point: covariate shift at the speed jump, the pause, the first late-anchored obstacle. Predicts post-step-screen failures clustered within the first ~2k points after 100k, while tier A and tier B failures past 130k are rare.
- **H-late:** the high-speed end (5–9-frame budgets, speed ≥ 50) is the weak point. Predicts post-step 1,000/1,000 but tier A or tier B failures above 130k, rising with score.
- **H-capacity:** 1×6 cannot fit the wider input range together with the release decision. Predicts 1×6 clean count well below 2×32's (McNemar, §7), with failures past 100k. On rules v2, 1×6 was the cleanest shape (P7), so this would be a v3-specific capacity limit.
- **H-opening-persists:** the v2 opening modes carry over at their v2 rates. The extra v3 rows change the balance of the loss, not the opening states. Predicts opening failures at 2×32 in about 14/20 seeds (early release), at 1×6 about 0/10, and no new opening mode.
- **H-dilution:** the v3 rows (about 40 % more rows, §4.1) dilute the opening's release rows and make the opening modes **more** frequent than on rules v2. Read against the v2 rates in P7 (descriptive: different seeds and demos).

## 4. Design
**Run dirs** (naming scheme): group `runs/exp-v3-clones/`, one level of nesting, so the dashboard lists every run.
- demos `v3-demos-duck-s1-40` (training) and `v3-demos-duck-s41-60` (held-out agreement);
- models `v3-bc-duck-d<D>w<W>-s<k>`: `v3-bc-duck-d1w6-s1` … `-s10`, `v3-bc-duck-d2w32-s1` … `-s10`, and `v3-bc-duck-d2w8-s<k>` if arm C is ruled in;
- screens in `opening/`, `poststep/`, `scans/`, `duckshare/`, `margin/` and `agree/` under it.

**Environment.** Mac only, one heavy job at a time. It starts **right after distillation D1 finishes** (about 1 h; the orchestrator sequences it). It runs from main at the tooling commit (§4.0), which has core hash `65f3546c…`. Settings: `PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`, `--backend python --snapshots --wasm`, 15 epochs, batch 1,024, lr 2e-3, default class weights, final-epoch params (no epoch selection). The hyperparameters are fixed to the v2 procedure on purpose, so the only change is the rules and the grid. More rows mean about 40 % more Adam steps at the same epoch count; this is noted as part of the procedure, not adjusted. Never `session`, `finish`, `--auto-export`, `export` to the root, or `assess`.

### 4.0 Tooling step (before any run; one commit, with controls)
The screens' tools are rules-v2 only today:
- `scan.js` pins `EXPECT_HASH = '0f329d4e'`, so on current main it refuses to start, because the hash is now `65f3546c…`. It also throws on `gameVersion !== 'v2'` and has no start option.
- `opening.js` and `duckshare.js` hard-code `version: 'v2'` and reject non-v2 models.
- `margin.js` already plays the model's own rules, but its fixed levels (0, 2k, 50k, 300k, 900k) leave 100k–300k unprobed and none of them crosses the step.

Already fine: `bc-gen --game v3`, `bc-train` (tier A plays the demos' rules), `survey.js --game --start --cap` and `assess --game`.

Changes (each off or unchanged by default):
1. **`scan.js`:**
   - Plays the model's `gameVersion` (v2 or v3).
   - Accepts the current core hash: the expected prefix becomes `--expect-hash <prefix>`, default the current `coreHash()`; the v2-era studies pass `0f329d4e`.
   - Adds **`--start S`**: every episode starts at score S (`createGame` `startScore`) and ends at `--target`, through the worker's existing per-job `starts`/`caps`. `caps` are absolute scores, as for the curriculum.
   - The header records `game` and `start`.
2. **`opening.js`, `duckshare.js`:** play the model's `gameVersion` (the teacher: `--game`, default v2), with `--game` as an explicit cross-rules override (warned and recorded, as `assess`). `opening.js` also gets **`--start S`**.
3. **`margin.js`:** **`--levels a,b,…`**, defaulting to the current list.
4. **Agreement split by regime:** the held-out agreement tool reports overall weighted agreement **and** the two subsets `x4 ≤ 2.5` (speed ≤ 23.5: rules v2 and the v3 part below the step) and `x4 > 2.5` (speed > 23.5: only past the step on rules v3). It is either `agreement.py` with a `--split-x4 2.5` flag, or `research/scripts/prereg-distill/agree.js` once `wt/exp-distill2` is merged. The executor names the one used.
5. **Dashboard (invariant):** `trainer/scans.js` and the Scans section show a scan header's `game` and `start` (survey rows already show `game`). `test/dashboard-check.js` gains a fixture for a `scan.js --start` file on rules v3.

**Controls for the tooling commit (all must pass before any arm is read):**

| Control | Must show | If not |
|---|---|---|
| `node --test test/` and `node test/dashboard-check.js` | all pass | STOP |
| Defaults unchanged, opening and duck share | `opening.js` on the committed `runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json` (base 30,003,000, n 1,000, target 2,000) gives output identical to `runs/exp-bc-duck-stuck/opening/ctl-d2w8-s1.json`, apart from time fields; `--teacher duck` gives `ctl-teacher.json`. `duckshare.js --teacher duck --n 2` gives the same counts as before the change | fix the tool |
| Defaults unchanged, scan | `scan.js --expect-hash 65f3546c` on a v2 model, `--n 20 --target 1000000`, equals the same 20 seeds of the step-2 tier B (`--n 20` is seconds) | fix |
| `--start` plumbing | `scan.js --start 2000 --target 3000` on a v2 model, `--n 5`: every row's game starts at 2,000 (checked by replaying one seed with `runEpisode(model, seed, 3000, 2000)`, same score and frames). On rules v3 the plumbing is checked by the detection control DC-post (§4.4) | fix |
| `margin.js` with default levels | identical to its previous output on one committed model | fix |

The tooling commit changes no `core.js`, `policy.js`, `test/solver.js` or training code, so the core hash stays `65f3546c…`.

### 4.1 Demonstrations (gate): the level grid
`bc-gen --teacher duck --game v3 --seeds 1-40 --span0 3000 --span 5000 --keep-boring 10 --levels <grid> --dir runs/exp-v3-clones/v3-demos-duck-s1-40`, and the same with `--seeds 41-60` into `v3-demos-duck-s41-60`.

**Grid (15 levels):**
`0, 2000, 50000, 98000, 101000, 127000, 162000, 207000, 264000, 336000, 428000, 546000, 695000, 886000, 995000`

| Part | Levels | Speeds covered (x4) | Why |
|---|---|---|---|
| Rules v2 part | 0 (span0 3,000), 2,000, 50,000 | 6 → 14 (0 → 1.1), 13.9 (1.13), 18.1 (1.73) | The v2 defaults below 100k, unchanged. On this grid the v2 clones never failed below 100k (P6) |
| **The step** | **98,000** (98k → 103k) | 19.0 (1.86), then the pause, then 41.0–41.4 (5.0–5.06) | The only start that crosses the step as a full game does (P5): about 4.2k frames at speed 19 with obstacles placed at the old speed, the pause, the jump, then about 2.9k frames at speed 41. It replaces v2's 300k level (speed 19.9), which on rules v3 lies past the step |
| Rules v3 part | 101,000, then 127k, 162k, 207k, 264k, 336k, 428k, 546k, 695k, 886k, 995k | 41.1, then **43, 45, 47, 49, 51, 53, 55, 57, 59, 59.96** (x4 5.01, then 5.29 … 7.71) | 101k gives a full span at the first post-step speed, already at the cap (no pause). Above it, the levels are evenly spaced in log score, hence in x4 (P4): one level every 2 px/frame of speed, S = 100k · 10^((v − 41)/19), rounded to 1k. 995k ends at 1M, the target |

- **Density justification.** On rules v2, the default grid's largest gap in x4 above 50k is 0.28 (50k → 300k: 1.73 → 1.99; 300k → 900k: 1.99 → 2.27). Clones on that grid interpolated across it to 1M without failure (tier B 0/1,000 on clean seeds). Here the v3 range is 2.71 wide (5.0 → 7.71). A 2-px/frame spacing gives gaps of **0.286** in x4, the same density as v2's grid, so 10 intervals over the v3 range. `game-v3.md` §8 item 3 suggested 100k, 120k, 150k, 200k, 300k, 500k, 1M. That list has gaps up to 0.85 in x4 (500k → 1M: speed 54.3 → 60), three times v2's density, and no start that crosses the step. This grid refines it.
- **Rows.** Not predicted exactly; `demos.json` records them. Estimate per seed: about 3 × 10–12k (the v2 part), about 7k (the 98k level) and 11 × 3.4–4.9k (the v3 levels), so **about 3.0–3.6M rows** for seeds 1–40, against 2,062,427 for the v2 duck demos. That is about 45–75 % more, mostly at high speed. At v3 speeds an obstacle is visible for only about 10–14 frames, so "boring" frames (none visible) are a larger share past the step. The boring rule and its weights are unchanged.
- **Gate (STOP on any failure):** manifest `gameVersion` v3, `teacherName` duck, `coreHash` `65f3546c…`, **0 solver crashes** over the 600 + 300 episodes (P1 predicts 0). Every episode at 98,000 shows the step: its rows include speed ≥ 41, which can be checked from x4 > 4.9 in that episode's rows. Record the rows per level and the label counts (none / jump / duck) per regime (x4 ≤ 2.5 and x4 > 2.5).

### 4.2 Arms (seeds 1–10 each)
| Arm | Shape | `--hidden` | Params | MACs/decision | Why |
|---|---|---|---|---|---|
| **A** | 1×6 (44-6-3) | `6` | 291 | 282 | The cleanest small shape on rules v2 (P7, 10/10); the cheap CSS candidate |
| **B** | 2×32 (44-32-32-3) | `32,32` | 2,595 | 2,528 | The teacher-size shape: the most capacity under the cap (on width) at 2 layers. Its v2 early-release rate is high (P7) |
| C (optional, orchestrator's ruling) | 2×8 (44-8-8-3) | `8,8` | 459 | 440 | The shipped v2 shape. Pairs the v3 result with the shipped architecture. 1×8 (395 params, 376 MACs) is the alternative if a one-layer midpoint is preferred |

`bc-train --backend python --hidden <h> --seed <k> --demos runs/exp-v3-clones/v3-demos-duck-s1-40 --snapshots --wasm --dir runs/exp-v3-clones/v3-bc-duck-d<D>w<W>-s<k>` (tier A runs inside it). Order: A s1–s10, B s1–s10, (C s1–s10). Each run is independent.

### 4.3 Screens (per model, in this order)
All seed sets are disjoint from the demo seeds (1–60) and from the held-out `EVAL_SEEDS`.
1. **Tier A (rules v3, built in):** the 25 `validationSeeds(25)` played as full rules-v3 games to 1M (the demos' rules, automatic). Recorded: count and min.
2. **Opening scan, the stuck-duck screen (rules v3):** `opening.js --model <best.json> --base 30003000 --n 1000 --target 2000`. These are rules-v3 games, identical to rules v2 below 100k (P6). The stuck-duck study's classes, verbatim: **stuck** (the final DOWN run contains a mid overlap and lasts ≥ 20 frames), **early release** (killed by a mid pterodactyl while not holding DOWN), and **other** (with the killer).
3. **Post-step screen (new):** `scan.js --model <best.json> --start 95000 --target 130000 --base 30003000 --n 1000 --wasm`. These are the 1,000 tier-B seeds, rules v3, from 95k (speed 19, e = 0) through the natural step at 100k to 130k (speed 43.2): about 39k frames per game, about 700 must-act obstacles past the step (§2 P8; the budget table's 553–557 must-act per 12 games × 2,000 points is about 46 per 2,000 points), including the first post-step obstacles that killed every v2 model. **Every model must pass 1,000/1,000.** Failures are classed by re-running `opening.js --start 95000 --target 130000` on that model (single-threaded, about 25–65 s), which records killer, score and final DOWN run.
   - Unlike the opening scan, these games are **not prefixes of the tier-B games** (`startScore` changes the trajectory), so post-step failures are not a subset of tier-B failures. The two screens are complementary: post-step concentrates 1,000 independent crossings of the step, and tier B crosses it once per game.
   - The 130k end covers the step, the 101k and 127k levels and about 3 s of play at each speed in 41–43. Its purpose is the step, not the late game (tier A and tier B cover that).
4. **Tier B (rules v3):** `scan.js --model <best.json> --base 30003000 --n 1000 --target 1000000 --wasm`, full rules-v3 games from 0. Only for models that pass tier A 25/25, the opening 1,000/1,000 and the post-step 1,000/1,000. **At most 2 per arm, in seed order** (§11b convention). Contiguity check: 1,000 rows from 30,003,000 and a footer. Every failure gets `opening.js --start <nearest level below> …` classification. If `diag.js` is not v3-ready, the hand review is limited to the killer and the final DOWN run.

### 4.4 Positive controls and detection control (before any arm is read)
| Control | Command | Must show | If not |
|---|---|---|---|
| **PC-open:** the teacher on the opening | `opening.js --teacher duck --game v3 --base 30003000 --n 1000 --target 2000` | 1,000/1,000, 0 fatal, max lag ≤ 2 (as the stuck study's control 1, which gave max lag 1) | the tool is wrong; fix it before reading |
| **PC-post:** the teacher on the post-step screen | `survey.js --solver duck --game v3 --start 95000 --cap 130000 --base 30003000 --n 1000`, and `opening.js --teacher duck --game v3 --start 95000 --target 130000` (lag) | 0/1,000 crashes; max lag ≤ 2 at v3 speeds | a crash: the screen or the rules are wrong: STOP. Lag > 2: fix the lag definition for v3 speeds, record the fix, then read |
| **PC-B:** the teacher on tier B | `survey.js --solver duck --game v3 --start 0 --cap 1000000 --base 30003000 --n 1000` | 0/1,000 (the survey's 0/3,000 used other seeds, 51M+) | STOP |
| **DC-post (detection):** the shipped v2 model, a v2 model on rules v3 | `scan.js --model runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json --game v3 --start 95000 --target 130000 …` | ≥ 990/1,000 failures, none below 100,000 and ≥ 95 % of them in 100,000–102,000 (P2: 150/150 in 100,016–101,555) | the screen does not reach the step as a full game does: fix `--start` before reading |
| **C-agree:** the teacher's labels against themselves | agreement tool on `v3-demos-duck-s41-60` with the solver's own labels | 1.000 | fix the tool |

### 4.5 Measures (per model)
1. **Duck share in play (rules v3):** `duckshare.js --model <m> --base 30003000 --n 10` (full v3 games), for every model with tier A ≥ 20/25, split before and after 100k where the tool allows. Teacher reference: 1.000 (P1, 9,591,107 of 9,591,107 past 100k). Bars as distill §7: **ducks** ≥ 0.9× the teacher's; **rarely ducks** < 0.5×.
2. **Min top-2 margin (rules v3):** `margin.js --model <m> --levels 0,2000,50000,98000,101000,207000,546000,886000` (120k decision frames per level, 960k in all). Recorded: min (with level and score), 1st percentile, median. Required for the ≤ 2 tier-B models per arm. Flag: < 5.8e-5 (the ES model's value; AGENTS.md).
3. **Agreement with the solver:** weighted held-out agreement on `v3-demos-duck-s41-60`, final params, overall and split at x4 = 2.5 (§4.0 item 4); `recallDuck`, `duckPredShare`. Per epoch snapshot for the tier-B models (descriptive: when does the v3 regime converge?).
4. **CSS cost note:** MACs per decision (282 / 2,528 / 440), `model.css` bytes after a scratch export, and the per-decision Chrome cost scaled from `bc-sweep-report.md` §2 (0.2 ms at d2w8, 1.0 ms at d2w32, roughly proportional to MACs: 1×6 ≈ 0.13 ms). Rules v3 do not change the cost per decision; the page still makes one decision per frame. CSS agreement on rules v3 (`test/css-agreement.js --start-score 98000`, which reaches the step) is **not** run here. It is a precondition of any ship step and would be registered with the certification.
5. **Per-seed record:** params sha256, final train loss, recall (none/jump/duck), tier A, opening (passes, stuck, early, other, max lag), post-step (passes, failure scores and classes), tier B, class.

## 5. Classification (fixed now)
- **Clean seed:** tier A 25/25 **and** opening 1,000/1,000 **and** post-step 1,000/1,000 **and** tier B 0/1,000 where tier B ran. Seeds beyond the tier-B cap are "clean (A + opening + post-step)", which has no power against a late-game mode past 130k beyond tier A's. The asymmetry is reported.
- **Failure class of a seed:** **opening-stuck**, **opening-early** or **opening-other** (from the opening scan), **step** (post-step or tier-B failures within 100,000–102,000), **post-step-other** (post-step failures in 102,000–130,000), **late** (failures above 130,000, from tier A or tier B), and **sub-100k-other** (any failure in 2,000–100,000). A seed can have several; all are reported.
- **Reliable arm:** clean ≥ 9/10 (distill §11b). **Candidate:** a clean model with tier B 0/1,000, duck share ≥ 0.9×, and min margin ≥ 5.8e-5.

## 6. Power
- **Per model:** 0/1,000 (post-step or tier B) bounds the per-game failure rate at **0.299 %** (one-sided 95 %), 0.368 % two-sided. 25/25 in tier A bounds it at 11.3 %. A seed with a per-game failure rate q fails 1,000 games with probability 1 − (1 − q)^1000: 99.3 % at q = 0.5 %, 63 % at 0.1 %. So the post-step screen detects any step failure mode of ≥ 0.5 % per crossing almost surely; tier A alone (25 crossings) detects a 5 % mode only 72 % of the time.
- **Per arm, the clean count of 10** (Clopper–Pearson 95 %, two-sided): 10/10 → 69.2–100 % (one-sided lower 74.1 %); 9/10 → 55.5–99.7 %; 5/10 → 18.7–81.3 %; 1/10 → 0.3–44.5 %; 0/10 → 0–30.8 % (one-sided upper 25.9 %).
- **Arm against arm, seed for seed** (seed k of A against seed k of B: same seed index, not the same initialisation, so the pairing is by index only; exact McNemar, one-sided). The smallest significant result is b = 5, c = 0 (p = 0.031). Power at 10 pairs (independent seeds, per-seed clean probabilities p_A, p_B): 1.0 vs 0.5 → 0.62; 0.9 vs 0.5 → 0.35; 1.0 vs 0.3 → 0.95; 0.9 vs 0.2 → 0.89. So **only large shape effects are detectable**, and a null excludes only those.
- **Against the v2 rates (H-opening-persists, H-dilution):** descriptive only, because the demos, seeds and rules differ. A Fisher exact test of the opening-failure seed counts against P7 (2×32: 16/20 seeds with an opening failure; 1×6: 0/10) is reported, labelled descriptive.

## 7. Readings (fixed in advance)
| Result | Reading | Next (each registered separately) |
|---|---|---|
| Any §4.0 or §4.4 control fails, or the §4.1 gate fails | the tools, the screen or the demos are wrong | STOP; fix and re-run that control; read nothing |
| **An arm clean ≥ 9/10** | **H-learnable at that shape:** BC with the v3 grid learns rules v3 to 1M on legal inputs | Its tier-B models are candidates if §5's bars hold. The next registration is a certification on rules v3 on fresh offsets (from **625**, or past whatever the v2 1×6 certification spends), against the baseline that v2 models score 0/150 (P2); then the user decides |
| An arm clean 3–8/10 | learnable but seed-dependent (a lottery, as d2w8 on rules v2). The failure classes say where | Its clean models (with tier B) are candidates; the class with the most seeds names the fix registration |
| Every arm clean ≤ 2/10 | the default procedure does not learn rules v3 at these shapes | No candidate. The dominant class names the fix |
| Failures mostly **step** (≥ half the failing seeds, at 100,000–102,000) | **H-step:** the step is the weak point, as for v2 models, despite the 98k level | Fix registration: more step demos (more seeds or extra demos at 95k–100k via `--extra-demos`), or DAgger at the step (`bc-gen --student --game v3 --levels 95000`) |
| Failures mostly **late** (tier A/B above 130k, post-step clean) | **H-late:** the high-speed end is the weak point | Fix registration: a denser grid above 500k or more seeds there; check the margin by level |
| Failures mostly **opening-stuck / opening-early** at rates like P7 | **H-opening-persists:** rules v3 adds no new failure mode; the v2 release problem is the bottleneck | Ties to the release-fix registration (`prereg-teacher-duck-release-fix.md`). The v3 part is not the problem |
| Opening failures at 1×6 (0/10 on v2), or clearly more at 2×32 than 16/20 | **H-dilution** (descriptive): the v3 rows weaken the opening | Fix registration: re-weight the opening or the release rows (`--transition-weight`) |
| A ≤ 2/10 and B ≥ 7/10, or McNemar B > A with p ≤ 0.05 | **H-capacity:** 1×6 is too small for rules v3 | The v3 line continues at the larger shape; arm C (if run) brackets it |
| A ≥ B in clean count (no significant difference) | no capacity effect detectable at this power (§6) | Prefer the smaller shape among candidates (CSS cost) |
| A clean model whose duck share < 0.5× | it found a jump-only plan on rules v3 (possible: the jump teacher also solves rules v3, P1) | Not a candidate in the duck generation; report |
| A candidate with min margin < 5.8e-5 | CSS-divergence risk flag | CSS agreement on rules v3 is mandatory before any ship reading |
| sub-100k-other failures (2k–100k) | a regression in the v2 part (P6 predicted none) | Report with killers; check the grid's v2 part |
| Agreement in the x4 > 2.5 regime well below the x4 ≤ 2.5 regime (descriptive) | the v3 regime is harder to imitate on these features | Report; informs a grid or seed change |

## 8. What would reopen a null
Every reading is scoped to:
- `createDuckSolver` as on main `63dcfa1` and core hash `65f3546c…`;
- the 15-level grid, span0 3,000, span 5,000, keep-boring 10, seeds 1–40;
- the python backend on the Mac with `OMP_NUM_THREADS=2`, 15 epochs, batch 1,024, lr 2e-3, default class weights, final-epoch params;
- these shapes and seeds 1–10;
- the post-step window 95k → 130k.

A null reopens with a registered change to any of these: a denser or extra-seeded grid, DAgger at the step, release-row weighting, epoch selection, a larger shape within the cap, or more seeds. "Not learnable at shape X" is never read from one arm's 10 seeds alone (§6).

## 9. Cost (Mac only, one heavy job at a time)
Frame counts from P8. Speeds from the survey (3,000 v3 games of 674k frames in 119 s on 18 workers) and from the v2 runs (step 2 and the stuck study §9; Stage 0, `prereg-distill.md` §11b(e)). Every figure scales v2 timings by frames.

| Item | Count | Each | Total |
|---|---|---|---|
| Tooling step (§4.0) and its controls | 1 | writing + about 2 min of controls | (before; not Mac-heavy) |
| `bc-gen` v3, training and held-out demos | 2 | seconds to about 1 min (the duck solver over about 4.5M frames) | ≤ 2 min |
| Positive controls (PC-open, PC-post, PC-B, DC-post, C-agree) | 5 | PC-B: 1,000 v3 games ≈ 0.9G frames, about 40 s on 18 workers; the others seconds | ≈ 2 min |
| `bc-train` + tier A, 1×6 | 10 | ≈ 15–20 s training (1.5–1.75× the rows of v2's ≈ 11 s) + ≈ 2 s tier A (25 v3 games = 22.6M frames, 0.47× v2's) | ≈ 4 min |
| `bc-train` + tier A, 2×32 | 10 | ≈ 22–27 s training (v2: 14.3 s) + ≈ 2–3 s tier A | ≈ 5 min |
| Opening scans | 20 | 5–13 s (8M frames; as on v2) | ≈ 3 min |
| Post-step screens (`scan.js --start`, 18 workers) | 20 | 39M frames ≈ 2 % of a v2 tier B: ≈ 3–6 s | ≈ 2 min |
| Post-step failure classification (`opening.js --start`, single-threaded) | only for models with failures | 25–65 s | ≤ 20 min (worst case: every model fails) |
| Tier B (v3), ≤ 2 per arm | ≤ 4 | 0.465× v2's: 1×6 ≈ 40–55 s, 2×32 ≈ 82–137 s | ≤ 7 min |
| Duck share, margin, agreement | ≤ 20 | ≈ 5–15 s, ≈ 5–10 s, seconds | ≈ 8 min |
| **Total, arms A and B** | | | **≈ 35–55 min** |
| Arm C (2×8), if ruled in | 10 + screens | ≈ 20 s + screens | **+ ≈ 12–18 min** |

The desktops are not used.

## 10. Scope, and what this does not do
- Rules v3 only, full games from 0 (tier A, tier B), the post-step window and the opening; the 44 unchanged features; `teacher-duck`; the python backend.
- **Tier-B and post-step seeds are not held out. Nothing here is a certification.** No `assess`, no held-out offsets: **625** stays the next fresh offset.
- Never writes the root `model.js`/`model.css`. Never `session`, `finish`, `--auto-export` or `export` to the root.
- `runs/exp-*/` is gitignored. Commit with `git add -f`: `best.json` and `bc-metrics.jsonl` of every run, both `demos.json`, and the JSON summaries of opening, post-step, duck share, margin and agreement. Scans, logs, snapshots and `demos.bin` stay local; the dashboard reads them live.
- **Execution order:**
  1. The orchestrator confirms that D1 has finished and the Mac is idle.
  2. Tooling commit (§4.0), with its controls.
  3. §4.1 demos (gate), then §4.4 controls.
  4. Arms in §4.2 order, each with tier A, the opening and the post-step screen.
  5. Tier B on the eligible models (≤ 2 per arm), then the measures.
  6. Results are appended here with the code commit, the core hash and the demo sha256 values. The orchestrator merges.

## 11. Open choices for the orchestrator
- **O1, arm C:** include 2×8 (the shipped shape; default proposal), 1×8, or neither.
- **O2, the grid:** the 15 levels of §4.1 (2-px/frame spacing). The alternative is `game-v3.md` §8's 7-level suggestion plus 98k, which is coarser (x4 gaps up to 0.85).
- **O3, the post-step window end** (130k) and start (95k). A start at 98k would shorten the pre-step part from about 10.5k frames to about 4.2k at speed 19. That saves little (the screen costs seconds), and 95k leaves more time for the start's effects on the obstacle stream to wash out before the step.
- **O4, the post-step tool:** `scan.js --start` (multi-threaded, shown on the dashboard), with `opening.js --start` only for classification (proposed). The alternative is `opening.js --start` alone (single-threaded, about 25–65 s per model, not a dashboard scan).
- **O5, `scan.js` hash handling:** an `--expect-hash` flag defaulting to the current `coreHash()` (proposed), against a hard-coded new prefix.
- **O6, the margin levels** (§4.5 item 2).
- **O7:** whether to also record agreement per snapshot for every model (cheap), not only for the tier-B ones.
