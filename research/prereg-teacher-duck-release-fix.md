# Pre-registration: `teacher-duck` release fix. Which training change makes duck clones reliably clean at d2w8 and d2w32?

Status: **registered (pending orchestrator review), 2026-10-10.** Written on main `319a51f` (core hash `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e`). Nothing here has run. Every number below is either read from the repo (cited) or derived from cited numbers; derived numbers are marked as such, and the ones that matter are registered as checks (§4.2).
Continues `research/prereg-teacher-duck-stuck.md` (its Results, cited here as "stuck Results") and `research/prereg-teacher-duck-step2.md` ("step 2"). Context: AGENTS.md "Research method", "Naming scheme", the shape cap and the dashboard invariant.

**User's choice (via the orchestrator):** "pre-register a fix". Target: the duck-release decision under the first mid pterodactyls, at both shapes, d2w8 (44-8-8-3) and d2w32 (44-32-32-3). The opening scan (`research/scripts/prereg-teacher-duck-stuck/opening.js`) becomes a required screen for every new duck clone (proposed AGENTS.md wording in §11).

## 1. Decision
**Which training change makes duck clones reliably clean across training seeds at d2w8 and d2w32?** "Clean" is the stuck study's class: tier A 25/25, opening scan 1,000/1,000 (no stuck duck, no early release, no other death) and tier B 0/1,000 where scanned (§4.6).

The answer decides:
- **the duck-generation BC procedure.** If an arm meets the bar (§7), its flag becomes the documented procedure for duck clones (an AGENTS.md change, approved separately). Defaults never change.
- **whether d2w32 is usable for the duck generation.** Under the current procedure only 4/20 fresh d2w32 seeds were clean.
- **whether a duck retrain (rules v3, new demos, a teacher change) can rely on one training seed plus the screen, or must train several seeds and keep the clean ones.** If nothing meets the bar, multi-seed training plus the mandatory screen (§11) is the procedure, and the fix search continues from the reading of §7.

It ships nothing and certifies nothing. The shipped `v2-bc-duck-d2w8-s1` is unaffected whatever this finds (opening 1,000/1,000, tier B 0/1,000, certification 150/150; stuck Results "§7 readings applied").

## 2. What is known, and the premises
**Two failure modes, one decision** (stuck Results, "Per seed" and "Surprises"):
- **Stuck duck:** 1,013 failures in 11 of 40 fresh seeds. The final DOWN run contains a mid pterodactyl and lasts 57–136 frames; the dino holds DOWN into the next cactus or low pterodactyl. Crash scores 479–767.
- **Early release:** 573 failures in 18 of 40 seeds. 5–8 frames of DOWN, then a release while the mid pterodactyl still overlaps the standing box; death one frame later. Crash scores 467–623 (12 reviewed by hand).
- Both are the release of DOWN under the first mid pterodactyls: too late or too early. No failure of any kind occurred above 767 points in the 7 tier-B scans of the stuck study, and step 2's only failing scan (`d2w32-s2`, 204/1,000) also failed only at 479–767. So the decision this registration targets lives in the opening.

**The control outcomes on training seeds 3–22** (stuck Results table; current procedure: `bc-gen --teacher duck` defaults, python backend, 15 epochs, batch 1,024, lr 2e-3, default class weights, final-epoch params):

| Shape | Clean | Stuck | Early release | Clean seeds |
|---|---|---|---|---|
| d2w32 | **4/20** | 2 (s15, s16) | 14 (s3, s4, s6, s7, s9, s10, s12, s14, s17–s22) | s5, s8, s11, s13 (tier B 0/1,000 each) |
| d2w8 | **8/20** | 9 (s7, s8, s10, s12, s13, s14, s17, s20, s22) | 3 (s3, s9, s16) | s4, s5, s6 (tier B 0/1,000); s11, s15, s18, s19, s21 (tier A + opening only) |

d2w8-s13 also had one jump-type death (192 points); it is counted stuck. Every seed that failed anywhere failed the opening scan; every seed that passed tier A and the opening and was scanned (7) scored 0/1,000 in tier B.

**Premises, each with its source or its check:**
- **P1, the failures are confined to the opening** (above). Supports scanning a capped subset in tier B (§4.5).
- **P2, python training is deterministic per seed on the Mac.** The stuck study's positive control reproduced `d2w32-s2`'s params sha256 `04207436…` exactly (stuck Results, "Gates and controls"). Each control seed's 12-hex params prefix is in the stuck Results table, and the full params are committed in `runs/exp-bc-duck-stuck/v2-bc-duck-<shape>-s<k>/best.json`. **The control arm re-tests this for all 40 seeds** (§4.3).
- **P3, the base demos contain few low-speed pterodactyls.** `createGame` starts any game with `startScore > 0` at `rules.maxSpeed` (`core.js` line 434: `speed = startDistance > 0 ? Math.max(CFG.speed, rules.maxSpeed) : CFG.speed`; `CFG.maxSpeed = 13`). So the `bc-gen` levels 2,000 / 50,000 / 300,000 / 900,000 never see a pterodactyl below speed 13. Pterodactyls spawn only at speed ≥ 8.5 (`core.js` line 230), and the only demonstrations at the speeds of the failures come from the 40 level-0 episodes (`span0` 3,000; about 9,900–10,100 rows each in `runs/exp-bc-duck-stuck/v2-demos-duck-s1-40/demos.json` `episodes`). **This is the coverage hypothesis behind arm (c)**. How many opening ducks the base demos hold is measured, not assumed (check C-c1, §4.2).
- **P4, a natural-ramp start is required for opening demos.** Because of P3, extra demos for the 400–800 window must be level-0 episodes on new seeds, cut to that score window; `--levels 400` would start at speed 13 and miss the target states. Derived, not measured (registered check C-c2): with `CFG.speed` 6, `accel` 0.001 per frame and `scoreCoef` 0.025, the speed at 400 / 453 / 767 / 800 points is about 8.25 / 8.5 / 9.9 / 10. That fits the measured onset of failures at 465 points and the last crash at 767.
- **P5, release rows are rare and underweighted.** In the held-out demos (seeds 41–60), only 2,823 rows hold the ducking flag with label none (step 2, §5.1 step 6). They are labelled *none*, class weight 0.751, while the duck rows they follow weigh 5.10 (step 2, §3, `classWeights`). The release row of each duck is therefore weighed 6.8 times less than the DOWN rows before it.
- **P6, the teacher can label the student's states only as a persistent shadow.** Read from `test/solver.js` `createDuckSolver`: the duck plan lives in a per-instance `WeakMap` and is formed only on the frame the planned jump would take off. A **fresh** solver queried in the middle of a duck has no plan for the pterodactyl overhead. Two cases, both from the code: if the pterodactyl's takeoff window has ended (`r.to <= 0`), it is skipped and the solver answers NONE (release under the pterodactyl). Otherwise the solver plans the duck on that frame but returns NONE, because `ducking` was computed before `plan.set`, and returns DOWN only from the next frame. Either way a fresh solver releases for at least one frame, and an early release is fatal one frame later (stuck Results). **So arm (b) labels with one shadow solver queried on every frame of the student's game from frame 0, as `research/scripts/prereg-bc-robustness/diag.js` does.** Its ground answers are exact. Its airborne answers are defined only after a takeoff the shadow chose (diag.js header, item 4); the others are dropped. Checked by the probe C-b1 (§4.2) before arm (b) runs.
- **P7, the screen's detection power is known** (stuck §6): a stuck seed with per-game rate q ≥ 0.5 % fails the opening scan with ≥ 99.4 % probability; below about 0.2 % it can be missed. "Clean" means q ≲ 0.3 %, not 0.

**Post hoc, from the stuck study, used only as a descriptive check:** in d2w32 the final losses are bimodal. The 14 early-release seeds have 0.0044–0.0098, and the stuck and clean seeds 0.0125–0.0140. d2w8 losses are 0.0139–0.0147, with no such split. No reading uses it as a predictor (§5).

**Shape cap** (user's order, AGENTS.md "Findings that constrain defaults"): ≤ 16 hidden layers, ≤ 32 units per layer. Both shapes comply.

## 3. Hypotheses and predictions
Let S be the number of screen-clean seeds of 20 (tier A 25/25 and opening 1,000/1,000, §4.6) for an arm at a shape.
- **H-weight (the release boundary is underweighted; P5):** the clone learns the release, but the few boundary rows are too light to fix its timing. Prediction: arm (a) meets the bar at both shapes, with both stuck and early-release counts falling; arms (b) and (c) help less.
- **H-coverage (too few low-speed opening ducks; P3):** the release timing at speeds 8.5–10 is learned from the few level-0 episodes, and the rest of the demos teach high-speed ducks with shorter overlaps. Prediction: (c) meets the bar; (b) also helps, since its rows are in the same window; (a) is partial.
- **H-shift (covariate shift):** small timing errors put the student in states the teacher never visits (holding DOWN past the release frame, or standing while a pterodactyl overlaps), and it has never been shown the recovery. Prediction: (b) meets the bar and (c) does not. This is the clean contrast of the design: (b1) and (c) add the same number of episodes in the same score window, on disjoint seeds, and differ only in whose trajectory generates the states (§4.4).
- **H-optimisation (seed-dependent optimisation, not data):** none of the arms meets the bar; the clean count stays near the control's. The d2w32 loss bimodality would be a symptom. The next step is then selection (epoch selection on a disjoint opening set, or multi-seed training plus the screen), registered separately.
- **Side effects predicted for (a):** upweighting the onset rows (duck class, weight 5.10 × 4) could cause early or false DOWN. Watched through the opening scan's no-mid DOWN runs (0 in every control model and the teacher; stuck Results) and through tier A.

## 4. Design
**Run dir:** `runs/exp-bc-duck-fix/`, one level of nesting, so the dashboard lists every run (AGENTS.md invariant). Names follow the scheme `v<R>-<method>-<teacher>-d<D>w<W>-s<seed>[-<tag>]`:

| What | Run dir |
|---|---|
| Base demos (regenerated, gate) | `v2-demos-duck-s1-40` |
| Control | `v2-bc-duck-<shape>-s<k>-ctrl` |
| Arm (a) | `v2-bc-duck-<shape>-s<k>-trw4` |
| Arm (c) demos / clones | `v2-demos-duck-s1001-1200-w400-800` / `v2-bc-duck-<shape>-s<k>-open200` |
| Arm (b) rollouts, round r | `v2-demos-duck-s2001-2200-dagger1-<shape>-s<k>` (r = 1), `v2-demos-duck-s2201-2400-dagger2-<shape>-s<k>` (r = 2) |
| Arm (b) clones | `v2-bc-duck-<shape>-s<k>-dagger1`, `v2-bc-duck-<shape>-s<k>-dagger2` |
| Screens | `opening/<run>.json`, `scans/<run>.jsonl` |

`<shape>` is `d2w8` or `d2w32`, and k runs over the training seeds 3–22.

**Environment (Mac only).** From the executor's worktree at the code commit (§4.1), with `PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`, exactly as the stuck study. One heavy job at a time. The desktops are not used.

### 4.1 Code (one branch, before any run; each flag off by default)
| Change | For | Behaviour | Defaults |
|---|---|---|---|
| `bc-train --extra-demos <dir>[,<dir>…]` | (b), (c) | Appends the rows of each extra demos dir, in the order given, after the base demos. Each must pass `checkDemos` (same core hash, same `obsDim`, same `gameVersion`, same `teacherName`). `best.json` `config` records each extra dir, its sha256 and its row count | Absent: the base-only path is unchanged (no concatenation, no copy) |
| `bc-train --transition-weight W` | (a) | Multiplies the weight of **duck-transition rows** by W after the class weights are computed from the unmodified weights. So class weights and every other row are unchanged. A duck-transition row is grounded (`x[2] == 0`) and either a **duck onset** (`x[3] == 0`, label 2) or a **release** (`x[3] == 1`, label ≠ 2; this includes release-into-jump). `x[3]` is `policy.js` `out[3] = d.ducking`, the state left by the previous frame's keys, so the definition is local to the row and survives boring-row subsampling and episode boundaries. Reports the onset and release counts in the log and in `config` | Absent or W = 1: the code path is skipped (no array op), so weights, RNG draws and params are bit-identical |
| `bc-gen --score-window A-B` | (c), (b) | Keeps only the rows of frames with `A ≤ game.score() < B`. The filter acts before the boring rule, so frames outside the window neither are pushed nor advance the boring counter. The manifest records the window | Absent: rows byte-identical (jump sha `3a67bfd4…`, duck sha `363872f6…`) |
| `bc-gen --student <best.json>` | (b) | The student plays (its keys are applied). A shadow `createDuckSolver` is queried on every frame from frame 0 (P6) and labels the row. Airborne frames after a takeoff the shadow did not choose are dropped and counted (`undefinedAir`). Same boring rule and windows as `bc-gen`. An episode ends at the student's crash or the cap. The manifest records `source: 'dagger'`, the student's path and params sha256, `teacherName: 'duck'`, `studentCrashes` (kept separate from `solverCrashes`, which the dashboard shows as teacher crashes) and `undefinedAir` | Absent: unchanged |

- **Python backend only** for `--extra-demos` and `--transition-weight`; the node backend rejects them, as it rejects `--snapshots` (precedent). Open choice 4.
- **Tests and checks, all before any arm runs:**
  - `node --test test/`, including new `test/bc.test.js` cases: the defaults pin the jump and duck demo sha and the 1-epoch params (existing); `--transition-weight 1` gives params equal to no flag; on a toy row set the onset/release mask is as defined; `--score-window` at 0–∞ is byte-identical to no flag.
  - **DAgger relabel control:** `bc-gen --student` with the duck teacher itself as the student (a test-only hook, as `o.createTeacher` in `genSeed`) must produce rows byte-identical to `bc-gen --teacher duck` with the same seeds, level and window, with `undefinedAir` 0. This shows the persistent shadow reproduces the teacher's labels on the teacher's own trajectory.
  - `node test/bit-identity.js` against `319a51f`, because `trainer/cli.js` changes. ES defaults must be IDENTICAL.
  - **Dashboard, in the same change (AGENTS.md invariant):** the new demos dirs (`source: 'dagger'`, a score window) and the clones with extra demos must be listed and shown. A dagger demos dir must not be shown as a teacher-only run: its summary names the student, and playback offers the student model. Checked with `node test/dashboard-check.js`, with a dagger demos dir in the fixture.
- The executor also writes `research/scripts/prereg-teacher-duck-release-fix/shadow-probe.js` (C-b1) and `counts.py` (C-a1, C-c1).

### 4.2 Gates and premise checks (seconds to minutes; all before any arm is trained)
| Check | Must show | If not |
|---|---|---|
| G1, base demos | `bc-gen --teacher duck --dir runs/exp-bc-duck-fix/v2-demos-duck-s1-40`: 2,062,427 rows, sha256 `363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7` | STOP; explain before training |
| G2, opening-tool controls | Re-run the stuck study's tool controls 1–3 on this commit: teacher 1,000/1,000 with max lag 1 and 0 no-mid runs; committed `runs/exp-bc-duck/v2-bc-duck-d2w32-s2/best.json` gives the same 204 failing seeds and scores; `v2-bc-duck-d2w8-s1` 1,000/1,000 | Fix the tool; read no opening output |
| C-a1, transition counts (`counts.py` on G1's `demos.bin`) | Onset and release row counts, overall and for rows with speed feature below the max-speed level (level-0 episodes). Expected: about one onset and one release per ducked pterodactyl, so about 7,244 each (step 2: 7,244 mid pterodactyls ducked in these demos). The held-out ratio (2,823 release rows with label none, against 21,215 / 5.75 ≈ 3,690 ducks, derived) suggests some releases go straight into a jump (label 1), which the definition counts | If onsets and releases differ from the ducked count by more than 5 %, explain (merged ducks, onsets in other states) before arm (a) runs |
| C-c1, opening coverage (`counts.py`) | The number of duck onsets in the base demos in the 400–800 window, from the level-0 episodes (identified by the row's position in `episodes`) | Recorded before arm (c) runs; it fixes the coverage multiple that (c) delivers. No bar |
| C-c2, speed in the window | One natural v2 game per seed for 5 seeds (the teacher, `core.js` only): the speed at 400 and 800 points and at the first pterodactyl. Expected (P4, derived): about 8.25 / 10 / 8.5 | If the first pterodactyl comes after 600 points or the window misses speeds 8.5–9.9, move the window to match the measured failure range (465–767) and record the change before (c) runs |
| C-b1, shadow probe (`shadow-probe.js`) | On the committed control models, from their opening JSONs: all 192 stuck failures of `d2w8-s10`, all 240 of `d2w32-s16` and the 29 early-release failures of `d2w32-s3`. The student plays, and one persistent shadow is queried from frame 0. (i) Stuck: from at most 2 frames after the last mid overlap of the final duck (`DUCK_MARGIN` 1 plus the teacher's max lag 1), the shadow answers none or jump on every remaining frame of the run, never DOWN. (ii) Early release: the shadow answers DOWN on the student's release frame. (iii) Handover: the same shadow takes control at the first effective disagreement of the final duck, and the game reaches 2,000 points in **≥ 90 %** of the failures. (iv) Documented, not barred: a fresh solver handed control at the same frame (P6), with its crash rate | (i) or (ii) fails: the shadow is not a valid relabeler, so arm (b) is not run; report. (iii) below 90 %: the shadow's labels lead to unrecoverable states; arm (b) is not run; report |

### 4.3 Control arm (positive control and bit-identity gate)
The current procedure, re-run on this branch with every new flag absent:
`bc-train --backend python --hidden <h> --seed <k> --demos runs/exp-bc-duck-fix/v2-demos-duck-s1-40 --snapshots --wasm --dir runs/exp-bc-duck-fix/v2-bc-duck-<shape>-s<k>-ctrl`, for k = 3–22 at both shapes (40 runs), then the opening scan of each.
- **Must show:** each params sha256 equal to the committed `runs/exp-bc-duck-stuck/v2-bc-duck-<shape>-s<k>/best.json` (computed as in stuck §4.2), the same tier-A count and min, and an opening JSON whose failing seeds and scores equal the committed `runs/exp-bc-duck-stuck/opening/v2-bc-duck-<shape>-s<k>.json`. Then the per-seed outcomes are exactly the §2 table (P2), and tier B is not re-run for the control.
- **If any sha differs:** STOP. Either the environment drifted or a new flag leaks into the default path. Nothing from an arm is read until this is explained.
- This is also the full-scale default-identity check for the new `bc-train` code.

### 4.4 Arms (each fixed now)
All arms use the control's command with these additions. Shapes d2w8 and d2w32, training seeds 3–22 (the same as the control, §6).

| Arm | Addition | Fixed parameters, and why |
|---|---|---|
| **(a) transition weight** | `--transition-weight 4` | **W = 4.** A release row then weighs 4 × 0.751 = 3.0, still below the 5.10 of the DOWN rows it follows (P5), so the boundary is no longer drowned but not reversed. The total weight added is about 6 % of the weighted loss (derived from §2 counts: onsets 7,244 × 5.10 × 3 + releases 7,244 × 0.751 × 3 ≈ 127k, against about 2.13M). Duck transitions only: jump takeoffs are not failing and are left unchanged, so the jump decision is not perturbed. Row count and RNG draws are unchanged, so **pairing with the control is exact**: same rows, same init, same permutations; only the transition weights differ. No dose-response is registered (open choice 2) |
| **(c) opening demos** | demos: `bc-gen --teacher duck --seeds 1001-1200 --levels 0 --span0 800 --score-window 400-800 --dir runs/exp-bc-duck-fix/v2-demos-duck-s1001-1200-w400-800`; train with `--extra-demos` that dir | **200 extra seeds.** Each adds one natural-ramp window of the failure range (P4). The base demos have 40 such windows, so (c) multiplies the window's demonstrations by 6, before C-c1 gives the exact duck count. Added rows are at most about 1,750 frames per seed (derived: frames from 400 to 800 points), so at most about 350k rows (+17 %), fewer after the boring rule. Seeds 1,001–1,200 are disjoint from the training demos (1–40), the held-out demos (41–60), the validation seeds (≥ 2^30), the tier-B and opening seeds (30,003,000–30,003,999), the step-1 survey (40,000,000+) and the eval seeds (≥ 2^31). Seed-matched pairing only: the row count changes the permutations |
| **(b1) DAgger, 1 round** | rollouts: `bc-gen --student runs/exp-bc-duck-stuck/v2-bc-duck-<shape>-s<k>/best.json --seeds 2001-2200 --levels 0 --span0 800 --score-window 400-800`; train with `--extra-demos` that dir | The round-1 student is the committed control model of the same shape and seed (equal to the re-run control by §4.3). **The same 200 episodes, window and row rule as (c)**, so (b1) against (c) isolates the state distribution, student against teacher (H-shift against H-coverage). Rollout seeds 2,001–2,200 are disjoint from everything above and from (c). Retrained from scratch with the same `--seed k` on base + rollouts. Seed-matched pairing only |
| **(b2) DAgger, 2 rounds** | rollouts of the (b1) model on seeds 2,201–2,400 (same window); train on base + round-1 + round-2 rollouts | Standard DAgger aggregation. Fresh rollout seeds per round, so round 2 adds new states. It is fixed at 2 rounds, and (b2) is read on its own row, so (b1) is not chosen post hoc |
| ~~(d)~~ | dropped | **Epoch selection on an opening scan** (snapshots exist) is the natural fourth arm. It is dropped here for three reasons. It is a selection, not a training change, so it answers a different decision. It could not select on 30,003,000–30,003,999 without contaminating the screen; it would need its own opening seed range. And it is the H-optimisation follow-up (§7), which is cheaper to register after this result. **Combinations** ((a)+(c), (a)+(b)) are also not registered: they are the follow-up if single arms are partial |

**Order:** control, (a), (c), (b1), (b2), each a separate job, seed order within a job; a crash of one run does not stop the job.

### 4.5 Screen (the order is mandatory, §11)
1. **Tier A:** `bc-train`'s built-in 25-seed validation to 1M.
2. **Opening scan:** `node research/scripts/prereg-teacher-duck-stuck/opening.js --model <run>/best.json --base 30003000 --n 1000 --target 2000 --out runs/exp-bc-duck-fix/opening/<run>.json`, on every new model whatever its tier A (the opening scan is cheap and gives the failure-mode counts).
3. **Tier B:** `node research/scripts/prereg-bc-robustness/scan.js --model <run>/best.json --out runs/exp-bc-duck-fix/scans/<run>.jsonl --base 30003000 --n 1000 --target 1000000 --wasm`, only on seeds that passed steps 1 and 2, with the stuck study's contiguity check. `diag.js --teacher duck` on any failure.

**Tier-B budget, capped now** (the Mac cannot scan every passer of every arm, §9):
- Only an arm that **meets the stage-1 bar at a shape** (S ≥ 19/20, §7) is scanned at that shape.
- Only the **first such arm in the order (a), (c), (b1), (b2)**, from the simplest procedure to the most complex, is scanned by default. Others wait for an orchestrator ruling.
- d2w8: **every** screen-clean seed (the ship shape). d2w32: the **first 6 screen-clean seeds in seed order**, as the stuck study did for S8. The other d2w32 seeds are classed on tier A and the opening scan, and the asymmetry is reported. P1 supports this: in all 13,000 duck-clone tier-B games so far (step 2's 6 duck scans, the stuck study's 7), every failure came at 479–767 points.

### 4.6 Per-seed classification (the stuck study's, unchanged)
- **Screen-clean:** tier A 25/25 and opening 1,000/1,000. This is **the primary paired outcome**, because every seed of every arm and of the control has it.
- **Clean:** screen-clean and, where scanned, tier B 0/1,000. A tier-B failure makes the seed not clean and is recounted in S.
- **Stuck:** a failure anywhere whose final DOWN run has a mid overlap and lasts ≥ 20 frames.
- **Early release:** a death by a mid pterodactyl while not pressing DOWN (stuck Results definition).
- **Other:** any other failure (for example jump-type), reported with `diag.js` hints.

## 5. Recorded per seed and arm
Params sha256, final loss, train `recall.duck`, the transition-row counts (a) or the extra row counts (b, c), `undefinedAir` and `studentCrashes` of each rollout (b), tier-A count and min, opening passes / 1,000 with stuck, early-release and other counts, lag quantiles, no-mid DOWN runs, tier B where scanned, and the class. Descriptive: the final-loss distribution per arm and shape, against the control's bimodality (§2).

## 6. Seeds and power
**Seeds: reuse training seeds 3–22, paired with the control** (decided now). Why:
- **The control is already measured, and exactly reproducible** (P2). Re-running it costs about 17 minutes and also serves as the bit-identity gate (§4.3). Fresh seeds would need a fresh control of the same size, or an unpaired comparison with a historical control.
- **Pairing is free and exact for (a)**: same rows, init and permutations. It is seed-matched for (b) and (c). The exact McNemar test on paired outcomes is valid either way; with c ≈ 0 its power is close to that of Fisher's test on unpaired counts, so **power does not decide between reuse and fresh seeds; cost and the exact baseline do**.
- **Risk of reuse:** the fix was designed after seeing these seeds' outcomes. But no arm is tuned per seed. W, the window and the episode counts are fixed now, and the arms train on game seeds disjoint from the screen seeds (§4.4). The generalisation check is the optional confirmation on fresh training seeds 23–42 (§10, open choice 1).

**What the bar bounds** (Clopper–Pearson, 95 %; computed for this registration):

| Screen-clean / n | Two-sided | Unclean rate, one-sided upper |
|---|---|---|
| 20/20 | 83.2–100 % | 13.9 % |
| **19/20 (the bar)** | 75.1–99.9 % | **21.6 %** |
| 17/20 | 62.1–96.8 % | 34.4 % |
| 39/40 (bar met at 3–22, plus 20/20 on fresh 23–42) | 86.8–99.9 % | 11.3 % |
| 40/40 | 91.2–100 % | 7.2 % |

- **So "fixed" does not mean "every seed is clean".** At the bar, up to about 1 seed in 5 could still be unclean (one-sided). Bounding that rate below 5 % needs 59/59 at a shape. What the bar does show is a large shift from the control: 20 % clean at d2w32 and 40 % at d2w8.
- **Power to meet the bar** at true clean rate p: P(S ≥ 19/20) = 0.98 at p = 0.99, 0.74 at 0.95, 0.39 at 0.90, 0.18 at 0.85, 0.07 at 0.80. The bar demands p near 0.95 or better; an arm at p = 0.85 (still a big gain) will usually land in the "partial" row.

**Paired comparison against the control** (exact McNemar, one-sided, H1: the arm is clean more often). b = seeds screen-clean under the arm only; c = under the control only. Four arms × two shapes = 8 tests, so **Bonferroni α = 0.05 / 8 = 0.00625**:
- c = 0: b = 7 gives p = 0.0078; **b = 8 is the smallest significant count (p = 0.0039)**.
- c = 1: b = 11 gives p = 0.0032.
- **The bar implies significance at both shapes.** d2w32 at 19/20 means b ≥ 15 (the control has 16 unclean seeds); with c ≤ 1, p ≤ 2.6e-4. d2w8 at 19/20 means b ≥ 11 (12 unclean); with c ≤ 1, p ≤ 3.2e-3.
- For reference, unpaired Fisher (one-sided): 19/20 against 8/20 gives p = 2.2e-4, and against 4/20, 1.1e-6.

## 7. Readings (fixed in advance)
Per arm and shape, S = screen-clean seeds of 20. "Significant" = McNemar p ≤ 0.00625 (§6).

| Result | Reading | Next |
|---|---|---|
| G1, G2 or §4.3 control fails | The demos, the tool or the default code path changed | STOP; explain; read no arm |
| C-b1 fails (i), (ii) or (iii) | The shadow does not give valid or recoverable labels on the student's states | (b1), (b2) not run; report. (a), (c) proceed |
| C-a1 or C-c2 outside its range | A premise of (a) or (c) is wrong | Fix and record before that arm runs (§4.2) |
| **S ≥ 19 at both shapes, and every tier-B scan of that arm 0/1,000** | **The arm fixes the release decision** at both shapes, within §6's bounds | The first such arm in the order (a), (c), (b1), (b2) is the proposed procedure: an AGENTS.md BC-procedure change, approved separately. Confirmation on seeds 23–42 if the orchestrator rules it (open choice 1) |
| S ≥ 19 at one shape only | **Fixed at that shape only** | Report per shape. A d2w8-only fix serves the ship shape; d2w32 stays multi-seed plus screen |
| S ≥ 19, but a tier-B failure in a screen-clean seed | The seed is not clean (S is recounted). A late mode exists, so P1 is wrong for this arm | Report; tier B for every passer of that arm at that shape before any reading of "fixed" |
| Significant, but S < 19 | **Partial: the arm helps, not enough** | A combination or dose registration, chosen by which mode remains (below) |
| Not significant, and S within the control ± 2 | **No detectable effect at this power** (a conditional null; scope §8) | That arm's hypothesis is not supported at this dose and scope |
| Significantly worse (c ≥ 8 with b = 0, by the reversed test) | **Harmful** | Report the mode that grew |
| (a) meets the bar, or is the only arm significant | H-weight | — |
| (c) significant and (b1) not, or (c) ≥ (b1) + 3 | H-coverage: the missing ingredient is low-speed demonstrations, not on-policy states | — |
| (b1) significant and (c) not, or (b1) ≥ (c) + 3 | **H-shift**: the student's own states matter, not just more opening data | — |
| (b2) ≥ (b1) + 3 | Iteration helps | (b2)'s procedure is the DAgger candidate |
| No arm significant at either shape | **H-optimisation**: the release is not fixed by weight, coverage or on-policy data at these doses | Register epoch selection on a disjoint opening seed range, or multi-seed training plus the screen, as the procedure |
| An arm removes stuck ducks but adds early releases, or the reverse | **The arm shifts the release timing, it does not sharpen it** | Report both counts. A dose registration, for example a release-only weight, targets the remaining direction |
| (a) adds no-mid DOWN runs or tier-A failures not seen in the control | The onset upweighting creates false ducks | Note; an onset-free variant is a follow-up |

The ±3 margins in the H-coverage and H-shift rows are descriptive. Two arms are not compared with a test here; that would need its own power analysis.

## 8. What would reopen a null
A "no effect" or "partial" reading is scoped to: these doses (W = 4; 200 extra opening episodes; 200 rollouts per DAgger round, 2 rounds), the window 400–800, the base demos (sha `363872f6…`), the python backend on the Mac with `OMP_NUM_THREADS=2`, 15 epochs, batch 1,024, lr 2e-3, final-epoch params, these two shapes and training seeds 3–22. It reopens with a different dose, a combination of arms, more seeds, a different window, or a failure mode found outside the opening.

## 9. Cost (Mac only; one heavy job at a time)
| Item | Count | Each | Source | Total |
|---|---|---|---|---|
| Code, tests, `bit-identity.js`, `dashboard-check.js` | 1 | about 9 s for `bit-identity.js` | AGENTS.md "Verification tools" | minutes |
| G1, G2, C-a1, C-c1, C-c2, C-b1 | 6 | `bc-gen` about 0.6 s; opening controls 5.3 s (teacher), 3.8–12.3 s (models); C-b1 replays 461 failures to 2,000 points | step 2 §9; `opening/ctl-teacher.json` and `opening/*.json` `seconds` | a few minutes |
| `bc-train` d2w32, control and (a) | 2 × 20 | 19 s wall (14.3 s train + 4.2–5.0 s validation) | stuck §9 | 13 min |
| `bc-train` d2w8, control and (a) | 2 × 20 | 15 s wall | stuck §9 | 10 min |
| `bc-train` with up to +17 % rows, (c), (b1) / up to +34 %, (b2) | 3 × 40 | about 17–22 s (scaled, derived) | | about 40 min |
| `bc-gen` for (c), rollouts for (b1), (b2) | 1 + 2 × 40 | seconds each: 200 games × at most 800 points (derived from the opening scan's 12.3 / 4 s per 1,000 games to 2,000 points, plus the shadow at about 5 s per 1,000 such games) | `opening/*.json` | a few minutes |
| Opening scans | 5 × 40 | 12.3 s (d2w32), 3.8–4.3 s (d2w8) | `opening/*.json` `seconds` | about 27 min |
| **Stage 1 total** | | The stuck study's stage 1 (41 trainings, 44 opening scans) took 18 min, against an estimate of about 20 min | stuck Results "Provenance" | **about 95–100 min**, in 5 jobs of 17–23 min (one per arm) |
| Tier B d2w8, one arm | up to 20 | 95–139 s | stuck §9 | 32–46 min |
| Tier B d2w32, one arm | up to 6 | 176–294 s | stuck §9 | 18–29 min |
| **Stage 2 total** (one arm) | | | | **about 50–75 min** |
| Stage 3, optional confirmation (seeds 23–42, winning arm) | 40 trainings + openings + tier B per the cap | as above | | about 20 min + 50–75 min |

Each arm and each scan is a separate job, so the queue can pause between them for other Mac work.

## 10. Scope, and execution order
- **Scope:** rules v2, the 44 unchanged features, `teacher-duck` as merged in `9616c3b`, the base demos of §4.2, the python backend. No `observe()` change, no class-weight default change, no change to `core.js`, `policy.js` or `test/solver.js` (so the core hash stays `0f329d4e…`).
- **Tier-B and opening seeds are not held out. Nothing here is a certification.** No held-out `eval` offsets are spent.
- **Never** `session`, `finish`, `--auto-export` or `export`. Root `model.js` and `model.css` are not touched.
- **Order:**
  1. Orchestrator review and rulings (§12).
  2. The code of §4.1 on a branch, with its tests, `bit-identity.js` and `dashboard-check.js`. Commit before any run.
  3. Gates and premise checks (§4.2); commit their outputs and any recorded change.
  4. Stage 1: control (the gate), then (a), (c), (b1), (b2), each with its opening scans.
  5. Commit, force-added because `runs/exp-*/` is gitignored: `best.json` and `bc-metrics.jsonl` of every new run, every new `demos.json`, and the opening JSONs. Provisional S counts may be reported here.
  6. Stage 2: tier B per the cap (§4.5); `diag.js` on any failure.
  7. Stage 3, only if ruled (open choice 1).
  8. Results appended here, with the code commit, the core hash and the demo shas. The orchestrator merges.

## 11. Proposed AGENTS.md wording (not applied; for the orchestrator)
Proposed for "Invariants to preserve", after "Evaluation uses the unmodified game":

> - **Screen every new duck clone before use** (user's choice, 2026-10-10). A `teacher-duck` clone of any shape or seed is used for nothing beyond its own experiment (ship candidate, certification, starting point for ES or distillation, sweep baseline) until it passes, in this order:
>   1. tier A: `bc-train`'s validation, 25/25;
>   2. **the opening scan:** `node research/scripts/prereg-teacher-duck-stuck/opening.js --model <dir>/best.json --base 30003000 --n 1000 --target 2000 --out <dir>/opening.json`, 1,000/1,000 (0 stuck, 0 early release, 0 other);
>   3. tier B: `node research/scripts/prereg-bc-robustness/scan.js --model <dir>/best.json --base 30003000 --n 1000 --target 1000000 --wasm`, 0/1,000.
>
>   Why: under the step-2 procedure only 12 of 40 fresh seeds were clean, and 4 stuck seeds passed tier A (`research/prereg-teacher-duck-stuck.md`). Tier A alone is not a screen. The opening scan takes seconds and catches both release failures, which so far all occurred below 800 points. On a new commit, first check the tool: `opening.js --teacher duck` must give 1,000/1,000 with max lag ≤ 2. The screen seeds are not held out, so passing it is not a certification. A new game version needs its own screen range and target.

And in "Behaviour cloning" (Commands), one line: "Duck clones: train, then screen (Invariants, 'Screen every new duck clone'); a clone that fails the screen is not used."

## 12. Open choices for the orchestrator
1. **Stage 3 (fresh confirmation on training seeds 23–42 for the winning arm):** proposed as optional, run only if an arm meets the bar. With 20/20 it takes the bound from 21.6 % to 11.3 % (§6). The default here is to run it only if ruled.
2. **Dose:** a single W = 4 for (a), 200 episodes for (c) and per DAgger round. A second dose (for example W = 8, or 400 episodes) would double that arm's stage 1 (about 20 min). Not registered.
3. **Tier-B cap:** d2w8 every passer, d2w32 the first 6, and only the first qualifying arm. The alternative is every passer at both shapes (d2w32 up to 20 × 3–5 min, about 60–100 min).
4. **Backends:** the new `bc-train` flags are python only (as `--snapshots`). Porting them to the node backend is not needed for this study.
5. **Arm (d):** dropped (§4.4). Epoch selection on a disjoint opening seed range is the follow-up if no arm is significant.
6. **The AGENTS.md wording** of §11: where it goes and whether the opening scan's seed range should be fixed there or left to each registration.
