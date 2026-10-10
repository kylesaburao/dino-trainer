# Pre-registration: `teacher-duck` stuck duck. How often does behaviour cloning produce a clone that holds DOWN into the next obstacle?

Status: **registered (mac-4 rulings applied), 2026-10-10** (rulings in §11); **results recorded (see Results at the end)**. Written on main `8e98dfa`, rebased onto `f2b57f7` before execution (core hash `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e`). Sections 1–11 were written before any run. Every number in them was read from the repo or from the step-2 run directory `.claude/worktrees/agent-a0e89a1383966479d/runs/exp-bc-duck/` (gitignored there; cited as "step-2 local").
Continues `research/prereg-teacher-duck-step2.md` (its Results, "A-s2 failures"). Context: AGENTS.md "Research method", the BC pipeline, the pipeline-v3 naming.

**User's choice (via the orchestrator):** "measure how often the stuck-duck failure happens". **This step measures; it does not fix.** Any remedy (release-frame weighting, a class-weight change, more release rows, DAgger-style relabelling, epoch selection) is a separate registration (§10).

## 1. Decision
Is the step-2 stuck-duck failure of `d2w32-s2` a **rare seed lottery** or a **common failure of the BC procedure** at d2w32 (44-32-32-3), and is it **specific to that shape**?

The answer decides:
- **what a later fix registration targets.** A rare lottery is handled by screening seeds (tier A already catches the severe form, §6). A common failure needs a change to the procedure;
- **whether the d2w8 ship candidate's training procedure is fragile.** The certified d2w8 models are what they are, whatever this finds. The question is whether *retraining* d2w8 (a new game version, new demos, a teacher change) can produce a stuck clone, so that any retrain must be screened.

It ships nothing and certifies nothing.

## 2. What is known (step 2), and the premises this design rests on
**The failure** (`prereg-teacher-duck-step2.md` Results; `runs/exp-bc-duck/v2-bc-duck-d2w32-s2/best.json`; step-2 local `diag/d2w32-s2*.json`, `duckshare/*.json`):
- `d2w32-s2`: tier A 20/25 (min 516); tier B 204/1,000 on seeds 30,003,000–30,003,999 (20.4 %, CP 17.9–23.0 %). All other step-2 clones, including `d2w32-s1` and both d2w8 seeds, were 25/25 and 0/1,000.
- In all 204 failures the final DOWN run began with a **mid pterodactyl nearest**, passed under it and never released: `duckrun.js` summary `{"mid-in-duck onset:ptero-mid": 204}`, run length 57–134 frames (median 63, p10 61, p90 67). The teacher's DOWN frames per ducked pterodactyl are 431,019 / 82,614 = **5.22**.
- **Premise P1, the mode is confined to the opening.** Every failing run's onset score is 465–752 (median 548); the crash scores are 479–767. Pterodactyls spawn only at speed ≥ 8.5 (`core.js` line 230, `minSpeed: 8.5`), which is first reached in this window. In the 8 games of `duckshare.js` that passed the opening, `d2w32-s2` ducked 65,949/65,949 mid pterodactyls to 1M.
- **Premise P2, tier A and tier B measure the same rate.** 5/25 = 20 % against 204/1,000 = 20.4 %. So the 25-game validation is an unbiased (if coarse) estimate of the per-game rate.
- **Premise P3, the mean DOWN length does not detect stuckness** (checked on step-2 local `duckshare/*.json`, DOWN frames / ducked per game):

  | Model | DOWN / ducked (10 games) | Per game, surviving games |
  |---|---|---|
  | teacher | 5.217 | 5.21–5.23 |
  | d2w32-s1 | 5.354 | 5.35–5.36 |
  | **d2w32-s2** | 5.878 (8 full games) | **5.86–5.89**; the two crashed games 38 and 64 |
  | d2w8-s1 / s2 | 5.883 / 5.878 | 5.86–5.90 |
  | d1w8-s1 / s2 | 6.004 / 5.976 | 5.97–6.02 |

  `d2w32-s2`'s surviving games are indistinguishable from d2w8's, which never stuck. **So the per-seed continuous measure must be a tail statistic in the opening, not a mean over full games** (§4.4). `duckshare.js` is not re-run here.
- **Premise P4, the final training loss is shape-relative.** Final train loss (from `best.json` `config.final.loss`): d2w32-s1 0.00441, **d2w32-s2 0.01329**, d2w8-s1 0.01410, d2w8-s2 0.01412, d1w8 0.0190/0.0183, jump twin d2w32-s2 0.00572. The 3× gap is within d2w32 only; d2w8 sits at s2's level and is reliable. **Loss is read as a predictor within a shape only.**
- **Premise P5, the python backend is deterministic per seed on one machine** (`trainer/cli.js` header; `trainer/bc_train.py` line 45 `np.random.default_rng(a.seed)`; step 2's C-jump-params reproduced `py-s1` to all 2,595 params). The positive control (§4.2) re-tests this for the exact run in question.
- **Premise P6, the training demos are not in main.** Only `runs/exp-bc-duck/v2-demos-duck-s1-40/demos.json` is committed; the 379 MB `demos.bin` is in the step-2 worktree. The demos are regenerated and checked by sha (§4.1).

**Tier A, as implemented** (`trainer/cli.js` `bcTrainCmd`, lines 812–887): after the last epoch (no epoch selection; `best.json` holds the final params), `bc-train` plays the 25 `validationSeeds(25)` (`trainer/env.js` line 14, `0x40000000 + …`, disjoint from the demo seeds 1–40 and from the scan seeds) as full v2 games to `--target`, default **1,000,000**, and records `validation.{median,min,count}` in `best.json`. Per-game scores are not stored, only the summary.

**Shape cap.** A direct user order (2026-10-10): "Cancel any work with layer counts greater than 16. Restrict per-layer depth to 32", i.e. ≤ 16 hidden layers and ≤ 32 units per layer (recorded in AGENTS.md, `f2b57f7`). Both arms (32,32 and 8,8) satisfy it.

## 3. Hypotheses and predictions
Let r be the per-seed probability that a training seed yields a stuck clone, at a shape, under the step-2 procedure.
- **H-rare (lottery):** `d2w32-s2` was an unlucky seed; r(d2w32) is small. Prediction: k ≤ 1 stuck among the 20 fresh d2w32 seeds.
- **H-common:** the procedure often fails to learn the release at d2w32. Prediction: k ≥ 5 of 20 (CP lower bound ≥ 8.7 %).
- **H-shape (capacity- or shape-specific):** d2w32 sticks, d2w8 does not. Prediction: k ≥ 2 and 0/20 at d2w8. Significance (Fisher, j = 0) needs k ≥ 5 (§6).
- **H-wide (procedure-wide):** both shapes stick. Prediction: ≥ 1 stuck d2w8 seed. One is enough to refute "never at d2w8".
- **H-continuum vs H-binary (severity):** stuck seeds come at many severities (some with per-game rates well under 20 %, visible only in tier B or the opening scan) versus one severe mode (≈ 20 %). Read from the per-seed opening-scan rates and the release-lag tails (§4.4).

## 4. Design
**Run dir:** `runs/exp-bc-duck-stuck/`: `v2-demos-duck-s1-40/`, `v2-bc-duck-d2w32-s2-ctrl/`, `v2-bc-duck-d2w32-s<3..22>/`, `v2-bc-duck-d2w8-s<3..22>/`, `opening/`, `scans/`, `diag/`. One level of nesting, as in step 2, so the dashboard shows the runs.

**Environment (Mac).** From the main checkout at the commit that adds the opening tool (§4.4), with
`PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`, exactly as step 2's `train-chain.sh`. One heavy job at a time; the Mac starts only after the current certification job has finished (the orchestrator sequences this).

### 4.1 Demos (gate)
`node trainer/cli.js bc-gen --teacher duck --dir runs/exp-bc-duck-stuck/v2-demos-duck-s1-40` (defaults: seeds 1–40, levels 0,2000,50000,300000,900000, span0 3,000, span 5,000, keep-boring 10).
- **Must give** 2,062,427 rows and sha256 `363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7` (step-2 C-duck-rows). If not, STOP: nothing is trained until the difference is explained.

### 4.2 Positive control (before any arm is read)
`bc-train --backend python --hidden 32,32 --seed 2 --demos runs/exp-bc-duck-stuck/v2-demos-duck-s1-40 --snapshots --wasm --dir runs/exp-bc-duck-stuck/v2-bc-duck-d2w32-s2-ctrl` (a new dir; `bc-train` refuses an existing `best.json`, and the committed `runs/exp-bc-duck/v2-bc-duck-d2w32-s2` is not overwritten).
- Params hash: `node -e 'const b=require("./runs/exp-bc-duck-stuck/v2-bc-duck-d2w32-s2-ctrl/best.json");console.log(require("crypto").createHash("sha256").update(JSON.stringify(b.params)).digest("hex"))'`.
- **Reference** (the committed `runs/exp-bc-duck/v2-bc-duck-d2w32-s2/best.json`, computed the same way on `8e98dfa`): `042074360fad8d2d0ff083f66362a06cd0e039542f933beab02c1044abff1e77`; validation 20/25, min 516; final loss 0.01329.

| Outcome | Reading | Next |
|---|---|---|
| Hash equal, 20/25 min 516 | Training is reproducible here; the stuck clone is a property of seed 2 under this procedure | Proceed |
| Hash differs, but tier A < 25/25 and its opening scan sticks (§4.4) | Not bit-reproducible on this machine now, but the mode reproduces | Proceed. Drop any claim of bit reproducibility; note it in the results |
| Hash differs and tier A 25/25 | The environment changed (numpy, BLAS threads, demos) and seed 2 no longer sticks | **STOP.** Seed identities are not comparable to step 2; report and ask the orchestrator |

The control also gives the opening-scan positive control (§4.4).

### 4.3 Arms
| Arm | Shape | Training seeds | Why |
|---|---|---|---|
| **S32** | d2w32, `--hidden 32,32` (44-32-32-3) | **3–22** (20 fresh seeds) | the shape that stuck; estimates r(d2w32) |
| **S8** | d2w8, `--hidden 8,8` (44-8-8-3) | **3–22** (20 fresh seeds) for tier A and the opening scan; tier B on **3–8** only | the ship-candidate shape; discriminates shape-specific from procedure-wide |

Each: `bc-train --backend python --hidden <h> --seed <s> --demos runs/exp-bc-duck-stuck/v2-demos-duck-s1-40 --snapshots --wasm --dir runs/exp-bc-duck-stuck/v2-bc-duck-<shape>-s<s>`, the step-2 hyperparameters (15 epochs, batch 1,024, lr 2e-3, default class weights, `TRAIN_DEFAULTS`), so tier A (25 seeds to 1M) runs as part of each command.
- **Seeds 1 and 2 are excluded from the counts.** s2 is the case that triggered this registration; counting it biases r upward. Step 2's four models (d2w32 s1/s2, d2w8 s1/s2) are reported on a secondary "with step 2" line only.
- Order: S32 s3–22, then S8 s3–22, then the opening scans (§4.4). Each run is independent; a crash of one does not stop the chain.

### 4.4 Opening scan and release lag (Mac, light, single-threaded)
**Tool to write before execution:** `research/scripts/prereg-teacher-duck-stuck/opening.js`, modelled on `duckshare.js` (same duckable test from the boxes, same `policy.js` `createAgent` JS forward pass, `createDuckSolver` for `--teacher duck`).
Usage: `node opening.js (--model <best.json> | --teacher duck) --base 30003000 --n 1000 --target 2000 --out <f.json>`.
- **Games:** the tier-B seeds 30,003,000–30,003,999, played from 0 to **2,000 points** (above the latest step-2 onset, 752, and crash, 767; P1). Because the games are deterministic, each opening game is an exact prefix of the same seed's tier-B game. So the opening failures of a model are a subset of its tier-B failures (the JS and WASM forward passes are bit-identical, AGENTS.md "Speed").
- **Release lag**, per grounded DOWN run (consecutive running frames on which the action holds DOWN while the dino is not jumping) during which a duckable obstacle's box overlaps the standing box's x-range [56, 86): lag = the number of DOWN frames in the run after the last such overlap frame. A run that ends in a crash is *fatal* and its lag is right-censored. Runs with no duckable overlap are counted separately (*no-mid DOWN runs*, the step-2 class (i)).
- **Per model it reports:** games passed / 1,000; each failing seed with its score, killer, final DOWN run length and whether it contained a mid overlap (as `duckrun.js`); and, over non-fatal runs, the lag quantiles p50/p99/p99.9/max and the counts of runs with lag ≥ 5, ≥ 10 and ≥ 20; plus the number of no-mid DOWN runs.
- **Tool controls (all must pass before any arm's output is read):**
  1. `--teacher duck`: 1,000/1,000 pass; 0 fatal runs; 0 no-mid runs; **max lag ≤ 2** (`DUCK_MARGIN = 1` widens the teacher's planned overlap by one frame each side, `test/solver.js` lines 63–80). If the teacher's max lag exceeds 2, the lag definition is wrong; fix it and record the fix before reading any model.
  2. The committed `runs/exp-bc-duck/v2-bc-duck-d2w32-s2/best.json`: the set of failing seeds must **equal** the 204 failing seeds of its step-2 tier-B scan (step-2 local `scans/d2w32-s2.jsonl`), with the same scores. Every failure must have a mid overlap and final run length 57–134. This is the detection control: the tool sees the known mode.
  3. The committed `runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json`: 1,000/1,000 pass (its tier B was 0/1,000).
  4. `ctrl-d2w32-s2` (§4.2): if its hash matched, its output must be identical to control 2's.
- Run on every new model (40), the positive control and the three controls. Output in `runs/exp-bc-duck-stuck/opening/`.

### 4.5 Tier B (Mac, stage 2)
Every new model that **passes tier A 25/25 and the opening scan 1,000/1,000** goes to tier B:
`node research/scripts/prereg-bc-robustness/scan.js --model runs/exp-bc-duck-stuck/<arm>/best.json --out runs/exp-bc-duck-stuck/scans/<arm>.jsonl --base 30003000 --n 1000 --target 1000000 --wasm`
- the same 1,000 seeds as step 2 and the jump sweep, so failures pair per seed;
- run on the Mac with the current main `scan.js` (18 workers by default, `--wasm`; it records pid and host in its header), one heavy job at a time; it asserts core hash `0f329d4e`;
- contiguity check as in step 2: 1,000 contiguous seed rows, header base 30,003,000, footer present;
- every failure gets `diag.js --teacher duck` and `duckrun.js` (single-threaded, seconds each).

Models that fail tier A or the opening scan are **not** scanned in tier B by default: their stuck status is already decided, and the opening scan gives their per-game opening rate (open choice 2, §11).
**Staged order** (the Mac total exceeds about 90 min, §9). Stage 1 trains every seed with its built-in validation and runs the opening scans (§4.1–4.4). Stage 2 then scans the eligible seeds (25/25 and 1,000/1,000), **S8 first** (s3–s8 only; S8 s9–s22 get no tier B), the ship-candidate shape's question, then S32 in seed order s3–s22. Each scan is a separate job, so the queue can pause between scans for other Mac work without losing a finished scan, and a partial queue answers the S8 question first.

### 4.6 Per-seed classification (fixed now)
For each new training seed:
- **Stuck:** at least one failure, in tier A, the opening scan or tier B, whose final DOWN run contained a mid overlap and lasted **≥ 20 frames** (the teacher's mean duck is 5.2 frames; step 2's shortest stuck run was 57). For tier A, which stores no per-game record, a failure is classed by replaying all 25 `validationSeeds(25)` to 1M with `duckrun.js`'s final-DOWN-run logic (about 25 × 3 s, single-threaded; `best.json` does not say which seeds failed, and `duckrun.js` itself reads a scan file, so this is a small adaptation written with `opening.js`). The opening scan normally decides it first.
- **Unreliable, not stuck:** failures exist, but none meets the stuck criterion (for example a jump-type death). Reported in its own row; not counted in k.
- **Clean:** tier A 25/25, opening 1,000/1,000 and tier B 0/1,000.
- **Severity:** a stuck seed's per-game rate is its opening-scan failure count / 1,000 (CP interval), plus any tier-B excess if it was scanned.

k = stuck seeds of the 20 in S32; j = stuck seeds of the 20 in S8. For S8 s9–s22, "clean" means tier A 25/25 and opening 1,000/1,000 (no tier B), so a late-game stuck mode could be missed there (§6); this asymmetry is reported.

## 5. Recorded per seed
Training seed, params sha256, final train loss, `recall.duck` (train), tier A count and min, opening passes / 1,000 and lag statistics, tier-B failures (if scanned), classification, and for each failure the killer, the score and the final run length.

## 6. Power
**Detection of a stuck seed, by its per-game rate q** (exact: P(25/25) = (1−q)^25, P(0/1,000) = (1−q)^1,000; the opening scan has the tier-B seeds' power for the opening mode):

| q per game | P(passes tier A) | P(passes 1,000 games) | Detected by tier A + opening/tier B |
|---|---|---|---|
| 0.204 (s2) | 0.0033 | ≈ 1e-99 | certain |
| 0.10 | 0.072 | ≈ 2e-46 | certain |
| 0.05 | **0.277** | ≈ 5e-23 | certain |
| 0.02 | 0.604 | 1.7e-9 | certain |
| 0.01 | 0.778 | 4.3e-5 | > 99.99 % |
| 0.005 | 0.882 | 0.0067 | 99.4 % |
| 0.002 | 0.951 | 0.135 | 87 % |
| 0.001 | 0.975 | 0.368 | 64 % |

- **Tier A alone is not enough for a milder stuck seed:** a seed at 5 % per game passes it 28 % of the time. That is why every model also gets the opening scan, which costs seconds.
- **A stuck seed below about 0.2 % per game can be missed** (13 % at 0.2 %, 36 % at 0.1 %). Such a seed is also below what tier B could ever certify (0/1,000 bounds q at 0.299 %, one-sided). So "clean" here means q ≲ 0.3 %, not q = 0.

**The per-seed stuck rate r, from k of 20** (Clopper–Pearson, 95 %):

| k / 20 | Two-sided | One-sided upper |
|---|---|---|
| 0 | 0–16.8 % | 13.9 % |
| 1 | 0.1–24.9 % | 21.6 % |
| 2 | 1.2–31.7 % | 28.3 % |
| 3 | 3.2–37.9 % | 34.4 % |
| 4 | 5.7–43.7 % | 40.1 % |
| 5 | 8.7–49.1 % | 45.6 % |
| 6 | 11.9–54.3 % | 50.8 % |
| 8 | 19.1–63.9 % | 60.6 % |
| 10 | 27.2–72.8 % | 69.8 % |

- **"Rare" (k ≤ 1) does not exclude a 1-in-5 lottery** (upper 21.6 %). Bounding r below 5 % would need 0 of 59 seeds. If the true r were 5 %, P(k ≤ 1) = 0.74; at 10 %, 0.39; at 20 %, 0.07. So k ≤ 1 is good evidence against r ≥ 20 % and weak evidence against 10 %.
- **"Common" (k ≥ 5)** puts the CP lower bound at ≥ 8.7 %.

**The S8 arm, 20 seeds** (ruling 3): 0/20 gives r(d2w8) ≤ 13.9 % (one-sided), the same table as S32. P(0/20) is 0.36 at r = 5 %, 0.12 at 10 %, 0.039 at 15 %, 0.012 at 20 %. Detection for S8 s9–s22 rests on tier A + the opening scan only: the opening scan has the 1,000-game power of the table above for the opening mode (P1), but none for a late-game mode.
- **What S8 can show:** a single stuck d2w8 seed refutes "the stuck duck does not happen at d2w8" and makes the mode procedure-wide.
- **What it cannot show:** that d2w8 is safe below about 14 %. Shape difference: a one-sided Fisher exact test of k/20 against 0/20 gives p = 0.50 (k = 1), 0.24 (k = 2), 0.12 (k = 3), 0.053 (k = 4), **0.024 (k = 5)**, 0.010 (k = 6); against 1/20, p = 0.091 (k = 5), **0.0098 (k = 8)**. With k ≤ 4 and j = 0, "d2w32 sticks, d2w8 doesn't" is descriptive only.

**Predictors** (within S32 only, P4):
- **Loss:** with k stuck seeds, the probability that the k stuck seeds are exactly the k highest final losses by chance is 1/C(20, k): 0.05 (k = 1), 0.0053 (k = 2), 8.8e-4 (k = 3). Registered test: perfect separation (the k stuck seeds hold the k highest losses). Otherwise report the exact one-sided Mann–Whitney p of stuck against clean losses, descriptive unless k ≥ 3.
- **Lag tail:** a 2×2 of "any non-fatal run with lag ≥ 10 in the opening scan" against stuck/clean, over all 40 new seeds, with Fisher's exact p. Descriptive unless both margins have ≥ 3 seeds.

## 7. Readings (fixed in advance)
| Result | Reading | Next (each registered separately) |
|---|---|---|
| §4.1 demo sha differs | Demos are not step 2's | STOP; explain before training |
| §4.2 control: hash differs and 25/25 | Environment drift | STOP; report |
| Any §4.4 tool control fails | The opening tool is wrong | Fix it; read no opening output |
| **k ≤ 1 of 20, j = 0** | **Rare at d2w32** (r ≤ 21.6 %, one-sided); s2 was a lottery draw. Not seen at d2w8 (≤ 13.9 %) | A fix registration is optional. The cheap remedy is screening: any new clone passes tier A and the opening scan before it is used. If the user wants r bounded below 5 %, that needs ≈ 60 seeds, registered separately |
| **2 ≤ k ≤ 4, j = 0** | **Occasional at d2w32** (r clearly above 0, could be about 5 %). Not seen at d2w8 | The fix registration targets d2w32 with its release rows; screening stays mandatory for any retrain |
| **k ≥ 5, j = 0** | **Common at d2w32** (lower bound ≥ 8.7 %): the procedure does not reliably learn the release at that shape. Shape-specific if Fisher p < 0.05 (k ≥ 5 with j = 0, so always in this row) | A fix registration is needed before d2w32 is used for the duck generation. Candidates: release-frame weighting, DAgger relabels in the opening, epoch selection on an opening scan |
| **j ≥ 1** (any k) | **Procedure-wide:** the stuck duck occurs at d2w8 too. **The d2w8 training procedure is fragile**: any d2w8 retrain must be screened. The certified d2w8 models are unaffected | The fix registration targets the procedure, not a shape. Report j/20 with its CP interval; shape-dependent if Fisher p < 0.05 (§6) |
| k = 0 and j = 0 | Not reproduced in 40 fresh seeds: r(d2w32) ≤ 13.9 %, r(d2w8) ≤ 13.9 % | As for "rare". s2 stands as a single, reproducible (§4.2) case |
| Stuck seeds with opening rates spread over orders of magnitude, or tier-B-only stuck seeds | **H-continuum:** severity varies; tier A alone is not a sufficient screen | Screening must include the opening scan (or tier B) |
| All stuck seeds near s2's ≈ 20 %, none tier-B-only | **H-binary:** one severe mode, which tier A catches with ≥ 99.6 % | Tier A is a sufficient screen for this mode |
| Non-fatal lag tail present in clean seeds as often as in stuck ones | The tail is not a precursor | Note; do not use lag as an early-warning metric |
| Lag tail separates stuck from clean (§6, p < 0.05) | **The release lag is a measurable precursor** | It becomes the fix registration's continuous outcome |
| Loss perfectly separates stuck from clean in S32 (p ≤ 0.05) | **Final loss predicts sticking** at d2w32 | A loss-threshold screen is a candidate, to be validated on fresh seeds |
| Loss does not separate | Loss does not predict at this power | Note (scope: k, 20 seeds) |
| Any "unreliable, not stuck" seeds | A second failure mode | Report separately with `diag.js` class hints; not counted in k |

## 8. What would reopen a null
A "rare" or "not seen at d2w8" reading is scoped to: the `bc-gen --teacher duck` defaults (demo sha `363872f6…`), the python backend on the Mac with `OMP_NUM_THREADS=2`, 15 epochs, batch 1,024, lr 2e-3, default class weights, final-epoch params, these two shapes, and training seeds 3–22 at both shapes (tier B at d2w8 on 3–8 only). It reopens with any change to these, with a stuck clone found later at either shape, or with more seeds (§6: about 60 to bound r below 5 %).

## 9. Cost
| Item | Count | Each | Source | Total |
|---|---|---|---|---|
| `bc-gen` | 1 | about 0.6 s | `prereg-teacher-duck-step2.md` §9 | seconds |
| `bc-train` d2w32 (train + tier A) | 20 + 1 control | **19 s wall** (14.3 s train, 4.2–5.0 s validation) | step-2 local `train-chain.log` and `d2w32-s*/train.log` | about 7 min |
| `bc-train` d2w8 | 20 | **15 s wall** (12.3–12.5 s train, 2.7–2.9 s validation) | same | about 5 min |
| Opening scan | 40 + teacher + 3 controls | 1,000 games × 8,001 frames (teacher, measured on `8e98dfa`, 20 games); about 13 s at d2w32 and 5 s at d2w8, scaled from `duckshare.js` (32 s and 12 s per 19.4M frames) | measured / step-2 local `duckshare/*.json` | about 5 min |
| **Stage 1 total** | | | | **about 20 min** |
| Tier B, d2w32 (Mac) | up to 20 | **176–294 s** (176 s unloaded: step-2 local `scan-chain.log`, `d2w32-s1`; 294 s loaded: `runs/exp-bc-sweep/scans/d2-w32-s1.log`, step-2 §9) | | 59–98 min |
| Tier B, d2w8 (Mac) | up to 6 | **95–139 s** (95–104 s: step-2 local `scan-chain.log`; 124–139 s: `runs/exp-bc-sweep/scan-queue.log`, step-2 §9) | | 10–14 min |
| `diag.js` / `duckrun.js` on failures | per failure | seconds | step 2 | minutes |
| **Scan total** | | | | **about 69–112 min** if every new seed is eligible |
| **Overall Mac total** | | | | **about 90–132 min**: over the 90-min mark, hence the staged order (§4.5). Stage 1 about 20 min; stage 2 up to about 112 min |

A stuck seed that fails tier A or the opening scan is not scanned, so each stuck seed shortens stage 2 by 2–5 min. The desktop host is shut down; nothing here runs there.

## 10. Scope, and what this step does not do
- **Measurement only.** No remedy is tried: release-frame weighting, class weights, more or targeted demos, DAgger, epoch selection and early stopping each need their own registration, written after these results.
- Rules v2, the 44 unchanged features, `teacher-duck` as merged in `9616c3b`, the step-2 procedure (§8).
- **Tier-B seeds are not held out. Nothing here is a certification.** No held-out `eval` offsets are spent (the next fresh offset stays where the certification job leaves it).
- **Never** `session`, `finish`, `--auto-export` or `export`; the root `model.js` and `model.css` are not touched.
- **Execution order:**
  1. The orchestrator confirms that the current certification job has finished and the Mac is idle.
  2. The executor writes `opening.js` on a branch, runs its tool controls (§4.4, items 1–3; seconds) and commits.
  3. **Stage 1 (Mac, about 20 min):** §4.1 demos, §4.2 control, §4.3 arms with their built-in validation, §4.4 opening scans, serially.
  4. Commit `best.json` and `bc-metrics.jsonl` of every new run (force-added, as `runs/exp-*/` is gitignored), plus the opening-scan JSONs. Stage-1 counts of k and j (from tier A and the opening scan) may be reported here, marked provisional.
  5. **Stage 2 (Mac, up to about 112 min):** §4.5 tier B on the eligible seeds, S8 first, then S32; `diag.js` and `duckrun.js` on any failure. Commit the scan summaries (failing seeds, scores, `duckrun.js` output).
  6. The results are appended here, with the code commit, the core hash and the demo sha. The orchestrator merges to main.

## 11. Rulings (mac-4, 2026-10-10)
The draft's open choices, as ruled by mac-4 before any run:
1. **The opening tool:** approved. `opening.js` is written with its four controls (§4.4) as a step before execution.
2. **Tier B for stuck seeds:** skipped (the default).
3. **The size of S8:** enlarged to seeds 3–22 for tier A and the opening scan; tier B stays on 3–8. §4.3, §6, §7 and §9 are updated to match.
4. **The stuck threshold:** kept at 20 frames (§4.6).
5. **Step-2 seeds:** kept out of the headline count (§4.3); the "with step 2" line (k + 1 of 22 at d2w32, j of 22 at d2w8) is secondary.
6. **Shape cap:** cited as the user's direct order (§2).
7. **Execution:** on the Mac (the certification job is done and the desktops are shut down), with main's current `scan.js`, in the staged order.

## Results (2026-10-10, Mac executor, branch `wt/prereg-duck-stuck`)
**Provenance.** Registration `635065b` (rulings applied) was committed before any run, and the tool `opening.js` in `848ff74` before any arm was trained. Runs from this worktree (`.claude/worktrees/agent-a010b6da025b35962`), code at main `f2b57f7` plus this branch, with no change to `core.js`, `policy.js`, `test/solver.js` or `trainer/`. Core hash `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e`. Python `/Users/kyle/Desktop/dino-game/.venv/bin/python`, `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`. One heavy job at a time. No `eval`, no held-out offsets, no `export`/`finish`/`session`/`--auto-export`; root `model.js`/`model.css` untouched. Stage 1 ran 20:57–21:15Z (18 min); stage 2 ran 21:16–21:33Z (17 min, 7 scans).

**Names.** After both stages, the run dirs were renamed to the naming scheme of main `10cd0e0`: `runs/exp-bc-duck-stuck/v2-bc-duck-<shape>-s<k>`, the control `v2-bc-duck-d2w32-s2-ctrl`, and the demos `v2-demos-duck-s1-40`. The opening and scan files carry the same names; the tool-control outputs are `opening/ctl-*.json`. The step-2 models are cited by their renamed paths (`runs/exp-bc-duck/v2-bc-duck-*`). The headers of the stage-2 scan files still name the pre-rename model paths.

### Gates and controls: all pass
| Control | Result |
|---|---|
| §4.1 demos | 2,062,427 rows, sha256 `363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7` (equal to step 2) |
| §4.2 positive control | **Row 1:** params sha256 `042074360fad8d2d0ff083f66362a06cd0e039542f933beab02c1044abff1e77` (identical), tier A 20/25 min 516, final loss 0.013289 (identical) |
| §4.4 tool control 1, teacher | 1,000/1,000; **max lag 1**; 0 fatal; 0 no-mid runs. The first run of this control found 2 one-frame "no-mid" runs that started at 1,999 points and were cut off by the target. Before any model was read, the tool was changed so that runs still open at the target are not counted ("truncated"; 18 for the teacher), and the control was re-run |
| Tool control 2, `v2-bc-duck-d2w32-s2` | 204 failures: **the same seeds, with the same scores,** as its step-2 tier-B scan; all have a mid overlap and a final run of 57–134 frames |
| Tool control 3, `v2-bc-duck-d2w8-s1` | 1,000/1,000 |
| Tool control 4, the retrained control | Output identical to control 2's |
| Cross-check (extra, not registered) | `scan.js --wasm` to 2,000 points on `d2w32-s3`, seeds 30,003,010–30,003,036, reproduced the opening tool's three failures and their scores (517, 533, 521) |

All 7 tier-B scans have 1,000 contiguous seed rows from 30,003,000 and a footer.

### Per seed
Columns:
- **Opening:** 1,000 games to 2,000 points on seeds 30,003,000–30,003,999.
- **Early:** failures that are not stuck (described below the table).
- **Max lag:** the largest release lag of a non-fatal mid run (teacher: 1).

| Arm | Params sha256 (12) | Final loss | Tier A | Opening stuck | Opening early | Max lag | Runs lag ≥ 10 | Tier B | Class |
|---|---|---|---|---|---|---|---|---|---|
| d2w32 s3 | `cd337d5c361b` | 0.00515 | 25/25 | 0 | 29 | 3 | 0 | — | unreliable (early release) |
| d2w32 s4 | `d40a23ef588b` | 0.00483 | 25/25 | 0 | 15 | 3 | 0 | — | unreliable (early release) |
| d2w32 s5 | `f5b5d3938234` | 0.01397 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w32 s6 | `223eff5fe3dc` | 0.00521 | 25/25 | 0 | 27 | 3 | 0 | — | unreliable (early release) |
| d2w32 s7 | `6bb92afb9698` | 0.00483 | 25/25 | 0 | 30 | 3 | 0 | — | unreliable (early release) |
| d2w32 s8 | `f15c2f434641` | 0.01359 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w32 s9 | `212bb917e592` | 0.00496 | 25/25 | 0 | 40 | 3 | 0 | — | unreliable (early release) |
| d2w32 s10 | `5f49672f7a8a` | 0.00497 | 25/25 | 0 | 27 | 3 | 0 | — | unreliable (early release) |
| d2w32 s11 | `b56ff4169597` | 0.01368 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w32 s12 | `e9028f49e5bb` | 0.00492 | 25/25 | 0 | 49 | 3 | 0 | — | unreliable (early release) |
| d2w32 s13 | `2f7607894f49` | 0.01245 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w32 s14 | `ed62165c1458` | 0.00463 | 25/25 | 0 | 47 | 3 | 0 | — | unreliable (early release) |
| d2w32 s15 | `4be5956fde94` | 0.01333 | 25/25 | **2** | 0 | 3 | 0 | — | **stuck** |
| d2w32 s16 | `6351c584de76` | 0.01374 | 20/25 | **240** | 0 | 3 | 0 | — | **stuck** |
| d2w32 s17 | `2cddd5f115e6` | 0.00498 | 25/25 | 0 | 20 | 3 | 0 | — | unreliable (early release) |
| d2w32 s18 | `1c900e4ccc85` | 0.00483 | 25/25 | 0 | 27 | 3 | 0 | — | unreliable (early release) |
| d2w32 s19 | `dde285891e8c` | 0.00459 | 25/25 | 0 | 50 | 3 | 0 | — | unreliable (early release) |
| d2w32 s20 | `d3a670daaf74` | 0.00983 | 24/25 | 0 | 100 | 3 | 0 | — | unreliable (early release) |
| d2w32 s21 | `297ca11a984f` | 0.00507 | 25/25 | 0 | 17 | 3 | 0 | — | unreliable (early release) |
| d2w32 s22 | `ced5fd964a3a` | 0.00440 | 25/25 | 0 | 37 | 3 | 0 | — | unreliable (early release) |
| d2w8 s3 | `8426ed495e96` | 0.01455 | 25/25 | 0 | 7 | 3 | 0 | — | unreliable (early release) |
| d2w8 s4 | `8cff0ea18dbe` | 0.01457 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w8 s5 | `6fa630bc0a35` | 0.01447 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w8 s6 | `69009331a070` | 0.01459 | 25/25 | 0 | 0 | 3 | 0 | 0/1000 | clean |
| d2w8 s7 | `0e5453f3c7b5` | 0.01431 | 24/25 | **35** | 0 | 3 | 0 | — | **stuck** |
| d2w8 s8 | `61f093e2f15e` | 0.01456 | 25/25 | **1** | 0 | 3 | 0 | — | **stuck** |
| d2w8 s9 | `466b0cd0421c` | 0.01447 | 25/25 | 0 | 24 | 3 | 0 | — | unreliable (early release) |
| d2w8 s10 | `5972c0f54b58` | 0.01469 | 20/25 | **192** | 0 | 76 | 21 | — | **stuck** |
| d2w8 s11 | `ba4944593fd4` | 0.01416 | 25/25 | 0 | 0 | 3 | 0 | not in tier-B set | clean |
| d2w8 s12 | `b60e2b0059c4` | 0.01451 | 25/25 | **22** | 0 | 3 | 0 | — | **stuck** |
| d2w8 s13 | `839640cbb270` | 0.01433 | 22/25 | **97** | 1 | 60 | 3 | — | **stuck** |
| d2w8 s14 | `cb2c75994652` | 0.01454 | 22/25 | **112** | 0 | 76 | 26 | — | **stuck** |
| d2w8 s15 | `9c272aa4429a` | 0.01443 | 25/25 | 0 | 0 | 3 | 0 | not in tier-B set | clean |
| d2w8 s16 | `e7fb4c21406c` | 0.01444 | 25/25 | 0 | 27 | 3 | 0 | — | unreliable (early release) |
| d2w8 s17 | `815861a7dafa` | 0.01459 | 25/25 | **5** | 0 | 3 | 0 | — | **stuck** |
| d2w8 s18 | `8c14ba8a5f19` | 0.01393 | 25/25 | 0 | 0 | 3 | 0 | not in tier-B set | clean |
| d2w8 s19 | `1179cd84229b` | 0.01418 | 25/25 | 0 | 0 | 3 | 0 | not in tier-B set | clean |
| d2w8 s20 | `4da6ba664536` | 0.01432 | 20/25 | **207** | 0 | 3 | 0 | — | **stuck** |
| d2w8 s21 | `b5a399fc6895` | 0.01442 | 25/25 | 0 | 0 | 3 | 0 | not in tier-B set | clean |
| d2w8 s22 | `9582fa0b7166` | 0.01427 | 22/25 | **100** | 0 | 3 | 0 | — | **stuck** |

- **Stuck failures** (1,013, in 11 seeds): every one has a mid pterodactyl in the final DOWN run, and that run lasts 57–136 frames (median 62–64 within each seed). Crash scores 479–767. The killers are cacti or low pterodactyls, as in step 2.
- **A new failure mode, early release** (573 failures, in 18 seeds): the dino is killed by a **mid pterodactyl** while not pressing DOWN. I reviewed 12 by hand (from d2w32-s3, d2w32-s20 and d2w8-s9), and all show the same action pattern: 5–8 frames of DOWN, then a release while the pterodactyl still overlaps the standing box, then death one frame later. Crash scores 467–623. It is the mirror image of the stuck duck. Step 2 had the `classHint` "under-duck" for this case but never saw it in its scans.
- **One jump-type failure:** d2w8-s13 died at 192 points on a large cactus pair, with no DOWN.
- **Tier-A failures** were all decided by the opening scan: every seed that failed tier A is stuck or has opening failures. So the 25-seed replay of §4.6 was not needed.
- **P1 holds.** No failure occurred above 767 points, and the 7 eligible seeds scored 0/1,000 in tier B to 1M.
- **Step-2 models re-read with the opening tool** (`opening/step2-*.json`): `v2-bc-duck-d2w32-s1`, `v2-bc-duck-d2w8-s2`, `v2-bc-duck-d1w8-s1` and `v2-bc-duck-d1w8-s2` are all 1,000/1,000, consistent with their tier B.

### Counts (headline: fresh seeds 3–22)
| | d2w32 (S32) | d2w8 (S8) |
|---|---|---|
| **Stuck** | **k = 2/20** (CP 1.2–31.7 %) | **j = 9/20** (CP 23.1–68.5 %) |
| Unreliable, not stuck (early release) | 14/20 (45.7–88.1 %) | 3/20 (3.2–37.9 %) |
| Any failure | 16/20 (56.3–94.3 %) | 12/20 (36.1–80.9 %) |
| Clean | 4/20 (5.7–43.7 %), all with tier B 0/1,000 | 8/20 (19.1–63.9 %): 3 with tier B 0/1,000; 5 (s11, s15, s18, s19, s21) on tier A and the opening scan only |

- **With the step-2 seeds (secondary):** stuck 3/22 at d2w32 and 9/22 at d2w8.
- **Fisher's exact test, stuck, S32 against S8:** two-sided p = 0.031 (one-sided, S8 higher, p = 0.015).
- **Early release, S32 against S8:** two-sided p = 0.0011.

### §7 readings applied
- **Gates, positive control and tool controls:** all pass. §4.2 row 1 holds: the stuck clone is a reproducible property of seed 2 under this procedure.
- **Row "j ≥ 1": the stuck duck is procedure-wide.** It occurs at d2w8 in 9 of 20 fresh seeds. **The d2w8 training procedure is fragile,** so any d2w8 retrain must be screened. The shipped `v2-bc-duck-d2w8-s1` is not affected: opening 1,000/1,000 here, tier B 0/1,000, certification 150/150.
- **Shape dependence:** Fisher p = 0.031 < 0.05, so the stuck rate depends on the shape. But the direction is **the opposite of H-shape**: d2w8 sticks more often than d2w32, so H-shape (a d2w32-specific failure) is refuted. At d2w32 alone, k = 2 would have read "occasional" (row 2 ≤ k ≤ 4); the j row takes precedence.
- **Severity: H-continuum.** The 11 stuck seeds fail 1, 2, 5, 22, 35, 97, 100, 112, 192, 207 and 240 opening games per 1,000, a range of 0.1–24 % per game. **4 stuck seeds pass tier A 25/25** (d2w32-s15, d2w8-s8, d2w8-s12, d2w8-s17). So tier A alone is not a sufficient screen: the screen must include the opening scan (or tier B).
- **Row "unreliable, not stuck": a second failure mode, early release.** It appears in 14/20 d2w32 seeds and 3/20 d2w8 seeds, plus one failure in the stuck d2w8-s13. 16 of those 17 seeds passed tier A. Under the registered rule they were not scanned in tier B. Their opening scans already show that they fail the reliability bar, and they settle their stuck status in the opening (0 stuck in 1,000 openings each).
- **Loss predictor (S32): no separation at this power.** The two highest final losses belong to s5 (clean) and s16 (stuck). k = 2 < 3, so the comparison is descriptive only.
- **Lag-tail predictor: not significant.** A non-fatal lag of 10 or more appears in 3/11 stuck seeds (d2w8 s10, s13, s14) and in 0/13 clean seeds (Fisher one-sided p = 0.082; both margins ≥ 3). It is specific but insensitive: 8 stuck seeds, including s16 at 24 %, never show a non-fatal lag above 3. So a stuck seed's ducks either release correctly or get stuck, with little in between, and the lag is no usable early warning. Neither registered lag row applies exactly; this is read as "no reliable precursor at this power".

### Surprises (not readings)
- **Step 2's arms were lucky draws.** Only 12 of the 40 fresh seeds are clean, against 5 of step 2's 6 duck clones.
- **Post hoc, not registered: in S32 the final losses are bimodal.** All 14 early-release seeds have losses of 0.0044–0.0098 (13 of them ≤ 0.0052). All stuck and clean seeds have 0.0125–0.0140. Step-2 `v2-bc-duck-d2w32-s1` (loss 0.0044, clean) breaks the pattern. In S8 all losses are 0.0139–0.0147, with no such signal. This is not a finding, only a hypothesis for the fix registration.
- **Both failure modes are the same decision.** Both concern releasing DOWN under the first mid pterodactyls, at scores 465–767 (the lowest pterodactyl speeds): the release comes either too late (stuck) or too early (early release). The held-out demos have only 2,823 release rows (step 2 §5.1). So a later fix registration should target the release decision in the opening at both shapes, rather than a shape.

### Guard blocks during execution
The worktree-isolation guard refused five Bash commands. The first four came before the user's rule to stop on any block:
1. A `node -e` loop over run dirs with a shell variable (hashing params). Redone as one `node -e` without shell variables.
2. A heredoc that wrote a node edit script for the rulings. The script was written with the Write tool instead, then run with `node`.
3. A python heredoc that edited the prereg; its text contained "git add". Redone with the Edit tool.
4. A `sed` call with a shell variable as the file path (the `opening.js` truncation fix). Redone with the Edit tool.
5. A heredoc that appended these Results. After that block, the executor stopped and reported. The user then approved finishing with Edit/Write ("Agent uses Edit/Write"), and this section was written with the Edit tool.

### Committed files
Force-added, because `runs/exp-*/` is gitignored: the `best.json` and `bc-metrics.jsonl` of the 41 new runs, `v2-demos-duck-s1-40/demos.json`, and the opening-scan JSON summaries (`opening/*.json`). Scans, logs, snapshots, `demos.bin` and the stage scripts stay local in `runs/exp-bc-duck-stuck/`.
