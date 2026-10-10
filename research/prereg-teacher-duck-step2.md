# Pre-registration: `teacher-duck`, step 2. Can a clone learn to jump *and* duck, as reliably as the jump generation?

Status: **registered (mac-4 review applied), 2026-10-10; step-1 gate passed; results recorded (see Results at the end).** Author: a mac-4 subagent; review: mac-4. Written on main `a53fc30` (core hash `0f329d4e3493…`).
Continues `research/prereg-teacher-duck.md` (step 1). Context: `research/follow-up-questions.md` Q2 (steps 2 and 3 of its "cheapest discriminating sequence"), AGENTS.md "Naming scheme", "Research method", the BC pipeline and the dashboard invariant. The orchestrator's rulings on the draft's open choices are in §11.

**User's order (verbatim):** "To be clear, I want the new solver to use both jump and duck." `teacher-duck` (`test/solver.js` `createDuckSolver`, merged in `9616c3b`) jumps cacti and low pterodactyls, stays grounded under high ones, ducks under mid ones, and falls back to the planned jump where a duck cannot fit. The clone trains on three classes: none (0), jump (1), duck (2).

## 1. Decision
Can a behaviour clone of `teacher-duck`, on the unchanged 44 `observe()` features, play rules v2 to 1M **as reliably as the jump generation** at the same shapes and seeds, **while actually ducking** under mid pterodactyls?

The answer decides two things:
- whether a duck-generation clone becomes a ship candidate. That is step 4 of Q2, a separate decision that needs the user's approval and a certification;
- whether the duck generation needs anything beyond the existing pipeline: more duck data, a different class weighting, or, last, an `observe()` change.

This step ships nothing and certifies nothing (§10).

## 2. Gates (all must hold before anything in §5 runs)
1. **Step 1 passed: OPEN, the only gate not yet met.** `research/prereg-teacher-duck.md` §5, row 2 requires:
   - 0 crashes in its 5,000-seed survey (seeds 40,000,000–40,004,999);
   - fallback < 10 % of mid pterodactyls.
   desktop-1's survey result is due at about 20:10Z. If step 1 lands in its row 3 (0 crashes, fallback ≥ 10 %), this step **waits for the user's ruling** on "ducking enough" (§11).
2. **`bc-gen --teacher duck`: met.** Merged in `6ef29c2`. It provides:
   - label 2 = down key;
   - `teacherName` (`jump` / `duck`) in the manifest and in `best.json` `config`;
   - `recall.duck` in both backends (`trainer/bc.js` `recallOn`; `trainer/bc_train.py`), present only when duck rows exist;
   - defaults byte-identical: jump demos sha256 `3a67bfd4e109d6bb…`, and node and python 1-epoch params unchanged (`2cdbee3` message and `test/bc.test.js`).
   `research/scripts/prereg-bc-sweep/agreement.py` reported none/jump recall only. It now also reports `recallDuck` and `duckPredShare` when the demos contain label 2 (this commit; the jump output is unchanged, §5.3).
3. **Dashboard visibility: met.** Merged in `90f129e`. mac-4 verified that the nested runs `runs/exp-bc-teacher-check/duck(-py)` render with `teacherName` duck, and that `createDuckSolver` is listed in playback. Every run in §5 is nested one level down (`runs/exp-bc-duck/<arm>`), as the AGENTS.md invariant allows.
4. **One CPU-heavy job at a time on the Mac.** The jump sweep is stopped, and its partial report is merged in `8632779`.

**Pilot, seen before registration (not a result, not read).** The `wt/bc-teacher` agent's plumbing check trained a 1-epoch python clone on `bc-gen --teacher duck` demos, `runs/exp-bc-teacher-check/duck-py`, as reported by mac-4: duck recall 0.9998 and validation 25/25 to 1,000 points. It used one epoch and a 1,000-point target, and served only as a plumbing test. It is recorded here because it was seen before the readings were fixed. Nothing in §7 uses it, and no bar was set from it.

## 3. Mechanics and premises (read from code; checked by seconds-scale probes)
**Boxes** (`core.js`). `DINO_X = 50` (line 222).
- **Standing box:** `standBox(y) = {x: DINO_X + 6, y: y + 2, w: 30, h: 38}` (line 247), so x **56–86**. `policy.js` measures obstacle distance from its front edge: `DINO_FRONT = 86`, `out[b+1] = (box.x − DINO_FRONT) / W`.
- **Ducking box:** `{x: dino.x + 4, y: GROUND − DUCK_H + 4, w: 50, h: 18}` (`dinoBox`, line 488), so x **54–104**, y 120–138. Its front edge is **18 px further forward** than the standing box's, and its back edge 2 px further back.
- **Pterodactyl box:** `{x: o.x + 4, y: o.y + 13, w: 36, h: 8}` (line 250), `ys: [108, 82, 58]` (line 230). Low hits both boxes. Mid (box y 95–103) hits only the standing box. High misses a grounded dino.

**Keys.**
- `pressDown` while grounded sets `ducking`; while jumping it fast-drops (`vy = 1`, `speedDrop`) (lines 687–696).
- `releaseDown` clears both (698–702).
- `startJump` is refused while ducking (line 456).
- On landing, `ducking = downHeld` (line 471).
- `applyKeys` releases before it presses (lines 707–713), and the policy's `ACTIONS` never hold both keys. So switching from duck to jump releases DOWN before the jump is pressed. **A "jump refused while ducking" cannot happen through the policy's actions.** The student's analogue is to keep holding DOWN when the teacher would take off (class (ii), §4).

**How `teacher-duck` decides** (`test/solver.js`, `createDuckSolver`).
- The decision is taken on the frame the planned jump over a jump-required obstacle P would take off. If P's box clears the ducking box (mid pterodactyls), the solver ducks, unless:
  1. another takeoff falls inside the duck;
  2. an obstacle that hits the ducking box overlaps it during the duck;
  3. an obstacle not yet spawned could reach it (bounded from speed alone, `MAX_PTERO_OFFSET`).
- DOWN is held over P's **standing-box** overlap, widened by `DUCK_MARGIN = 1` frame each side, and never while airborne.

**Premise P1: is the duck decision a function of `observe()`?** What the student can see:
- **Which pterodactyl to duck:** the height one-hot `out[b+9..b+11]` (`policy.js` line 75) and `out[b+2] = box.y / H` separate mid from low and high exactly.
- **When to press and release:** the onset and release frames are functions of P's x, its velocity (`out[b+12]`) and the speed (`out[4]`). All are observed while P is the nearest obstacle. While ducking, `out[3] = 1`.
- **Release before the next obstacle reaches the wider box:** the next obstacle Q is in slot 1 if it is on screen. The ducking box's reach is a fixed 18 px shift from the standing front, a constant 18/600 = 0.03 offset in the x feature, so the threshold is observable. It is a threshold the jump generation never learned.
- **Where it can fail:** fallback reasons 1 and 2 are computed over `game.obstacles`, which includes obstacles that `placeFairly` parks at x ≥ W. A fallback caused by an off-screen obstacle is **not** observable. Reason 3 depends only on speed and P, so it is observable.

**Premise probe on the training-demo seeds 1–40 (a premise check, not a gate).** It ran on 2026-10-10, before registration, single-threaded, about 1.5 s per run.
- `research/scripts/prereg-teacher-duck-step2/duckdemo-premise.js` plays `createDuckSolver` over the default `bc-gen` episode set: seeds 1–40, levels 0, 2k, 50k, 300k, 900k, span0 3,000, span 5,000, keep-boring 10.
- It re-implements `bc-gen`'s row rule (`trainer/bc.js` `genEpisode`/`isBoring`) and labels DOWN as class 2.
- Seeds 1–40 are **exactly the registered training-demo set.** mac-4 reports that `bc-gen --teacher duck` produced the same 2,062,427 rows, 41,678 of them duck. So this check is not blind, and the registered observability gate runs on the held-out seeds 41–60 instead (§5.1).

| Demos | Rows | none | jump | duck | Boring rows (w 10) | Weighted duck share | duck / jump (weighted) | Mid pterodactyls ducked / met | Fallback | Teacher crashes |
|---|---|---|---|---|---|---|---|---|---|---|
| `teacher-jump`, `runs/exp-bc-e2e/demos.json` | 2,096,268 | 1,856,669 | 239,599 | 0 | 8,114 | 0 | 0 | — | — | 0 |
| `teacher-duck` premise probe | 2,062,427 | 1,817,370 | 203,379 | 41,678 | 11,875 | 1.92 % | 0.205 | 7,244 / 7,244 | **0** | 0 / 200 episodes |

- **Row sha256 of the probe output:** `363872f61a785729…` (full value in §5.1).
- **Duck frames per ducked pterodactyl:** 41,678 / 7,244 = 5.75. Jump rows fall by 36,220, because each duck replaces a longer jump hold.
- **Class weights:** `trainer/bc.js` `classWeights` (sqrt(total / (2 count_c))) gives none 0.751, jump 2.31, **duck 5.10** on these counts. The 2 is a uniform scale (`2cdbee3`). These default weights are what is registered (§11).
- **Fallback:** 0 in the demo distribution. The jump-versus-duck choice at a mid pterodactyl never appears in the demos, so this prediction is tested only in play (§4 (v)).

The observability probe, `research/scripts/prereg-teacher-duck-step2/conflicts.js`, on the same rows, `--grid 1e-3`, weighted:

| Demos | Masked | Irreducible none | Irreducible jump | Irreducible duck | Pair none/duck | Pair jump/duck |
|---|---|---|---|---|---|---|
| `teacher-jump`, seeds 1–40 (baseline) | — | 0.0037 % | 0.0288 % | — | — | — |
| `teacher-jump`, seeds 1–40 (control) | height one-hot + box y | 0.1106 % | **1.7650 %** | — | — | — |
| `teacher-duck` premise, seeds 1–40 | — | 0.0044 % | 0.0266 % | **0.0792 %** | 0.1224 % | 0 (no jump at mid) |
| `teacher-duck` premise, seeds 1–40 (control) | height one-hot + box y | 0.0963 % | 1.9338 % | **6.0248 %** | 7.5651 % | 0 |

- **Exact float matching** (`--grid exact`) finds **no** mixed cells in either file (1.97M and 1.96M cells), so exact identity is uninformative. The grid is what makes the probe bite.
- **Where the conflicts are:** all in grounded frames; airborne frames have 0 at every grid tried. They are takeoff, onset and release frames at sub-pixel resolution. No sign of a hidden variable.
- **The mask control fires:** hiding the height features raises the duck figure 76×, from 0.079 % to 6.02 %. So the probe can detect a decision-relevant variable that the features lack.
- **The §5.1 bars were set after seeing these numbers:** pass ≤ 0.5 %, fail ≥ 2 %, mask control ≥ 10×. That is why the gate runs on different seeds.

**Premise P2: margins.** Under `research/scripts/prereg-bc-sweep/margin.js` (600k frames of own play; minimum gap between the top two of all three logits), the jump-generation baselines are:

| Model | Min top-2 margin | Source |
|---|---|---|
| shipped `model.js` (`m_base_s1`, prototype demos) | 1.01e-3 | `runs/exp-bc-sweep/margins/shipped.json` |
| jump d2-w32 s1 (`bc-gen` demos, equal to `py-s1`) | 1.20e-4 | measured for this pre-registration (2.7 s), `margin.js --model runs/exp-bc-sweep/d2-w32-s1/best.json` |
| jump d2-w8 s1 / s2 | 1.64e-4 / 9.92e-5 | `research/bc-sweep-report.md` §3, seed-2 table |
| jump d1-w8 s1 / s2 | **2.47e-5** / 5.48e-4 | same |

- The CSS-risk flag is < 5.8e-5 (`research/prereg-bc-shapes.md` line 61). **d1-w8 s1 is already flagged in the jump generation**, so arm C carries a CSS risk from the start.
- The decision log's 2.7e-4 for the shipped clone came from a different, non-repo script and is not comparable (sweep report §6).

## 4. Hypotheses and predictions
- **H-learn (expected):** the three-class clone is as reliable as its jump-generation twin, and ducks like the teacher. Predictions:
  - tier A 25/25;
  - tier B 0/1,000 on each seed;
  - held-out duck recall in the range of the jump class (sweep: 99.69–99.84 % at these shapes);
  - the student's duck share of mid pterodactyls ≥ 0.9 × the teacher's on the same seeds.
- **H-jump-instead (covariate shift or class imbalance):** the clone is reliable but often jumps mid pterodactyls instead. Jumping them is the `teacher-jump` plan, and it is still feasible. Predictions:
  - tier B passes;
  - duck share well below the teacher's;
  - low held-out duck recall, with the misses predicted as jump or none at onset.
- **H-unreliable (the third class costs reliability):** new failure modes appear. The prediction is tier-B failures of duck-specific classes; `diag.js --teacher duck` gives each failure a `classHint` (§5.3 step 6):
  - (i) false DOWN under a low pterodactyl or a cactus;
  - (ii) DOWN held through a takeoff the teacher would take. This is the policy-level analogue of a refused jump, which `applyKeys` makes impossible (§3);
  - (iii) DOWN while airborne, an unintended fast-drop into the next obstacle;
  - (iv) a release too late for the wider box (18 px, §3);
  - (v) **untrained fallback states:** a mid pterodactyl where `createDuckSolver` would have fallen back to a jump, because ducking was unsafe (reasons 1–2) or not provably safe (reason 3). The demos contain none of these (0 fallbacks in 7,244), so the student has never seen "mid pterodactyl nearest, but jump", and it will most likely duck there. This covariate-shift prediction is distinct from H-jump-instead and from observability. Its expected frequency comes from step 1's survey (§5.1 step 5).

  Failures of the jump-generation type (a wrong takeoff, as in d8-w8 s1) count against reliability, but not as duck-specific.
- **H-unobservable:** the duck decision is not a function of `observe()`. Prediction: the conflict bars of §5.1 fail on the held-out demos. The premise probe argues against it.

## 5. Design
**Run dirs.** One group, `runs/exp-bc-duck/`, with one arm per dir: `runs/exp-bc-duck/d<d>w<w>-s<seed>`. Demos go in `runs/exp-bc-duck/demos` and `runs/exp-bc-duck/heldout-demos`, the jump twin in `runs/exp-bc-duck/jump-d2w32-s2`, and scans in `runs/exp-bc-duck/scans/`. All runs nest one level down, and the dashboard shows them (§2.3).

**Environment.** Every command runs from the main checkout at the recorded commit, with:
- `PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python`;
- `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`, as in the sweep (`research/prereg-bc-sweep.md` §5).

### 5.0 Tool and plumbing controls (before any duck result is read)
- **C-tools:** `node research/scripts/prereg-teacher-duck-step2/selftest.js` (single-threaded, about 4 s; it passed on this commit) checks:
  1. the `duckshare.js` counter control on the duck teacher;
  2. the `duckshare.js` negative control: the jump clone d2-w8-s1 meets the same 1,625 duckable pterodactyls as the teacher, ducks none of them and jumps all of them;
  3. `diag.js` with the default teacher reproduces the sweep's recorded `runs/exp-bc-sweep/diag/d8-w8-s1.json` byte for byte;
  4. `diag.js --teacher duck` runs and emits a `classHint`.
- **C-jump-sha:** `bc-gen` (default teacher) into a scratch dir must reproduce the `runs/exp-bc-e2e` demos byte for byte, sha256 `3a67bfd4e109d6bbee7a6eb21b6fa7922eb254cc0fc92dc7695538a222bcfd09` (`runs/exp-bc-e2e/demos.json`).
- **C-jump-params:** `bc-train --backend python --seed 1 --hidden 32,32` on those demos must give params equal to `runs/exp-bc-e2e/py-s1`'s. This is the sweep's harness control (`research/bc-sweep-report.md` §2), and it shows that the three-class code path, with a duck count of 0, is unchanged.
- `test/bc.test.js` (`6ef29c2`) already pins the jump sha and the 1-epoch params. The two checks above repeat it at full scale and take seconds.

### 5.1 Demos and the observability gate (static; no training)
1. **Training demos:** `bc-gen --teacher duck --dir runs/exp-bc-duck/demos` with the defaults (seeds 1–40, default levels). Report:
   - rows and weighted counts per class;
   - duck share and the duck/jump ratio;
   - the sha256.

   **C-duck-rows:** the rows are expected to be **byte-identical** to the premise probe's (sha256 `363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7`, 2,062,427 rows), because the probe re-implements `bc-gen`'s row rule. A mismatch is not fatal by itself, but nothing is trained until it is explained, for example by a different label rule or boring rule.
2. **Held-out demos:** `bc-gen --teacher duck --seeds 41-60 --dir runs/exp-bc-duck/heldout-demos`. These are disjoint from the training demos, and they are the sweep's held-out range (`research/bc-sweep-report.md` §1). **No probe has seen them.** The jump-generation held-out demos for the same seeds are at `runs/exp-bc-sweep/heldout-demos`, and they form the baseline.
3. **Observability gate, on the held-out demos only:**
   - `node research/scripts/prereg-teacher-duck-step2/conflicts.js --demos runs/exp-bc-duck/heldout-demos --grid 1e-3`;
   - the same with `--grid 1e-2`;
   - the mask control `--mask 7,14,15,16,20,27,28,29,33,40,41,42` (box y and the height one-hot of all three slots) at 1e-3;
   - the baseline: the same commands on `runs/exp-bc-sweep/heldout-demos` (teacher-jump, seeds 41–60).

   Bars, fixed now (they were chosen after the seeds 1–40 premise check, §3), at grid 1e-3, split `all`:
   - **pass:** irreducible duck ≤ 0.5 % **and** pair none/duck ≤ 1 %. If any jump weight occurs with a mid pterodactyl nearest, pair jump/duck ≤ 1 % is also required;
   - **fail:** irreducible duck ≥ 2 %, or either pair ≥ 5 %;
   - **ambiguous:** anything in between. Proceed, and every reading in §7 is conditional on it;
   - **probe validity:** the mask control must raise irreducible duck at least 10× over the unmasked figure. Otherwise the probe is invalid, and nothing is read from it.

   The 1e-2 grid is reported, not barred. It merges cells about 6 px wide, so it overstates conflicts. The same statistics on the training demos (seeds 1–40) are reported descriptively, not as a gate.
4. **Fallback visibility (only if step 1's survey reports fallbacks > 0):** a replay probe, written then and registered as an amendment before it runs. For each fallback, it records whether the blocking obstacle was on screen (x < W) at the decision frame. A fallback blocked only by off-screen obstacles is by construction not learnable from `observe()`. Report the share. If it exceeds 10 % of fallbacks, record it as a known ceiling on duck share, not as a student failure. If step 1 reports 0 fallbacks, this item is void.
5. **Fallback coverage (from step 1, before any scan):** convert step 1's counts into expected untrained states per tier-B scan: fallbacks per full game, by `fallbackWhy` reason, × 1,000 games. Measured: 8,176 duckable pterodactyls in one full teacher game (seed 30,003,000, `duckshare.js`). So even a 0.1 % fallback share means about 8 per game, or about 8,000 per scan. Count the fallbacks in `runs/exp-bc-duck/demos` too (premise: 0). Record both numbers before the scans run, so that §7's fallback row is read against a fixed expectation. Whether to add demos that contain fallbacks is **decided after step 1** (§11).
6. **Release timing (descriptive):** report the none/duck pair in the `ducking` split (release frames; premise value 0.48 %). Also report the distribution of the slot-1 obstacle's x feature on release frames, against the 0.03 offset (§3).

### 5.2 Clones
| Arm | Shape | Seeds | Why | Jump-generation twin: same shape and seed, `runs/exp-bc-sweep/` (tier B on seeds 30,003,000–30,003,999) |
|---|---|---|---|---|
| **A** | d2-w32 (44-32-32-3, the shipped shape) | 1, 2 | shipped shape | `d2-w32-s1` (0/1,000); s2: **new twin** `runs/exp-bc-duck/jump-d2w32-s2` (below) |
| **B** | d2-w8 (44-8-8-3) | 1, 2 | the small arm: 0/1,000 on both seeds | `d2-w8-s1`, `d2-w8-s2` (each 0/1,000) |
| **C** | d1-w8 (44-8-3) | 1, 2 | the smallest shape in the grid, 0/1,000 on both seeds; jump s1 margin already CSS-flagged | `d1-w8-s1`, `d1-w8-s2` (each 0/1,000) |

- **Source of the shapes:** `research/bc-sweep-report.md` §3/§4 (merged `8632779`), where d1-w8 and d2-w8 scored 0/1,000 in tier B on both seeds 1 and 2.
- **Excluded shapes:** d8-w8 s1 scored 2/1,000 (seeds 30,003,071 at 47 points and 30,003,596 at 70), and d4-w8 s2 stopped after 360 games.
- I checked the twins' scan files directly (`runs/exp-bc-sweep/scans/*.jsonl`): each header has `base: 30003000`, `n: 1000`, with 1,000 seed rows and a footer. `runs/exp-bc-sweep/summary.json` is stale for tier B and is not used.

**Training.** `bc-train --backend python --hidden <h> --seed <s> --demos runs/exp-bc-duck/demos --snapshots --wasm --dir runs/exp-bc-duck/d<d>w<w>-s<s>`:
- the shipped hyperparameters (15 epochs, batch 1024, lr 2e-3);
- the default class weights;
- `--snapshots`, so that held-out agreement can be computed per epoch as in the sweep. It does not change training (`research/bc-sweep-report.md` §2).

**Jump twin for A-s2:** `bc-train --backend python --seed 2 --hidden 32,32 --demos runs/exp-bc-e2e --snapshots --wasm --dir runs/exp-bc-duck/jump-d2w32-s2`, then a scan like the others.

### 5.3 Evaluation (paired with the jump generation)
For each duck clone, and for the jump twin where marked:
1. **Tier A:** `bc-train`'s own validation, 25 validation seeds to 1M (`trainer/env.js` `VALIDATION_SEEDS`). The jump twins are all 25/25 (sweep report §3).
2. **Tier B** (also for the jump twin):

   `node research/scripts/prereg-bc-robustness/scan.js --model <arm>/best.json --out runs/exp-bc-duck/scans/<arm>.jsonl --base 30003000 --n 1000 --target 1000000 --wasm`

   - **The seeds are 30,003,000–30,003,999, the same 1,000 as the sweep's tier-B scans** (`research/prereg-bc-sweep.md` §5; scan headers `base: 30003000`), so each comparison is per seed.
   - `scan.js` asserts core hash `0f329d4e`. `test/solver.js` is not covered by the hash.
   - Contiguity check, as in the `prereg-bc-pipeline-e2e.md` Results: 1,000 seed rows, contiguous, footer present.
3. **Duck recall:**
   - train-set `recall.duck` from `bc-metrics.jsonl`;
   - **held-out** weighted agreement and per-class recall on `runs/exp-bc-duck/heldout-demos`, per epoch snapshot, with `research/scripts/prereg-bc-sweep/agreement.py`. It now reports `recallDuck`, plus `duckPredShare`: the weighted share of true-duck rows predicted none, jump or duck. The extra keys appear only when the demos contain label 2.

   Sanity check on this commit: the jump clone d2-w8-s1's epoch-15 snapshot, on the premise demos, gives recallDuck 0 and duckPredShare [0, 1, 0], meaning it jumps every duck frame.
4. **Duck share in play:** `node research/scripts/prereg-teacher-duck-step2/duckshare.js --model <arm>/best.json --base 30003000 --n 10 --out runs/exp-bc-duck/duckshare/<arm>.json`, and once with `--teacher duck` as the reference. It uses the first 10 tier-B seeds, full games to 1M, and the `policy.js` `createAgent` JS forward pass.
   - **Duckable obstacles** are those whose box overlaps the standing box vertically but clears the ducking box, computed from the boxes.
   - Each one is classified over the frames during which its box overlaps the standing box's x-range [56, 86), using the state after each step:
     - *ducked*: grounded and ducking on every such frame;
     - *jumped*: airborne on every such frame;
     - *mixed*: anything else;
     - *crashed*: the game ended during the overlap.
   - **Duck share** = ducked / (ducked + jumped + mixed + crashed). It is reported pooled and per game.
   - Also counted: DOWN frames and DOWN-while-airborne frames (fast-drops, which the teacher never makes).
   - **Counter control, built in** (the script exits 1 if it fails), on the teacher:
     - DOWN frames = `act.stats.downFrames`;
     - DOWN-airborne = 0;
     - mixed = crashed = 0;
     - ducked ≤ `stats.midDucked` ≤ ducked + pending.
   - **Sample size:** about 8,000 duckable pterodactyls per full game (8,176 measured), so 10 games give about 80k per model.
5. **Margins:** `margin.js --model <arm>/best.json` (also for the jump twin), compared with the twin's figure (§3 P2) and with the 5.8e-5 flag.
6. **Failures:** run `node research/scripts/prereg-bc-robustness/diag.js --teacher duck --arm <arm> --scan <scan.jsonl> --model <arm>/best.json --out <diag.json>` on every tier-B failure. `--teacher duck` was added on this commit; the default output is unchanged (C-tools 3). With duck, the shadow reference is `createDuckSolver`. Each record gains:
   - `fallbackFrames`: frames where the shadow fell back from a duck to a jump, with the reason;
   - `downAirFrames`: model DOWN while airborne, total and inside the near-fatal window;
   - a heuristic `classHint`, checked in a fixed order:

     | Order | `classHint` | Condition |
     |---|---|---|
     | 1 | (iii) | DOWN airborne inside the near-fatal window |
     | 2 | (v) | the shadow fell back inside the window |
     | 3 | (ii) | the last effective disagreement was model duck, teacher jump |
     | 4 | (iv) | the dino crashed while ducking, into a non-mid obstacle, after ducking a mid pterodactyl in the same duck |
     | 5 | (i) | the dino crashed while ducking, with no mid pterodactyl in that duck |
     | 6 | under-duck | killed by a mid pterodactyl |
     | 7 | jump-type | none of the above |

   Every hint is reviewed by hand before a failure is counted as duck-specific. The shadow forms its duck plan on the model's own trajectory, so its ground answers are the teacher's answers given the model's history.

   Hint validity on known cases: the jump-generation d8-w8-s1 deaths (seeds 30,003,071 and 30,003,596) each contain one DOWN action while airborne (`last30`: `jumpx4 duckx1`), and they get hint (iii). This is a tool check only, and it does not reclassify the sweep's verdicts.
7. **CSS check:** `test/browser-check.js` on the Mac, only for a clone the orchestrator puts forward for shipping (§10), with its `model.css` on a scratch page as in `prereg-bc-shapes.md`. It is not part of this step's readings.

### 5.4 Positive controls (summary)
| Control | Must show | If not |
|---|---|---|
| C-tools (§5.0) | 4/4 ok | Fix the tools; read nothing they produce |
| C-jump-sha, C-jump-params (§5.0) | Byte- and param-identical | STOP: the plumbing changed the jump generation |
| C-duck-rows (§5.1) | sha `363872f6…` | Explain the difference before training |
| Probe validity (§5.1) | Mask control ≥ 10× unmasked, on the held-out demos | Probe invalid; read nothing from it |
| Tier-B harness | The jump twins' 0/1,000 under the same runner and seeds (sweep; not re-run). `jump-d2w32-s2` should also score 0/1,000 | A jump-twin failure is not a duck result. Diagnose it, and pair against the twin as it stands (general McNemar form, §6) |
| duckshare counter (§5.3) | Built-in control passes on the teacher | Counter invalid |

## 6. Power
**Clopper–Pearson upper bounds for 0 failures.** Formula: 1 − α^(1/n), with α = 0.05 for the one-sided bound and 0.025 for the upper end of the two-sided interval. These values match `prereg-bc-pipeline-e2e.md` Results step 3 and the mac-4 handoff §6.

| Games | One-sided 95 % | Two-sided 95 % (upper end) |
|---|---|---|
| 25 (tier A) | 11.3 % | 13.7 % |
| 1,000 (tier B, one seed) | 0.299 % | 0.368 % |
| 2,000 (two seeds pooled) | 0.150 % | 0.184 % |

**Paired comparison: an exact McNemar test** on the per-seed pass/fail pairs (duck clone against jump twin), with one-sided H1 "the duck clone fails more".
- Let b be the number of seeds where only the duck clone fails, and c the number where only the twin fails. In general, one-sided p = P(X ≥ b), with X ~ Binomial(b + c, ½).
- Every existing twin scored 0/1,000, so c = 0 and p = 0.5^b. If `jump-d2w32-s2` fails any seed, use the general form.
- With c = 0:
  - b = 4 gives p = 0.0625;
  - **b = 5 is the smallest significant count (p = 0.031)**;
  - a two-sided test needs b = 6 (p = 0.031).
- The test is applied per seed, and per arm pooled over its two training seeds (2,000 pairs). Pooling reuses the same 1,000 game seeds, so the pooled test treats the two training seeds as independent replicates.

**Power of the paired test** (one-sided, b ≥ 5), for a true duck-clone failure rate p against a twin at about 0:
- 1,000 pairs: 80 % at p ≈ 0.7 %, 56 % at 0.5 %, 5 % at 0.2 %;
- 2,000 pairs: 90 % at 0.4 %, 37 % at 0.2 %.

**So a tier-B comparison cannot show equivalence.** A "no significant difference" result only excludes excess failure rates of about 0.4 % or more (pooled).

**The reliability bar is stricter than significance**, as in the jump generation: **0 failures per seed** (`prereg-bc-pipeline-e2e.md` §4).
- A single failure detects p = 0.2 % with 86 % probability at 1,000 games.
- One to four failures fail the bar, but they are "not significantly worse than the jump twin" and are reported that way.

**Duck share:** about 80k duckable pterodactyls per model put its binomial uncertainty below 0.5 percentage points. So the 0.9× and 0.5× bars (§7) are limited by game-to-game variation, not by count. It is reported per game as well.

## 7. Readings (fixed in advance)
**Definitions.**
- **Reliable:** tier A 25/25, and tier B 0/1,000 on **both** seeds of the arm.
- **Ducks:** duck share ≥ 0.9 × the teacher's on the same seeds.
- **Rarely ducks:** duck share < 0.5 × the teacher's.

| Result | Reading | Next |
|---|---|---|
| Step 1 not passed, or step 1 in its row 3 | Not run | Wait for step 1, or for the user's ruling (§11) |
| C-tools, C-jump-sha or C-jump-params fails | The tools or the plumbing are wrong | STOP. Fix them before anything else |
| Probe invalid (mask control < 10×) | The observability gate says nothing | Fix the probe. Read no observability row |
| Observability **fails** on the held-out demos (§5.1 bars) | **The duck decision is not a function of `observe()`** at this resolution | **Stop and report.** Do not train. Name the conflicting states. An `observe()` change needs a clear gain and the user's say (memory "policy.js features"; the AGENTS.md anti-cheat rules), so it is a separate decision |
| Arm reliable **and** ducks | **H-learn holds** for that shape (scope §10) | Report it as a potential ship candidate. Shipping needs §10 |
| Arm reliable but rarely ducks | **H-jump-instead:** the student found the jump-generation plan. Reliability is intact, the behaviour is not delivered | Report the onset confusion (`duckPredShare`: duck→jump against duck→none). Candidate next steps, each registered separately: a class-weight change, more duck data, a DAgger-style relabel on the student's own states |
| Arm reliable, duck share between 0.5× and 0.9× | Partial ducking | Report it. The user decides whether it is enough |
| Arm has ≥ 1 tier-B failure, or tier A < 25/25 | **Reliability bar not met** at that shape. Pooled b ≥ 5 (exact McNemar): **significantly worse than the jump twin** (one-sided p ≤ 0.031). b = 1–4: not significant | Classify each failure (§5.3 step 6). If most are duck-specific (i)–(iv): **the third class costs reliability** (H-unreliable). If most are jump-type: the usual clone-death rate; compare with the sweep. No ship candidate from that arm either way |
| Tier-B failures concentrated at teacher-fallback situations ((v), §5.3 step 6) | **The demo distribution lacks fallback states.** This is not an observability failure, and not a reason to change `observe()` | Next, registered separately: more or longer demos that contain fallbacks, or a DAgger-style relabel on the student's states |
| Arm C (d1-w8) reliable and ducks, but margin < 5.8e-5 | As above, plus a **CSS-divergence risk flag** (its jump twin s1 is already flagged at 2.47e-5) | Any shipping step must include the CSS agreement check |
| Any arm's min margin < 0.5 × its jump twin's, or < 5.8e-5 | **The third class thins margins** at that shape | Flag it for CSS fidelity. It is not a reliability failure |
| Held-out duck recall < 99 % while the arm is reliable and ducks | Demonstration agreement is not the bottleneck (as in the sweep's T2 row) | Note only |
| Arms disagree (e.g. A passes, B fails) | Shape-dependent | Report per arm. The smallest arm that is reliable and ducks is the one to report as a ship candidate |

## 8. What would reopen a null
- **A "rarely ducks" or "unreliable" result** is scoped to:
  - the default class weights, 15 epochs and the shipped learning rate;
  - the `bc-gen` defaults (seeds 1–40; duck about 1.9 % of the weight);
  - these shapes, and training seeds 1–2.

  It reopens with a registered change to any of these.
- **An observability failure** is scoped to the 1e-3 grid on held-out seeds 41–60. It reopens with a finer test, for example an exact match on a subset with many repeated states, or a per-feature conflict attribution.

## 9. Cost (from repo numbers; Mac, one heavy job at a time)
| Item | Count | Each | Source | Total |
|---|---|---|---|---|
| C-tools self-test | 1 | about 4 s | measured on this commit | seconds |
| `bc-gen` (training and held-out demos) | 2 | about 0.6 s on 18 workers; the premise probe took 1.3 s single-threaded | `prereg-bc-pipeline-e2e.md` Results step 1; §3 | seconds |
| Plumbing controls | 1 gen + 1 train | 0.6 s + 19 s | `prereg-bc-pipeline-e2e.md` Results step 2 (Python job wall 19 s) | < 1 min |
| Observability probes | about 8 runs | 1.4–2.4 s on 2.1M rows (about half for 1M held-out rows) | §3 | < 1 min |
| `bc-train` python, d2-w32 | 2 + 1 jump twin | 14.6 s train + 4.4 s validation, unloaded | `prereg-bc-pipeline-e2e.md` Results step 2 | about 1 min |
| `bc-train` python, d2-w8 / d1-w8 | 2 + 2 | 28.8–42.6 s / 25.3–32.0 s train wall, under parallel load | `research/bc-sweep-report.md` "Secondary" | about 2.5 min |
| Tier-B scans, d2-w32 | 2 + 1 twin | 176 s unloaded to 294 s loaded | `prereg-bc-pipeline-e2e.md` step 3; `runs/exp-bc-sweep/scans/d2-w32-s1.log` | 9–15 min |
| Tier-B scans, d2-w8 / d1-w8 | 2 + 2 | 124–139 s / 107–112 s | `runs/exp-bc-sweep/scan-queue.log` | about 8 min |
| `margin.js` | 7 | 2.7 s at d2-w32 | measured for this pre-registration | < 1 min |
| `duckshare.js` | 7 models + teacher, × 10 games | teacher about 0.9 s per full game (measured); a model 1–3 s per game (JS forward pass) | measured on this commit | about 3–5 min |
| `agreement.py` | 7 runs × 15 snapshots | seconds per run | sweep report | about 1 min |

**Total: about 25–40 min of Mac time**, dominated by the scans. A duck clone has the same MACs as its twin and plays the same 1,943,237 frames per full game (`prereg-bc-pipeline-e2e.md` step 3). Ducking does not change game length, so scan costs should match the twins'.

## 10. Scope, and what this step does not do
- **Scope:**
  - rules v2, full games from 0;
  - the 44 unchanged features;
  - `teacher-duck` as merged in `9616c3b`: ducking under mid pterodactyls only, no fast-drop;
  - the Python backend and the shipped hyperparameters;
  - demos from seeds 1–40 with the `bc-gen` defaults;
  - training seeds 1 and 2.
- **Tier-B seeds are not held out. Nothing here is a certification.**
- **Shipping is out of scope.** It would need, in order:
  1. the user's approval;
  2. certification on fresh held-out offsets, paired with the shipped jump-generation clone. **The next fresh offset is 325** (325–474 for 150 seeds). The sweep spent none (`research/bc-sweep-report.md` §6), and `runs/bc-v2-h32x32/evals.jsonl` ends at offsets 175–324;
  3. `export-css` and `test/browser-check.js` with 0 CSS/JS disagreements;
  4. an AGENTS.md decision-log entry.
- **Never** `session`, `finish` or `--auto-export`, and never write the root `model.js`/`model.css` in this step.
- The results record the code commit, the core hash, the demo sha256 values and step 1's result commit.

## 11. Rulings on the draft's open choices (mac-4, 2026-10-10)
1. **Layout and dashboard:** nested `runs/exp-bc-duck/<arm>`. The gate is met (`90f129e`).
2. **Class weights:** the defaults. Nothing else is registered.
3. **Fallback visibility:** a replay probe, only if step 1 reports fallbacks > 0 (§5.1 step 4).
4. **Jump twin for d2-w32 s2:** included.
5. **Arm C (d1-w8):** included.
6. **`duckshare.js`:** written now, with its counter control. 10 seeds.
   - mac-4's estimate was about 180 mid pterodactyls per seed. The measured count is about 8,000 per full game, so 10 seeds is more than enough.
7. **`diag.js` for three actions:** extended now, with `--teacher duck` and the `createDuckSolver` reference. The default output is byte-identical.
8. **Duck-share bars:** kept at 0.9× and 0.5×.
9. **Observability bars:** kept (pass ≤ 0.5 %, fail ≥ 2 %, mask control ≥ 10×). The gate moves to held-out seeds 41–60, because the bars were set after seeing seeds 1–40.
10. **Extra demos with fallbacks:** decided after step 1, once its fallback counts are known (§5.1 step 5).
11. **Step 1 in its row 3** (fallback ≥ 10 %): wait for the user.

## Results (2026-10-10, mac executor, branch `wt/exp-bc-duck`)
**Provenance.** Code: main `5fd8248`, with no change to `core.js`, `policy.js`, `test/solver.js` or `trainer/`. This branch adds only the analysis helpers `research/scripts/prereg-teacher-duck-step2/{release,pair,duckrun}.js`. Core hash `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e` (every scan header). Step-1 result commit `5fd8248`. Python backend `/Users/kyle/Desktop/dino-game/.venv/bin/python` with `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`. Runs from the worktree `.claude/worktrees/agent-a0e89a1383966479d`, under `runs/exp-bc-duck/<arm>`; the dashboard (`--port 8095 --runs <worktree>/runs`) lists all seven arms and both demo dirs, the arms with `teacherName` duck. One heavy job at a time. No held-out eval offsets spent; no `export`/`finish`/`session`/`--auto-export`; root `model.js`/`model.css` untouched.

### §2 gates
All met: step 1 passed (0/5,000 crashes, fallback 0/41,372,227, `5fd8248`); `6ef29c2`; `90f129e`; no other heavy job running. **Step 1 reports 0 fallbacks, so §5.1 step 4 (the fallback-visibility replay, OPEN-3) is void and §11 item 10 (extra fallback demos, OPEN-10) is moot. Neither was run.**

### §5.0 tool and plumbing controls: all pass
| Control | Result |
|---|---|
| C-tools (`selftest.js`) | 4/4 ok: teacher 1,625/1,625 ducked, 9,396 DOWN frames; jump clone d2-w8-s1 1,625/1,625 jumped, 0 ducked; diag default byte-identical; diag `--teacher duck` gives `(iii)` on 30,003,071 and 30,003,596 |
| C-jump-sha | `bc-gen` with the default teacher: 2,096,268 rows, sha256 `3a67bfd4e109d6bbee7a6eb21b6fa7922eb254cc0fc92dc7695538a222bcfd09`, byte-identical to `runs/exp-bc-e2e` |
| C-jump-params | `bc-train --backend python --seed 1 --hidden 32,32` on those demos: all 2,595 params identical to `runs/exp-bc-e2e/py-s1` (validation 25/25) |
| C-duck-rows | `bc-gen --teacher duck`: 2,062,427 rows, sha256 `363872f61a785729820c3729eda3c6261a91b19fc32598baf620679ee0149ea7`, identical to the premise probe |

### §5.1 demos and the observability gate
- **Training demos** (`runs/exp-bc-duck/demos`, seeds 1–40):
  - rows: none 1,817,370, jump 203,379, duck 41,678;
  - weighted: 1,924,245 / 203,379 / 41,678; weighted duck share 1.921 %, duck/jump 0.205;
  - teacher crashes 0; fallbacks 0 of 7,244 mid pterodactyls (`duckdemo-premise.js` on the same rows).
- **Held-out demos** (`runs/exp-bc-duck/heldout-demos`, seeds 41–60):
  - 1,031,320 rows: none 908,229, jump 101,876, duck 21,215; weighted none 961,545;
  - weighted duck share 1.956 %; sha256 `814c0686107f9a5933a668a56260adac908b171cf37528c3d3bbb749297ae4ea`;
  - the jump-generation baseline is `runs/exp-bc-sweep/heldout-demos` (sha `dcf49f2f…`).
- **Gate** (`conflicts.js`, weighted, split `all`; outputs in `runs/exp-bc-duck/obs/`):

| Demos | Grid | Mask | Irred. none | Irred. jump | Irred. duck | Pair n/d | Pair j/d |
|---|---|---|---|---|---|---|---|
| duck held-out 41–60 | 1e-3 | — | 0.0033 % | 0.0304 % | **0.0424 %** | **0.0801 %** | 0 (no jump weight with a mid pterodactyl nearest) |
| duck held-out 41–60 | 1e-3 | height + box y | 0.0806 % | 1.7767 % | **5.5951 %** | 6.7122 % | 0 |
| duck held-out 41–60 | 1e-2 | — | 0.0762 % | 0.3131 % | 1.4377 % | 1.9703 % | 0 |
| jump held-out 41–60 (baseline) | 1e-3 | — | 0.0026 % | 0.0332 % | — | — | — |
| jump held-out 41–60 (baseline) | 1e-3 | height + box y | 0.0922 % | 1.7231 % | — | — | — |
| jump held-out 41–60 (baseline) | 1e-2 | — | 0.0769 % | 0.3075 % | — | — | — |
| duck training 1–40 (descriptive) | 1e-3 | — | 0.0044 % | 0.0266 % | 0.0792 % | 0.1224 % | 0 |

  - **The probe is valid:** the mask raises irreducible duck 132× (bar ≥ 10×).
  - **Observability PASSES:** 0.0424 % ≤ 0.5 % and 0.0801 % ≤ 1 %. The j/d condition is vacuous. The airborne split has 0 conflicts.
- **Fallback coverage (§5.1 step 5), fixed before the scans.** Step 1 had 0 fallbacks in 5,000 full games, so the expected number of untrained fallback states per 1,000-game tier-B scan is **0**. Demos: 0/7,244.
- **Release timing (§5.1 step 6), descriptive.**
  - Pair none/duck in the `ducking` split: 0.2905 % held-out (training 0.4804 %).
  - `release.js` counts the rows with the ducking flag and label none: 2,823 held-out release rows. Slot 1 is present on only 6 of them, with x feature 0.786–0.830 (none below 0.03 or 0.10). The slot-0 x median is −0.120.
  - So the release is cued by the pterodactyl leaving the standing box, and the 18 px threshold of the wider box **never binds in the demos**. This is a coverage note for class (iv), consistent with 0 fallbacks.

### §5.2–5.3 clones (15 epochs, batch 1,024, lr 2e-3, default class weights, `--snapshots --wasm`)
**Tier B** ran on seeds 30,003,000–30,003,999 to 1M with `scan.js --wasm`. Every scan file (the duck arms, the new twin and the sweep twins) passed the contiguity check: 1,000 contiguous seed rows, header base 30,003,000, footer present.

**Clopper–Pearson bounds:** 0/1,000 gives a one-sided 95 % upper bound of 0.299 %; 0/2,000 pooled, one-sided 0.150 %; 25/25, one-sided 11.3 %. Intervals for non-zero counts are two-sided 95 %.

**McNemar:** exact and one-sided (H1: the duck clone fails more), per seed against its twin and pooled per arm (2,000 pairs).

| Arm | Final train loss | Tier A | Tier B failures | Held-out recallDuck (ep 15) | duckPredShare [none, jump, duck] | Train recall.duck | Duck share in play (10 games; teacher 1.000) | DOWN frames / teacher's | Min margin | Twin min margin | McNemar b / c, p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A d2w32-s1 | 0.0044 | 25/25 | **0/1,000** | 0.99995 | [4.7e-5, 0, 0.99995] | 1.0000 | **1.000** (82,614/82,614; 10/10 games) | 1.026 | 2.04e-3 | 1.20e-4 (sweep d2-w32-s1) | 0 / 0, 1 |
| A d2w32-s2 | 0.0133 | **20/25** (min 516) | **204/1,000** (20.4 %, CP 17.9–23.0 %) | 1.0000 | [0, 0, 1] | 1.0000 | 1.000 (65,949/65,949; 8 full games; 2 crashed, at 572 and 502) | — | 4.82e-4 | 2.79e-5 (new twin) | 204 / 0, 3.9e-62 |
| B d2w8-s1 | 0.0141 | 25/25 | **0/1,000** | 1.0000 | [0, 0, 1] | 1.0000 | **1.000** (10/10) | 1.128 | 5.61e-4 | 1.64e-4 | 0 / 0, 1 |
| B d2w8-s2 | 0.0141 | 25/25 | **0/1,000** | 1.0000 | [0, 0, 1] | 1.0000 | **1.000** (10/10) | 1.127 | 3.58e-4 | 9.92e-5 | 0 / 0, 1 |
| C d1w8-s1 | 0.0190 | 25/25 | **0/1,000** | 0.99981 | [1.9e-4, 0, 0.99981] | 1.0000 | **1.000** (10/10) | 1.151 | 1.82e-4 | 2.47e-5 | 0 / 0, 1 |
| C d1w8-s2 | 0.0183 | 25/25 | **0/1,000** | 1.0000 | [0, 0, 1] | 1.0000 | **1.000** (10/10) | 1.146 | **7.57e-5** | 5.48e-4 | 0 / 0, 1 |
| twin jump-d2w32-s2 | 0.0057 | 25/25 | **0/1,000** | 0 | [0, 1, 0] | — | 0 (82,614 jumped) | 0 | **2.79e-5** | — | — |

- **Pooled per arm:**
  - **A: b = 204, c = 0, p = 3.9e-62** (204/2,000, CP 8.9–11.6 %);
  - B: b = 0, c = 0, p = 1 (0/2,000, ≤ 0.150 %);
  - C: b = 0, c = 0, p = 1 (0/2,000, ≤ 0.150 %).
- **Teacher reference** (`duckshare.js --teacher duck`, 10 games): 82,614/82,614 ducked, 431,019 DOWN frames, counter control passes. No arm ever pressed DOWN while airborne (downAir 0 everywhere), and no arm had a duckable obstacle classed mixed or crashed.
- **Tier-B harness control:** the new jump twin `jump-d2w32-s2` scored 0/1,000 and 25/25, so A-s2's failures are not a harness artefact.

**A-s2 failures (§5.3 step 6), from `diag.js --teacher duck` and a hand review** (`runs/exp-bc-duck/diag/d2w32-s2.json`, and `…-duckrun.json` from `duckrun.js`):
- All 204 get `classHint` **(ii) duck-through-takeoff**.
  - In every one, the last effective disagreements are model duck against teacher jump, and the dino dies while ducking (`last30` is `duckx30` in all 204).
  - fallbackFrames 0 and downAir 0. The shadow solver passes all 204 seeds.
  - Killers: cacti in 201 (small×2 44, small×3 37, small 35, large×2 33, large 30, large×3 22), a low pterodactyl in 3. Scores 479–767, median 562.
- **Hand review:** in **every** case the final DOWN run began with a **mid pterodactyl nearest** and passed under it, then **never released**. DOWN stayed held for 57–134 frames (median 63) into the next obstacle; the teacher's duck lasts about 6 frames.
  - So these are a duck-specific stuck-duck (late-release) mode. The fixed hint order classes it (ii), and its content also matches (iv).
  - All 204 are duck-specific. None is jump-type or (v).

### §7 readings applied
- **Gates and controls:** all pass. Observability **passes** and the probe is valid, so training went ahead.
- **Arm A (d2w32): reliability bar not met** (row "≥ 1 tier-B failure, or tier A < 25/25").
  - Pooled b = 204 ≥ 5: **significantly worse than the jump twin** (one-sided p = 3.9e-62).
  - All the failures are duck-specific, so **the third class costs reliability (H-unreliable) at d2w32**. Training seed 2 drives it.
  - **No ship candidate from arm A.** Seed 1 alone is reliable and ducks, but the arm reading needs both seeds.
  - "Margins" row: neither seed is below 0.5× its twin or below 5.8e-5.
- **Arm B (d2w8): reliable and ducks, so H-learn holds** for d2w8 (scope §10). Its margins are 3.4× and 3.6× the twins'. No flag.
- **Arm C (d1w8): reliable and ducks, so H-learn holds** for d1w8.
  - The arm-C CSS row (margin < 5.8e-5) does **not** apply: 1.82e-4 and 7.57e-5.
  - **The "Margins" row applies to d1w8-s2:** 7.57e-5 is below 0.5× its twin's 5.48e-4, so **the third class thins margins** at that shape and seed. Flag it for CSS fidelity. It is not a reliability failure.
- **Fallback row ((v)):** does not apply. 0 fallback states were expected and 0 observed.
- **"Held-out duck recall < 99 %" row:** does not apply (all ≥ 99.98 %).
- **"Arms disagree": the result is shape-dependent.**
  - The smallest arm that is reliable and ducks is **C (d1w8, 44-8-3)**. It is the potential ship candidate, with the d1w8-s2 margin flag. B (d2w8) is next.
  - Shipping still needs §10: the user's approval, certification from offset 325, `export-css` plus the browser check, and a decision-log entry.

### Surprises (not readings)
- **A-s2 is the shipped shape, but its training converged to a 3× higher loss:** 0.0133, against 0.0044 for s1 and 0.0057 for its jump twin. Yet held-out duck recall was 1.0000 and agreement 99.60 %.
  - So per-frame agreement did not reveal the stuck-duck mode, as the T2 row anticipated for the jump generation.
  - The release decision has very few rows (2,823 held-out release frames), which may make it fragile across training seeds.
- **Every duck clone holds DOWN 3–15 % longer than the teacher** (DOWN-frame ratio 1.026–1.151). This does no harm in B and C.
- **The new jump twin `jump-d2w32-s2` has a min margin of 2.79e-5**, itself below the 5.8e-5 CSS flag.

### Committed files
- `runs/exp-*/` is gitignored, so these were added with `git add -f`:
  - `runs/exp-bc-duck/{demos,heldout-demos}/demos.json`;
  - `best.json` and `bc-metrics.jsonl` for the six duck arms and `jump-d2w32-s2`.
- There is no `evals.jsonl`. `eval` was not run, because it spends held-out offsets and certification is out of scope.
- Scans, diag, obs, duckshare, margins, agreement output and logs stay local in `runs/exp-bc-duck/`.
