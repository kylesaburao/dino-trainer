# Pre-registration: does a curriculum that starts episodes inside 400–800 help stuck θs through the wall? (natural-ramp arm vs hyg + `--episodes 8`)

**This tests the register row "curriculum targeting the 400–800-point zone".** B′ (`research/prereg-es-curriculum.md`) could not test it, because B′ never starts an episode below 2000. This arm runs inside desktop-1's B′ confirmation batch, on the same start θs. Its paired control is desktop-1's ablation arm, hyg + `--episodes 8` (called **hyg8** here).

Status: **draft, paper only; launch gated on §4.6.** Author: desktop-2, 2026-10-10, on main `b1fbab8` (natural-ramp merged in `6331aee`).
Process: retrospective `handoff/2026-10-10T1635Z-retrospective.md` §4 and §6.

## 1. Decision this informs

Register row **"curriculum targeting the 400–800-point zone"** (currently OPEN). The question is whether natural-ramp starts (`--ramp-frac`; the solver plays 0 → L, L uniform in [400, 800], then hands over to the candidate) get stuck θs through the wall in fewer training frames than the same number of from-0 episodes.

Possible register outcomes:
- **promising, confirmation pending** (reading A). Next: a ramp batch on θs selected to die inside 400–800 (§4.5).
- **pruned (this design)** (reading B or C), with the scope in §7.
- **inconclusive** (reading D).
- **artefact** (reading F): the ramp helps where zone exposure can't explain it. Investigate the handover before reading the row.

A secondary decision: if the ramp helps per generation but not per frame, the next question is a prefix cache. That needs a game-clone API in `core.js`, a core change that needs the user's approval. This experiment only says whether that would be worth asking for.

## 2. Premise checks (paper; the CPU checks are in §4.6)

1. **What a ramp episode costs, against the right control.** Speed and score depend only on time while running (`core.js`; the speed ramp is 6 + 0.001·frames, independent of the controller), so reaching level L takes frames(L) = (√(36 + 0.08·L) − 6)/0.001 frames, whoever plays.
   - frames(L) is 2247 at 400, 3165 at 600 and 4000 at 800. The measured teacher prefix is 3,159 frames per episode (L uniform on 400..800) plus a handover delay under 100 frames (test/ramp.test.js).
   - **So a ramp episode costs exactly what a from-0 episode costs when the student survives to L.** A ramp episode differs from a from-0 episode only when the θ would have died before L. Then it pays teacher frames to show the student the stretch beyond its own death point. That is the mechanism this row is about.
   - Consequence 1: on a θ that never dies before 800, the ramp arm ≈ hyg8 by construction. R ≈ 1 is the **predicted null**, and those θs are a built-in negative control (§3, H-artefact).
   - Consequence 2: the readout is stratified by each start θ's **share of from-0 deaths below 800** (§4.3). The median death score is not enough, because a θ with a median at 1,100 can still die below 800 on a third of its episodes.
2. **Where the stuck θs die** (desktop-1's §2/§9 data on the B′ set): s9 at about 550 (low pteros), s7 at about 520–850 (mid pteros, cactus), s5 at about 1,200 and s8 at about 1,400 (best validation medians 1,297 and 1,622). So about half of a typical stuck set dies inside the zone. The ES model's held-out failures (467, 485, 664, 673, 740) all lie inside it.
3. **Faithfulness** (merged with the feature; test/ramp.test.js): the world and dino at handover are byte-equal to pure solver play at that frame (v1 and v2, 6 seeds × 4 levels). A solver student after handover replays pure solver play exactly. The student acts first on the handover frame (running, grounded, keys up) and then once per frame. wasm equals JS. Straight and split+resume runs are identical. With the flags off, training is bit-identical to `d4a1224` (harness 7/7).
4. **Fitness is unchanged apart from where episodes start.** With `--window 1000000 --start-zero-cap 1000000`, mean aggregation and no adapt, a from-0 episode's fitness is min(score, 1M)/1M. That is hyg's capped score divided by a constant, so the centred ranks are the same. A ramp episode's fitness is (score − L)/1M, where L is the same for every candidate on that episode (common random numbers: same seed, same deterministic teacher), so it ranks the same as its absolute score. One point gained counts the same in either kind of episode. A shorter window would bring back B′'s fitness reshaping, the confound B′ couldn't separate. **That is why the window is 1M.** Checked by gate (a) in §4.6.
5. **Overhead while stuck. The two earlier figures don't apply here.**
   - 65% (random init, students die at once) and about 9% (students that survive 2000-point windows) bracket this regime without describing it. Runs stop at the event, so the 9% regime is never reached.
   - Estimate from the §2 per-episode frames (from-0 F0 = 3.1–5.1k; teacher T = 3.16k; student after handover Fr ≈ 0.3–2k, not yet measured): the teacher's share of the ramp arm's frames is T/(F0 + T + Fr) ≈ **30–48%**.
   - Against hyg8, the ramp arm's frames per generation are (F0 + T + Fr)/(2·F0) ≈ 1.0–1.15×. That is ≈ 1 exactly for θs that survive past L (premise 1).
   - **Power cost:** under a matched frames budget, the ramp arm gets only 52–70% of hyg8's student frames. This makes a ramp win harder to show, and a null correspondingly weaker. The estimate is replaced by the measured values from probe (c) in §4.6.
6. **Teacher frames count toward the budget and toward R. This is the choice, and why.**
   - Cost parity: a solver frame (sim included) costs 885 ns single-threaded. A 32,32 student frame costs 1,547 ns in JS, and about 0.9–1.0 µs with wasm (the network part is 1.65–1.7× faster; the sim is 237 ns). The two are equal to within about 15%, and the teacher is the cheaper one, so counting 1:1 slightly penalises the ramp.
   - The prefix is a fixed cost of the method, with no clone API to cache it.
   - Teacher frames carry no gradient signal.
   - Student-only frames are reported as a secondary. **A win that appears only on student-only frames is read as no compute win** (H-cost).
7. **Handover failures.** `rampHandover` throws if the solver crashes or misses a clean handover within 600 frames. The tests cover about 40 seeds; the batch draws about 4 new seeds per generation per θ, roughly 8k prefixes per run. A throw kills the run. It is an infrastructure failure, not a result: resume from the last checkpoint and record it. Gate (b) sweeps about 10k seeds first.

## 3. Competing hypotheses and predictions

Strata (from gate (c), on each start θ): **in-band**: at least 30% of from-0 deaths fall below 800. **Above-band**: under 10%. Anything in between is reported but not read.

| Hypothesis | Mechanism | ramp vs hyg8, in-band θs | ramp vs hyg8, above-band θs | Positive controls |
|---|---|---|---|---|
| H0 — **start location irrelevant** | The wall is a slow climb on the from-0 signal. Seeing the post-death stretch adds nothing that 4 more from-0 episodes don't. | R ≈ 1 (within chaos) | R ≈ 1 | Both clear |
| H-zone — **targeting the zone helps** | Ramp episodes put the student past its killer inside 400–800, so it is graded on obstacles it rarely reaches from 0 | **R ≤ 1/4**, or ramp clears where hyg8 is censored | R ≈ 1 (predicted by premise 1) | Both clear; ramp no slower |
| H-cost — **real signal, eaten by the prefix** | The ramp helps per generation, but the teacher's frames cancel it | R ≈ 1 on total frames, but **< 1 on generations and student-only frames** | R ≈ 1 | Both clear |
| H-harm — **fewer from-0 episodes** | 4 from-0 episodes instead of 8 slow the from-0 skill, or the opening is lost | R ≥ 4, or valMedian < 200 at two consecutive validations after +100 | R ≥ 4 or opening loss | Ramp control delayed > 2× or lost |
| H-artefact — **not zone exposure** | Clean grounded handovers, or less early-death noise, help by themselves | R < 1 | **R ≤ 1/4 on above-band θs** | – |

Which comparisons separate which:
- H0 vs H-zone: in-band R.
- H-zone vs H-artefact: above-band R.
- H-zone vs H-cost: total frames vs generations and student-only frames.
- H-harm: the opening check.
- Ties on type T θs (s9-like) are a secondary result. hyg8 also plays 8 episodes, so tie-breaking alone shouldn't separate the arms.

Secondary comparisons on the same θs, not decisive: ramp vs B′ (`cur`) and ramp vs hyg. They differ in more than one variable (fitness aggregation; episode count).

## 4. Design

### 4.1 Paired θs
desktop-1's confirmation set, unchanged: the same frozen pre-resume checkpoints (sha256-checked copies), the same positive controls and the same failure-type classification. Pairing is of the starting state only. Each start θ gets a `ramp` run directory next to desktop-1's: `runs/exp-bp-sS-ramp`.

### 4.2 Arm
- **ramp:** desktop-1's hyg8 command exactly (`train --resume --wasm --episodes 8`, the same `--generations`, `--time-budget`, `--eval-every 10 --val-seeds 12`, watcher and budget), plus
  `--window 1000000 --start-zero-cap 1000000 --start-zero-frac 0.5 --ramp-frac 0.5 --ramp-min 400 --ramp-max 800`
  - No `--fitness-agg geomean` and no `--window-adapt`. Leaving them off is what makes premise 4 hold.
  - `--start-dist` stays at its default. There are no late (startScore) episodes, so it is unused.
- **Why ramp-frac 0.5:** it gives 4 from-0 episodes + 4 ramp episodes. That matches B′'s 4 + 4 and hyg's 4 from-0 episodes, where no opening loss was seen, while the total stays at hyg8's 8.
  - It is not tuned to match B′'s frame share. That would break the hyg8 pairing, which needs 8 episodes.
  - Estimated share of student frames after handover: 7–14% of all frames, against 8–13% for B′'s late windows (§2 of the B′ prereg). Gate (c) reports the measured value.
- **Why ramp-min 400 and ramp-max 800:** this is the register row's band. It runs from just before the first pterodactyls (minSpeed 8.5 at about 453) to past all the ES model's held-out early failures (467–740). A wider band (400–1,400, covering type-M walls) would test a different row. It is the follow-up if the above-band θs turn out to be the common case.

### 4.3 Primary metric
- **Event:** as desktop-1's batch: valMedian ≥ 2000, confirmed per that prereg's deviation 1 (the second validation or the trainer's held-out stop).
- **Primary: R = ramp / hyg8 training frames from the resume point to the event, counting `frames + teacherFrames` for the ramp arm.** Both arms are censored on the same frames budget as the rest of the batch (≥ 10B, desktop-1's §9.2). The watcher and `tools/bp-table.js` sum `r.frames + (r.teacherFrames || 0)`; `teacherFrames` is absent on every other arm. The censoring bounds are the same as in the B′ prereg's §4.3.
- **Secondary:** generations to the event; student-only frames; wall time; mean tiedPairs/64; the opening check (the lowest valMedian after +100); and teacher frames as a share of all frames, measured.

### 4.4 Horizon
The batch's frames budget, matched across arms. The ramp's ~1.0–1.15× frames per generation means it gets about as many generations as hyg8 (premise 5).

### 4.5 Sample size, and what it can detect
- n is set by desktop-1's θ set (6 θs). The in-band stratum may hold only 2–3 θs.
- **This is a screen for qualitative effects:** ramp clears where hyg8 is censored, or R ≤ 1/4. It is the same scale as the B′ screen.
- **If promising:** run a ramp-only batch on about 10 θs selected in-band (from desktop-1's survey: from-0 death share below 800 of at least 30%), paired against hyg8. Adopt only on the sign-test rule of the B′ prereg's §4.5 (n ≥ 6 → 6/6), then certify on fresh held-out offsets **from 325**.

### 4.6 Launch gates (needs CPU: run only in a slot desktop-1 offers; about 2 min in total)
- **(a) Fitness plumbing is neutral.** Resume frozen `ckpt-s9.json` for 5 generations at hyg8, and again with `--episodes 8 --window 1000000 --start-zero-cap 1000000 --start-zero-frac 1`. θ, Adam m and the RNG state must be bit-identical.
  - **Exact, with no floating-point allowance** (mac-3, on approval). Any difference blocks the arm. Before anything else runs, find and report the first differing value.
- **(b) Handover robustness.** A single-thread solver sweep over about 10k training-range seeds to 800 + 330 points. No throw is allowed. Report the largest handover delay.
- **(c) Probe on the start θs** (single thread, 40 episodes each, as in the B′ prereg's §2):
  - each θ's share of from-0 deaths below 800 (the stratum);
  - student frames after handover per ramp episode (Fr);
  - the post-handover killers;
  - the teacher's share of the ramp arm's frames.
  These numbers replace the estimates in premise 5.

## 5. Readings (fixed before the run)

| Outcome | Reading | Register action / next step |
|---|---|---|
| **A** — ramp wins (R ≤ 1/4, or clears where hyg8 is censored) on every in-band θ (at least 2), AND above-band θs are within 1/4 < R < 4, AND the positive controls clear in both arms, AND no ramp opening loss | Supports H-zone | Row → **promising, confirmation pending**: the in-band batch of §4.5 |
| **B** — ramp wins on at most 1 in-band θ, no harm, controls clear | H0 in this scope; no effect larger than about 4× detected | Prune **this design** (natural-ramp 400–800, frac 0.5, mean fitness), scope as in §7. The row stays open only for a different design (a wider band, or a larger ramp share) |
| **C** — opening loss on any ramp run, OR a ramp positive control fails within the budget, OR hyg8 wins (R ≥ 4) on most in-band θs | H-harm | Prune as **harmful in this scope** |
| **F** — ramp wins with R ≤ 1/4 on at least 2 above-band θs | H-artefact | **Don't read the row.** Find out what the handover gives above-band θs before any confirmation |
| **G** — ramp wins on in-band θs in generations and student-only frames, but not on total frames | H-cost | Row → **inconclusive: compute-limited**. Report how large the effect is in generations. A prefix cache (a core clone API) needs the user's approval before it is tried |
| **D** — anything else, including fewer than 2 in-band θs in the set | Inconclusive at this n | Report the per-θ R table, stratified. The ramp gets its own batch on in-band θs |
| **Invalid** — gate (a) fails for a reason other than float summation, OR a hyg8 positive control misses the event | The test can't show what it claims | Fix it before reading anything. A handover throw is not "invalid": resume and record it |

## 6. Compute (desktop host, 16 workers, `--wasm`)
- Per θ, up to the batch's frames budget. At ≥ 10B frames that is about 25–30 min per censored run (hygiene's measured 4.4B frames ≈ 708 s, which is about 6.2M frames/s; teacher frames are no slower).
- Worst case, all censored: 6 θs × about 27 min ≈ **2.7 h** added to desktop-1's batch. Expected less, because the controls and any successes stop at the event.
- One trainer at a time, scheduled by desktop-1.

## 7. Scope and caveats recorded with the result
- Shape 32,32; v2; pop 128; σ 0.1; lr 0.03; mid ranks; decoupled wd 0.01; 8 episodes (4 from 0 + 4 ramp); ramp band 400–800; mean fitness; window = cap = 1M.
- Main `b1fbab8` code, `--wasm`. Teacher: `test/solver.js` as merged.
- The θs were selected as stuck under hygiene, which affects both arms equally (paired). They are the B′ confirmation θs, not θs selected in-band, so the in-band n is whatever the set holds.
- The ramp arm plays 4 from-0 episodes against hyg8's 8. The comparison is "half the from-0 episodes replaced by ramp starts", not "added".

## 8. Gate results
(pending: run in a slot desktop-1 offers)
