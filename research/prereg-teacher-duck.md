# Pre-registration: `teacher-duck`, step 1. Does a ducking solver never crash?

Status: **registered.** Author: mac-3, 2026-10-10, on main `aa83438`.
Context: `research/follow-up-questions.md` Q2 and AGENTS.md "Naming scheme". This opens the **duck generation**.

## 1. Decision
Is `teacher-duck` a valid teacher, one that never crashes and actually ducks, so that the duck generation can go on to demonstrations and cloning (step 2)? This is the necessary condition. Nothing is trained until it holds.

## 2. Mechanics (read from `core.js` on `aa83438`)
**Hitboxes.**
- Standing box: x 56–86, y 100–138 (`standBox(STAND_TOP)`; `STAND_TOP` 98).
- Ducking box: `{x: dino.x+4, y: GROUND−DUCK_H+4 = 120, w: 50, h: 18}`, so y **120–138**. It is **wider** than the standing box and reaches further forward.
- Pterodactyl box: `{x: o.x+4, y: o.y+13, w: 36, h: 8}`, with `ys: [108, 82, 58]`.
  - **mid** (y = 82) has box y 95–103: it hits a standing dino and misses a ducking one;
  - **low** (y = 108) has box y 121–129: it hits both;
  - **high** (y = 58) has box y 71–79: it misses a grounded dino.

**Keys** (`applyKeys`: transitions only).
- `pressDown` while grounded sets `ducking`. While jumping it fast-drops instead (`speedDrop`, `vy = 1`), which changes the jump arc.
- `releaseDown` clears both.
- `startJump` is refused while `ducking`.
- On landing, `ducking = downHeld`.

**Fairness model** (`requirement()`). Mid pterodactyls are classed as must-be-**airborne**. Ducking is ignored: "Fast-drop and ducking only add options, so the model is conservative."

## 3. Hypothesis and its premise
**H-safe:** replacing each planned jump over a mid pterodactyl with "stay grounded and duck through its overlap" never crashes.

**Argument** (from the code, untested):
- Call P the mid pterodactyl, and Q any other upcoming obstacle.
- P's standing overlap interval O_P lies inside its planned airborne interval A_P.
- The generator keeps every other requirement compatible with A_P: airborne requirements are separated from it by at least minGround, and grounded ones are disjoint from it.
- So the dino is free to be on the ground throughout A_P, and the ground time between Q's jumps is unchanged.

**Two places this can fail, which the test must probe:**
1. **The ducking box is wider (w = 50, reaching further forward), so it touches obstacles earlier.** If the dino is still ducking when the *next* obstacle (a cactus or low pterodactyl) reaches the ducking box's front edge, it collides, even though the standing box would not have.
   - So DOWN must be released early enough. That margin comes out of the ground time before Q's takeoff, and it has to fit.
2. **Pressing DOWN while airborne fast-drops** and changes a planned jump. The solver must press DOWN only when grounded, and must release it before any jump: `startJump` is refused while ducking.

## 4. Design
**Solver.** Add `createDuckSolver` to `test/solver.js`, keeping one teacher owner. `createSolver` stays unchanged, because the bc tests and the shipped clone's exact reproduction depend on it.
- Behaviour: identical to `createSolver`, except for an obstacle that `requirement()` classes as airborne whose box clears the ducking box (`box.y + box.h ≤ 120`; compute this from the boxes, don't hard-code y = 82):
  - don't take the jump;
  - hold DOWN while grounded from just before its overlap with the **ducking** box begins to just after it ends;
  - release DOWN early enough that the next planned jump, and the ducking box's reach into the next obstacle, are both respected.
- If DOWN can't be released in time (failure 1), **fall back to the planned jump for that obstacle and count it.** The teacher must never crash because it chose to duck.

**Survey.** Full rules-v2 games from 0 to 1,000,000 on **seeds 40,000,000–40,004,999** (5,000 seeds), which no earlier work has used.

**Positive control:** `createSolver` on the same 5,000 seeds must give 0 crashes. That is the known behaviour, and it validates the harness.

**Measures:**
- crashes, with seed, score, obstacle and the last 30 frames for each;
- coverage: mid pterodactyls ducked, mid pterodactyls that fell back to a jump, and mid pterodactyls encountered;
- the share of frames with DOWN held;
- the share of all obstacles ducked.

**Unit tests** (`test/solver.test.js` or similar):
- the duck solver's actions equal `createSolver`'s on every frame before the first mid pterodactyl;
- on a constructed or seeded mid-pterodactyl case, it ducks, survives, and releases DOWN before the next jump;
- it never presses DOWN while airborne.

**Cost:** the solver runs at about 0.9–1.4 µs/frame, and a full game is 1.94M frames. Two arms × 5,000 games × 1.94M frames is about 1.9e10 frames, roughly 5 h CPU, or about 20 min on 16 workers.

## 5. Readings (fixed)
| Result | Reading | Next |
|---|---|---|
| Control (`createSolver`) crashes on any seed | Invalid: the harness or game is suspect | Diagnose; read nothing |
| Duck solver: 0 crashes in 5,000, and fallback < 10% of mid pterodactyls | **H-safe holds** at this scope (crash rate < 0.06% one-sided 95%). `teacher-duck` is valid | Step 2: `bc-gen --teacher duck`, clone and compare (separate pre-registration) |
| 0 crashes, but fallback ≥ 10% | Valid but often can't duck (failure 1 binds) | Report the fallback rate and where it happens. The user decides whether that is "ducking enough" |
| Any crash | H-safe is false for this implementation | Classify each crash (failure 1, failure 2, or other). Fix the solver and re-test on **new** seeds (40,005,000+), never the same ones. If the only fix needs a change to the fairness model, that is a game-rules change: **stop and report** |

## 6. Scope
- Rules v2, from score 0, full games.
- Ducking under mid pterodactyls only. No fast-drop.
- Code commit recorded with the result.

## Results

### Control arm (`createSolver`), positive control: **PASS (harness valid)**
- Run: `research/scripts/prereg-teacher-duck/survey.js --solver jump --base 40000000 --n 5000 --workers 16` on the desktop host (Ryzen 7 9700X, WSL2).
  - Harness commit `fd145d1`; core hash `0f329d4e3493…`; node v26.9.0.
  - 2026-10-10 19:29:26Z → 19:45:45Z (979 s).
  - Raw JSONL in `runs/exp-teacher-duck/` (gitignored).
- **0 crashes in 5,000 full rules-v2 games to 1,000,000 points** (seeds 40,000,000–40,004,999). One-sided 95% Clopper–Pearson upper bound on the crash rate: 0.060%.
- 9.72B frames. DOWN held on 0.000% of frames; 0 of 287,419,501 passed obstacles ducked, as expected for the jump-only solver.
- Mid pterodactyls (game-side: the box clears the ducking box but hits the standing box, i.e. y = 82): 41,372,440 encountered, 41,371,831 passed, 0 ducked.
- `createSolver`'s body is unchanged by the merge of `createDuckSolver` (9616c3b; the diff touches only the import and export lines).

### Duck arm (`createDuckSolver`, solver commit `9616c3b`)
Running on the same seeds since 19:45:52Z (branch head `e635de4`). Results follow.
