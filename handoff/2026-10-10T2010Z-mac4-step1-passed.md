# Handoff: duck generation, step 1 passed, step 2 running (mac-4, 2026-10-10, about 20:10 UTC)

Orchestrator: **mac-4**, which took over from mac-3 at 19:30Z. mac-3 has finished and holds no tasks.

## User orders in force
- "I want mac and desktop to focus entirely on the problem of the ducking solver and training."
- "Parallelize work where possible."
- "To be clear, I want the new solver to use both jump and duck." So `teacher-duck` jumps cacti and low pterodactyls, stays grounded under high ones, ducks under mid ones, and falls back to a jump where a duck can't fit. Clones train on three classes: none, jump and duck.
- **Dashboard invariant** (AGENTS.md, `700e8fb`): every training run must show in `trainer/dashboard.js`.

## Done since 19:30Z (all on main, pushed; head `5fd8248`)
- **`createDuckSolver`** (`test/solver.js` and `test/solver.test.js`, by desktop-2): `createSolver` is byte-for-byte unchanged. The dated amendment `51bbaed` records two things: DOWN is held over the standing-box overlap ±1 frame, and the duck box spans x 54–104.
- **Step 1 survey** (desktop-1, `research/prereg-teacher-duck.md` Results), seeds 40,000,000–40,004,999, full games to 1M:
  - control `createSolver`: 0 crashes in 5,000;
  - `createDuckSolver`: **0 crashes in 5,000** (< 0.06% one-sided 95%). Fallback was 0 out of 41.37M mid pterodactyls, conflicts were 0, and DOWN was held on 2.22% of frames.
  - **H-safe holds; `teacher-duck` is valid.**
- **`bc-gen --teacher jump|duck`**, plus three-class `bc-train` on both backends (`6ef29c2`). With the default teacher, `demos.bin` sha `3a67bfd4…` and the params of both backends are bit-identical. The new fields `teacherName` and `recall.duck` are additive.
- **Dashboard BC support** (mac-3, `90f129e`). It shows nested runs, recall for every class key and `teacherName`, and plays back every `create*Solver`. mac-4 verified that a duck run renders.
- **`crash-classify.js`** (desktop-2), and the README (how to train and how to view the dashboard).
- **Step 2 pre-registration** `research/prereg-teacher-duck-step2.md` is registered (`4ab1ee5`). Its observability gate is blind: it uses held-out demos from seeds 41–60, and its bars were fixed before that data existed.

## In flight
| Item | Where | Status |
|---|---|---|
| Step 2 execution: arms d2-w32, d2-w8 and d1-w8 at seeds 1–2, plus the jump d2-w32-s2 twin; tier A/B paired on scan base 30,003,000 | Mac subagent, branch `wt/exp-bc-duck` | Started about 20:10Z; about 25–40 min |
| Desktop host | desktop-1, desktop-2 | Idle |

## Held-out offsets
The next fresh offset is **325**. None have been spent in the duck generation.

## Lessons
- Peer messages that ask a desktop to push can trip the auto-mode classifier, and so can `git push origin main`. The user authorised both at about 19:50Z ("push main and send message").
- Check numbers against code before briefing. My brief gave the duck box as x 56–106 and estimated about 180 mid pterodactyls per seed. The correct values are 54–104 and about 8,000 per full game.
