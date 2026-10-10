# Handoff: orchestration moves from `mac-3` to `mac-4` for the duck generation (2026-10-10, about 19:30 UTC)

**You (mac-4) are the orchestrator for the duck generation, by the user's order:** "I want orchestration to transfer to 'mac-4' for the ducking phase."

The one jump-generation job still in flight is the Mac depth × width sweep. It's a subagent of mac-3, so mac-3 will review and merge it and send you its result. Everything else is yours from now on.

Read in order:
1. `AGENTS.md` in full. New today:
   - "Naming scheme": bare vN means game rules only; the teachers are `teacher-jump` and `teacher-duck`; the eras are the jump and duck generations;
   - the BC commands;
   - the ramp flags.
2. This file.
3. `research/follow-up-questions.md`: Q2 (the duck generation) and Q1 (policy → large → small distillation, which waits on the sweep).
4. `research/prereg-teacher-duck.md`: step 1, registered and **not yet briefed or run**. It's your first action.
5. `handoff/2026-10-10T1815Z-mac3-milestone.md` and `research/retrospective-research-value.md`, for context.
6. Earlier handoffs as needed: `…1711Z-orchestrator-transfer-mac3.md`, the retrospective at `…1635Z`, and `…1626Z` §1 for standing instructions.
7. The memory index `~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/MEMORY.md`. See "Naming scheme" and "Experiment pre-registration".

## 0. Newest user instructions (verbatim, 19:3xZ; these override older priorities)
- "I want mac and desktop to focus entirely on the problem of the ducking solver and training."
- "Parallelize work where possible."

Consequences:
- **The jump-generation sweep has been STOPPED.** mac-3 told its subagent to kill its jobs and write a PARTIAL report from completed data, with no certification, so offsets 325–474 stay unspent. mac-3 merges that and relays it to you.
- **ES research stays stopped.**
- **Both hosts work only on the duck generation.**

### Suggested parallel plan (mac-3's view; you decide)
Run these at the same time. They are independent until step 2 joins them.
- **Desktop host (Node-only, 16 workers): step 1** (`research/prereg-teacher-duck.md`).
  - desktop-2 implements `createDuckSolver` and its unit tests.
  - Then the 5,000-seed survey (control plus duck, about 20 min). desktop-1 can run the survey once the code is pushed. Only one CPU-heavy job runs on the host at a time.
- **Mac, in parallel, as subagents:**
  - (a) **`bc-gen --teacher jump|duck` plumbing**, defaulting to jump. Jump output must be sha-identical to `runs/exp-bc-e2e`. `bc-train`/`bc_train.py` must handle three classes with duck present: class weights and recall for duck. Add a `teacher` field to the demos manifest and `best.json` config. Develop it against a stub teacher interface, so it doesn't wait for step 1. The two can merge in either order: `bc.js` selects the teacher by name.
  - (b) **Write the step-2 pre-registration:**
    - demonstrations from `teacher-duck`, seeds 1–40;
    - clones at d2-w32 (shipped) and the smallest shapes that passed the sweep's tier B (d1-w8 and d2-w8 were 0/1,000 at seed 1);
    - paired comparison with the jump generation on scan seeds;
    - duck recall;
    - an observability check: is the duck decision a function of `observe()`?
    - margins and the CSS check.
  - (c) **A Sonnet research note on any open design question,** for example how much the ducking box's extra width constrains release timing. Keep this one optional.
- **Join:** step 1 passes, (a) merges, then step 2 runs on the Mac. That needs numpy, so the Mac is the only place for clone training.


- **Orchestrate.** Delegate edits to subagents or to the desktops. Keep Sonnet research going while idle, choosing topics by the decision they serve.
- **Branches → main.** The orchestrator merges and pushes main; the desktops push only their own `wt/*` branches.
- **Pre-register every run** (AGENTS.md "Research method"). Verify every claim in a brief against code. Consult the native advisor at design time.
- **Training rules:**
  - `runs/exp-*` only;
  - never `session`, `finish` or `--auto-export`;
  - never write the root `model.js`/`model.css` without the user's approval;
  - one trainer per host;
  - stay generic over `--hidden`;
  - bit-identical defaults;
  - `rm` only on literal paths (hook-enforced).
- **Resources:** the Mac CPU is for clone research (it is the only machine with numpy and a browser). The desktop host (desktop-1, desktop-2, sharing one CPU, no numpy, no browser) is Node-only.
- **Held-out offsets:** the next fresh one is **325**, unless the sweep's certification rung spends 325–474; mac-3 will tell you.

## 2. State of main (`ccbb600`, pushed)
- **Shipped model:** `runs/bc-v2-h32x32`, a jump-generation clone (44-32-32-3). It certified 150/150, and three clones scanned 0/2,000 each.
- **BC pipeline:**
  - `bc-gen` → `bc-train --backend node|python [--snapshots]`;
  - validated end to end (`research/prereg-bc-pipeline-e2e.md`);
  - the teacher is hard-wired to `test/solver.js` `createSolver` (`trainer/bc.js`). **A `--teacher` option is needed for the duck generation's step 2.**
- **ES research:**
  - Round-1 curriculum: "B′ as specified: promising, confirmation pending".
  - Round-2 attribution: **stopped by the user** at 19:23Z (`research/prereg-es-curriculum.md` §10.9). It is partial and was not read. Its checkpoints are kept on the desktop host for resuming.
  - Natural-ramp starts are merged, and their gates passed (`research/prereg-es-ramp.md`). That arm never ran.
  - Survey finding: the second wall is a minority outcome under hygiene (about 7/22 of escapers).
  - **The user wants focus on the duck phase. Don't restart ES without asking.**

## 3. In flight
| Item | Where | Owner | Status |
|---|---|---|---|
| Depth × width sweep (jump generation; `research/prereg-bc-sweep.md`): **being STOPPED; a partial report is coming** | Mac, subagent of mac-3; worktree `/Users/kyle/Desktop/dino-game-sweep`, branch `wt/bc-sweep`; data in `runs/exp-bc-sweep/` | **mac-3** reviews, merges and relays it to you | At 19:12Z: 28 configs trained; depth ≥ 32 and d16-w64 fail tier A (0/25); tier-B scans mostly 0/1,000, but d8-w8 is 2/1,000. Second-seed scans, margins, and possibly certification still to come. The Mac CPU frees as soon as its stop completes; Mac code-writing can start right away |
| Desktop host | desktop-1, desktop-2 | you | **Idle.** desktop-2 has read Q2 and is waiting for a brief. desktop-1 is idle |

## 4. Your first actions (mac-3's recommendation)
1. **Tell desktop-1 and desktop-2 that they report to mac-4.** Use `SendMessage` with `to: "desktop-1"` / `"desktop-2"`; their replies come from bridge addresses.
2. **Brief `teacher-duck` step 1** (`research/prereg-teacher-duck.md`). It's Node-only, so it suits the idle desktop host. desktop-2 is the trainer-code owner.
   - Implement `createDuckSolver` in `test/solver.js`, leaving `createSolver` unchanged, with unit tests.
   - Run the 5,000-seed survey: `createSolver` as the control, then the duck solver (about 20 min on 16 workers).
   - Apply §5 verbatim.
   - desktop-2's note: `trainer/ramp.js` is hard-wired to `createSolver` and already refuses to hand over mid-duck. Leave it alone.
3. **Step 2** (after step 1 passes): add `bc-gen --teacher jump|duck`, defaulting to jump. This is a bit-identical-defaults change, and `bc-gen` output for jump must be sha-identical to `runs/exp-bc-e2e`. Three-class demos follow, then cloning on the Mac and a paired comparison with the jump generation. That needs its own pre-registration. Use the shapes the sweep found passing, so wait for its report.
4. Keep the register rows current. Write a handoff at each milestone.

## 5. Peers
- desktop-1: `SendMessage to: "desktop-1"` (bridge `session_017dSzHB17BR88LBomfMfTzk`).
- desktop-2: `"desktop-2"` (bridge `session_01LGggTtejaCN5qNEutis31D`).
- Delivery on these routes is unconfirmed. Rely on replies, and don't resend blindly.

## 6. Lessons from mac-3's session
- **A `python3 - <<EOF … EOF` heredoc followed by `git add/commit` on the next line commits even when the script fails.** End the heredoc line with `|| exit 1`.
- **Peer messages can fail the auto-mode classifier.** If one is denied, report it to the user rather than working around it. The user then authorised it with "do whatever work is required".
- **Verify numbers before putting them in a brief:**
  - a "significance" claim (1/1,000 vs 0/6,000 is not significant, Fisher p ≈ 0.14);
  - a parameter count (16-16 is 1,043, not 1,059);
  - one-sided vs two-sided CP bounds (0/2,000 is 0.150% one-sided, 0.184% two-sided).
