# Handoff: orchestration moves from `mac-4` to `mac-5` (2026-10-10, about 23:00 UTC)

**You (mac-5) are the orchestrator, by the user's order:** "transfer orchestration to mac-5".

Read in order:
1. `AGENTS.md` in full. New today: the naming scheme (run dirs `v<R>-<method>-…`), rules v3, the shape cap (≤ 16 layers, ≤ 32 units per layer), the `assess` command (which replaces `eval`), and the shipped duck clone in the decision log.
2. This file.
3. Memory index `~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/MEMORY.md`. Key entries: blocking-questions-via-ask, guard-blocks-stop-and-ask (with its two amendments), shape-limits, naming scheme, orchestrator-workflow (the worktree lesson).
4. Pre-registrations: `research/prereg-distill.md` (§11a, §12; §11b is on branch `wt/exp-distill2`), `research/prereg-teacher-duck-release-fix.md`, `research/game-v3.md`, and `research/prereg-teacher-duck-stuck.md`.

## 0. Standing user rules
- **Every decision for the user goes through AskUserQuestion.** The user reads them on a phone over Remote Control. Plain text is for status only.
- **Guard and permission blocks:**
  - If the worktree guard's message names a safe rewrite (plain commands, literal paths, Edit/Write), an agent may apply it **once**.
  - Any other block, a second block, or **any refusal by the auto-mode permission check** means: stop and ask the user.
  - File edits are made with Edit/Write only, never python or sed heredocs. Put this wording in every brief.
- **Worktree agents:** an isolated worktree is deleted if its agent stops with no commits. Have agents commit early. If an agent stopped with no commits, spawn a fresh one; don't try to resume it.
- **Agents hand back mid-run** while their detached runners keep going. The "resume point" in their report is the way to drive them on (SendMessage to the agent).
- **Shape cap:** ≤ 16 layers, ≤ 32 units per layer.
- **Runs:** under `runs/exp-*`; never `export`, `finish` or `--auto-export` without the user's approval; one heavy job at a time on the Mac.
- **Desktops** (desktop-1, desktop-2) are **shut down** by the user. Everything runs on the Mac.
- **Held-out offsets:** the next fresh one is **625**. Offsets 475–624 were spent on rules v3.

## 1. State of main (`63dcfa1`, pushed)
- **Shipped:** `runs/exp-bc-duck/v2-bc-duck-d2w8-s1`, 44-8-8-3, a duck-generation clone. It certified 150/150 on offsets 325–474, and the CSS check found 0 disagreements over 421 duck frames.
- **Rules v3** is merged:
  - v2 up to 100k, then the speed cap steps 19 → 41 and ramps to 60 at 1M;
  - 51% of obstacles leave under 150 ms at 101k, and 100% at 1M;
  - both solvers crashed 0/3,000 times from 100k to 1M;
  - both v2 clones scored 0/150 on rules v3, all dying within 1.6k points of the step;
  - core hash `65f3546c…`.
- **Tools:**
  - `bc-train` flags: `--distill-from`, `--quant-bits` with `--quant-scheme pow2` (every param N-bit, with a per-layer 2^-k), `--extra-demos`, `--transition-weight`;
  - `bc-gen` flags: `--score-window`, `--student`;
  - `bc-quantize`, and `pack.js` (storage bits);
  - `test/css-agreement.js` for candidate models;
  - the dashboard shows runs, scans, surveys and distill/quant/DAgger runs.
- **Stuck-duck study:** at d2w32, 2 of 20 seeds were stuck; at d2w8, 9 of 20 were stuck; plus an early-release mode. The opening scan is a required screen.

## 2. In flight
| Item | Owner | State | Next |
|---|---|---|---|
| **Distillation D1** (`research/prereg-distill.md`) | mac-4 subagent, worktree `.claude/worktrees/agent-aab38dbf9659a16f4`, branch `wt/exp-distill2` (base `b4624b3`; core hash 0f329d4e is kept on purpose; don't rebase it mid-study) | The detached runner is on 2×6 τ = 4 at 22:55Z, arm 3 of 10, about 40 min left. Progress is in `runs/exp-distill/status.log`. So far at 2×6: direct 4/10 clean, τ = 1 3/10, τ = 2 4/10 (τ = 2 trades stuck ducks for early releases) | When status.log shows "runner done": tier B on at most 2 clean seeds per arm, agree.js, summarize, McNemar, then the readings. The full command list is in the agent's last report, and in the prereg's resume notes. Then merge `wt/exp-distill2`: expect a conflict in `research/prereg-distill.md`, because main has §12 |
| **Stage 0 finding** (on wt/exp-distill2) | — | Direct duck clones: 1×6 10/10 and 1×4 10/10 clean; 2×6 4/10; 2×4 5/10. Depth, not size, looks like the cause of stuck ducks (not a registered reading) | — |
| **Prereg writer: v3 clones plus 1×6 certification** | mac-4 subagent (running) | Writing `research/prereg-v3-clones.md` and `research/prereg-cert-duck-1x6.md` on `wt/prereg-v3-clones` | Review and merge. The user's order: run the v3 clone study **right after D1**. Certify 1×6 "later" (offsets 625–774) |
| **D2 quantization** (pow2, 8 and 4 bits; prereg §12) | — | Registered; on hold until D1 finishes. The old row-scale arms are dropped (user) | After the v3 clones, unless the user reorders |
| **Release-fix study** | — | Registered and code merged | Queue position: after the v3 clones |

mac-4 will relay the two subagents' final reports to mac-5 when they arrive, and will issue no further instructions.

## 3. Queue (Mac)
1. Finish D1.
2. v3 clone study.
3. D2 quantization.
4. Release fix.
5. 1×6 certification.

The user ranked the v3 clones "next after D1". Ask the user about any other reordering.

## 4. Lessons
- `pgrep -f` matches its own command line, so wait on the status.log "runner done" line instead.
- Merges between the bc feature branches conflict in bc.js, bc_train.py and cli.js. Have the branch owner rebase rather than resolving it yourself.
- `test/bit-identity.js` needs `--core-changed` across a core-hash change, and `--allow-busy` while a runner is going (it's deterministic).
