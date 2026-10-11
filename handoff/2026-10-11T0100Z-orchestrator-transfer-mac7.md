# Handoff: orchestration moves from `mac-5` to `mac-7` (2026-10-11)

**You (mac-7) are the orchestrator, by the user's order:** "Transfer control to mac-7 after the webpage work is done".

Read in this order:
1. `AGENTS.md`, in full.
2. This file.
3. The memory index, `~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/MEMORY.md`. New since mac-4:
   - **detached-runner-launch:** subagents can't launch nohup runners on an approval I relay. The user adds an allow rule and the orchestrator launches.
   - **gpu-cpu-overlap:** one GPU job may run alongside one CPU-heavy job.
4. The previous handoff, `handoff/2026-10-10T2300Z-orchestrator-transfer-mac5.md`. Its §0 rules still apply.

## 0. Standing user rules (unchanged unless noted)
- **Every decision for the user goes through AskUserQuestion.** The user reads them on a phone. Plain text is for status only.
- **Guard and permission blocks:**
  - An agent may apply the safe rewrite a guard message names **once**.
  - Any other block, a second block, or **any auto-mode refusal**: stop and ask the user.
  - A user approval relayed to a subagent does **not** get past auto mode. It escalates to [Auto-Mode Bypass], and retrying is refused even in the orchestrator's session.
- **Edits with Edit/Write only.** Two agents broke this tonight with `sed -i` on comments. Restate it in every brief.
- **Git in worktree agents:** run every git command alone, as `git -C <worktree> …`. Never chain git with other commands, and never `cd` to the shared checkout. Chaining caused most of tonight's guard blocks.
- **Worktree agents:** an isolated worktree is deleted when its agent stops with no commits, and that also happens on a hand-back. Have agents commit first. To continue such an agent, spawn a fresh one; resuming it fails.
- **Machine (new 2026-10-11):** one GPU job plus one CPU-heavy job may run together; two CPU-heavy jobs may not. Parallel numpy with K > 2 counts as CPU-heavy.
- **Shape cap:** ≤ 16 layers, ≤ 32 units per layer.
- **No `export`/`finish`/`--auto-export`** to the root model files without the user's approval.
- **Held-out offsets:** the next fresh one is **625**. The registered 1×6 v2 certification will spend 625–774; a rules-v3 certification starts at 775.
- **Pushing main:** the user authorized pushes ("Push is authorized").

## 1. State of main (pushed)
- **Shipped:** unchanged, `runs/exp-bc-duck/v2-bc-duck-d2w8-s1` (44-8-8-3).
- **D1 distillation: dropped by the user** ("not tenable") at 23:51Z.
  - Provisional result: no arm reliable, none better than direct cloning (all p ≥ 0.19).
  - Merged `c87d861`, which also carries Stage 0 (1×6 and 1×4 10/10 clean on rules v2) and the 1×6 certification candidate `runs/exp-distill/v2-bc-duck-d1w6-s1` (params sha `eb89e31f…`, verified).
- **Rules-v3 clone study: done** (`research/prereg-v3-clones.md`, Results plus the executor's detailed record).
  - Every one of 30 models learned rules v3 past 2,000 points: post-step 30 × 1,000/1,000, tier B 5 × 0/1,000.
  - Every failure is the **opening release decision**. Clean counts: 1×6 4/10, 2×32 2/10, 2×8 1/10.
  - Candidates: `v3-bc-duck-d2w8-s2` (margin 8.1e-4), `d2w32-s5` (1.9e-4) and `d1w6-s4` (6.8e-5). Not certified.
- **v3 tooling** (`b0cd6b9`): `scan.js`/`opening.js`/`duckshare.js` play the model's own rules, with `--game`, `--start` and `--expect-hash`. Also `margin.js --levels`, `agreement.py --split-x4`, and dashboard display of rules and start.
- **GPU trainer plan** (`research/plan-gpu-trainer.md`, merged): MLX stacked models, plus a parallel-numpy `bc-train-batch`. The user approved implementation and the phase order "parallel numpy first".
- **Release-fix prereg §13** (`4d426de`): a rules-v3 extension (stage 4) and the core-hash note.

## 2. In flight
| Item | Owner | State | Next |
|---|---|---|---|
| **Model picker + speed 1×–4096×** on index.html and index-css.html | done (mac-5 subagent) | **Merged.** The menu has 7 models: shipped duck 2×8, jump 2×32, ES, duck 1×6, and the three rules-v3 uncertified candidates. It comes from `node trainer/cli.js export-models` (`models.js`, `models-css/*.css` and `index.js`). Speed is AI-only; index.html reaches the full 4096×, index-css.html 7–99× (it says so). CSS/JS: 0 disagreements for every model. On main: browser-check 66/66, `node --test` 137/137 | Nothing. If a model is added to or removed from `BUNDLED_MODELS` in `trainer/cli.js`, re-run `export-models`; `test/models.test.js` fails when the generated files are stale |
| **GPU trainer** | **mac-6** (its own session) | Branch `wt/gpu-batch`, HEAD `ebb8965`. Done so far:<br>• Phase 1 (`bc_common.py` refactor, byte-identical);<br>• the parallel-numpy `bc-train-batch --jobs` slice, every gate passing;<br>• Phase 0 `.venv-mlx` installed (py 3.14.8, mlx 0.32.3, numpy 2.5.4; RNG streams identical to numpy 2.0.2);<br>• the MLX engine `trainer/bc_train_mlx.py` (TF32 off; E0 10/10).<br>mac-5 cleared the CPU for its §7 benchmark at 00:30Z. Nothing from that branch is merged yet | Review and merge `wt/gpu-batch` when mac-6 asks. Tell mac-6 when the CPU is free or busy |

## 3. Queue (Mac)
1. **Release fix** (`research/prereg-teacher-duck-release-fix.md` with §13). **ON HOLD by the user.** Before it can run:
   - the user must add the allow rule `Bash(PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nohup node research/scripts/prereg-teacher-duck-release-fix/run.js:*)`;
   - an executor writes that `run.js`;
   - the orchestrator reads it and launches it.
   Cost is about 3 h of CPU. Ask the user before starting.
2. **D2 quantization** (pow2, 8 and 4 bits, `research/prereg-distill.md` §12). The user kept it. It needs a short amendment first.
3. **1×6 certification** (`research/prereg-cert-duck-1x6.md`): offsets 625–774. The precondition (the candidate on main, sha verified) is met.
4. Later: a rules-v3 certification of the v3 candidates (offsets from 775). Needs its own pre-registration and the user's go.

The user moved the release fix ahead of D2 on 2026-10-11, then put it on hold.

## 4. Open user-facing threads
- **Float32 GPU evaluation:** mac-5 measured 0 decision flips in 13.6M decisions (4 models, float32 forward pass on the CPU). The whole-game float32 port breaks scoring above about 420k points unless the accumulators keep more precision, and it plays different games per seed. So it is good for triage only. Probe script: `f32probe.js` in mac-5's scratchpad, not committed. The user hasn't decided whether to pre-register a GPU triage comparison.
- **The allow rule for the v3 runner** (`prereg-v3-clones/run.js`) is in the user's **global** `~/.claude/settings.json` (line 10), not in a project settings file. mac-5 suggested the user remove it once the study was done; the study is now done. The release-fix rule (§3) would go in the same place, wherever the user chooses.
- **The naming memory** (`project-generations-v2-v3`) maps the user's old "v3" to the duck generation, which is ambiguous now that rules v3 exist. mac-7 will ask the user for a ruling.

- **PROGRESS.md was archived** at the user's request, as stale, to `handoff/archive/2026-10-10T2203Z-PROGRESS.md`. `trainer/cli.js report` recreates a fresh one if it is run.

## 5. Lessons from tonight
- `status.log` files keep old "runner done" lines, so wait on new lines only (count the lines first).
- A killed runner writes no "runner done", so tell anything waiting on it.
- `bc-gen` is now sub-second on 18 workers. The cost tables overstate it.
- `margin.js` on rules v3 can fill a level's quota from one game, so check where its minimum falls (the executor's note 3 in the v3 record).
