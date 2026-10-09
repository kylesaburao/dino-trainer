GOAL: Build all the scaffolding for the dino-game neural-network project and
stop BEFORE real training starts.

Repository: /Users/kyle/Downloads/dino-game (a standalone git repo on main).

## Read first
- Read TrainingPrompt.md at the repo root in full before doing anything.
- It is the authoritative spec for:
  - the baseline facts (commit 2da34de, how game.js works)
  - the refactor contract
  - the observation and anti-cheat rules
  - the browser AI-mode requirements
  - the trainer CLI
  - the checkpoint and resume design
  - PROGRESS.md
- Follow it exactly, EXCEPT where this prompt narrows the scope.
- Do not edit TrainingPrompt.md. If you have to deviate from it, record the
  deviation and the reason in PROGRESS.md.
- If TrainingPrompt.md is untracked, commit it unchanged as the first commit.

## In scope for this session
1. Phase 1 of TrainingPrompt.md in full:
   - the DOM-free simulation core with createGame({ rng }) and a seeded PRNG
   - the classic-script/require dual format
   - golden traces recorded from 2da34de, and the parity test, which must
     pass
   - the browser game unchanged
   - the `bench` command
2. Phase 2 in full:
   - the action interface through the real input functions
   - observations limited to what is on screen (x < 600), with the
     forbidden-feature rules enforced
   - the shared policy.js (observation extraction + MLP inference) used by
     both Node and the browser
3. Phase 3 in full, the browser AI-mode toggle, driven by a model.js that
   holds an UNTRAINED, randomly initialised model (from the smoke run below):
   - The goal is to prove the plumbing works, not to play well.
   - Verify every Phase 3 check in Chrome. Headless is fine.
   - "The AI plays and scores" means it acts and the score advances until it
     crashes.
4. The Phase 4 trainer, built and tested but NOT used for real training:
   - commands: `train` (with --target, --resume, --workers, --time-budget),
     `eval` (on the 25 fixed held-out seeds), `bench`, `export` (writes
     model.js)
   - worker_threads with the default worker count, and the lockfile with a
     PID check
   - atomic checkpoints written every N generations and on SIGINT/SIGTERM,
     containing the core-source hash; --resume refuses a hash mismatch
     unless --force is given
   - per-generation JSONL/CSV metrics
   - .gitignore entries for bulky checkpoints and logs
5. PROGRESS.md at the repo root, containing:
   - status: "scaffolding complete; training not started"
   - the benchmark throughput and the number of workers detected
   - the design decisions, with reasons
   - the smoke-test results
   - the exact command to start real training next session, and the exact
     command to resume it

## Smoke test (the only training allowed)
Prove the pipeline works end to end with tiny settings:
- Run `train` with a very small population, for 2–3 generations and a few
  workers, with --time-budget capped at 2 minutes or less. Check that:
  - checkpoints are written
  - metrics are logged
- Kill it with SIGINT, then check that:
  - a checkpoint is saved
  - the lockfile is released
- Run `train --resume` and check that it continues from the saved
  generation.
- Check that a second concurrent `train` is refused by the lockfile.
- Check that resume refuses a deliberately mismatched core hash. Use a
  temporary copy; do not edit the real core to test this.
- Run `eval` on the smoke checkpoint, then `export` it to model.js.

Do not run longer or larger training runs. Do not tune for score. Do not
claim any skill level for the model.

## Out of scope this session
- Real training runs and hyperparameter tuning
- Pursuing the score target
- The 1h30 time budget and the training stop condition in TrainingPrompt.md
  (they apply to the later training session)
- Everything listed as out of scope in TrainingPrompt.md

## Done when
- The parity test passes (`node`), and the browser game plays exactly as at
  2da34de with AI mode off.
- All Phase 3 AI-mode checks pass in Chrome using the untrained model.js.
- Every smoke-test check above passes.
- PROGRESS.md is written, and everything is committed with clear messages
  (do not push).

Then report:
- what was built
- the test and smoke results, including any failures
- the benchmark numbers
- the command to start training

Then STOP. Do not start real training.
