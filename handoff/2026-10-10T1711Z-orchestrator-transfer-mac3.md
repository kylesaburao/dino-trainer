# Handoff: orchestration transfer from `mac-2` to `mac-3` (2026-10-10, 17:11 UTC)

**You (mac-3) are now the orchestrator.** The user ordered the transfer: "Immediately transfer control to "mac-3" after subagent finishes. You're making too many mistakes." mac-2's mistakes are listed in §6. Read them so you don't repeat them.

Read in order:
1. `AGENTS.md` in full. It was updated this session: model provenance, the decision log, `--wasm`, 7 bit-identity scenarios.
2. This file.
3. `handoff/2026-10-10T1653Z-ship-bc-clone.md` (the ship).
4. `handoff/2026-10-10T1635Z-retrospective.md` (method, pruned-branch register).
5. `handoff/2026-10-10T1626Z-orchestrator-transfer.md` (standing instructions §1, peers §2).
6. The memory index `~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/MEMORY.md`.

## 1. User decisions this session (verbatim, authoritative)
1. "Shipping the clone is allowed." The solver behaviour clone is now `model.js`.
2. "Clearly document this pivot in architecture." Done: AGENTS.md has "Model provenance", the clarified anti-cheat rule and the "Decision log".
3. "Make cloning first class trainer (support both modes), pure-ES research runs parallel to clone research. Clone research has priority over resources."
   - Both modes = a Node backend and a Python/numpy backend.
   - The Mac CPU is the clone-research resource; ES runs on the desktop host.
4. The standing instructions in transfer-handoff §1 still apply:
   - delegate edits;
   - keep Sonnet research going while idle;
   - branches go to main, and the orchestrator may push main;
   - use runs/exp-*, never session, finish or --auto-export;
   - one trainer per host;
   - stay generic over --hidden;
   - bit-identical defaults;
   - rm only on literal paths.

## 2. State of main (pushed, `1ba7231`)
- **Shipped model:** `runs/bc-v2-h32x32` (44-32-32-3 clone of `test/solver.js`). It passed 150/150 on fresh held-out offsets 175–324, against 143/150 for the ES model (McNemar p = 0.008). **The next fresh held-out offset is 325.**
- **Merged this session:**
  - `--wasm` (1.65–1.7×, bit-identical on arm64 and x86);
  - research note 8 (`research/validation-selection.md`);
  - desktop-1's `tools/`;
  - desktop-2's allpass bit-identity scenario (7 scenarios).
- **Untracked `RECONTEXT_PROMPT.md` in the repo root:** a stale, pre-existing file. Leave it.

## 3. IN FLIGHT
1. **Mac subagent (general-purpose): the BC pipeline.**
   - Worktree `/Users/kyle/Desktop/dino-game-bc`, branch `wt/bc`, from main `322fcaf`.
   - Its plan, the authoritative spec, is `runs/exp-bc-clones/BC-PIPELINE-PLAN.md` (gitignored, local). It covers `bc-gen`, `bc-train --backend node|python`, `trainer/bc.js`, `trainer/bc_train.py`, and positive controls (a) data match, (b) exact Python reproduction of `m_base_s1`, (c) Node clone 25/25 validation.
   - Its first launch prompt was missing the plan, so it was re-sent by path. mac-2 transfers when it reports, and its report is relayed to you.
   - **To merge:**
     - check the report against plan §7;
     - rerun `node --test test/`;
     - run `node test/bit-identity.js /Users/kyle/Desktop/dino-game /Users/kyle/Desktop/dino-game-bc`. Main has moved to `1ba7231`; the difference is only the harness and docs;
     - then `--no-ff` merge and push.
2. **desktop-1: ES experiment "dense late exposure transfers down to the second wall"** (B′ as merged).
   - Pre-registration draft: `wt/desktop-1` `d312df5`, `research/prereg-es-curriculum.md`.
   - mac-2 approved option (a) with 4 amendments:
     - rename the row;
     - make the paired time-to-event ratio the primary reading, with "both clear" covered;
     - a reproduction check: hygiene arm resumed from frozen s12 must replay its escape at about +65 gens, otherwise STOP;
     - verify that select/fold/stop flags don't move θ.
   - It runs on the desktop host (about 100–110 min).
   - **Don't merge the pre-registration until the reproduction check passes.**
   - The register row "curriculum targeting 400–800" stays OPEN whatever this shows.
3. **desktop-2: scoping "natural-ramp starts"** (the solver drives from 0, then hands over to the student at 400–800).
   - Step 1 is a premise check only: why env.js forces starts to be 0 or ≥ ~2000, plus a cost estimate.
   - **It writes no code until the orchestrator OKs step 1.**
   - The flag must be off by default, training-only, and bit-identical.

## 4. Peers
- desktop-1: SendMessage to `bridge:session_017dSzHB17BR88LBomfMfTzk`.
- desktop-2: `bridge:session_01LGggTtejaCN5qNEutis31D`.
- Both were told they report to mac-2. **Tell them they report to mac-3 now.**
- Delivery over these routes is unconfirmed; rely on their replies.

## 5. Open next steps
1. Merge `wt/bc` after verification. Then pre-register clone research on the Mac (priority). Candidates:
   - DAgger or robustness of the clone;
   - smaller or different shapes;
   - Node vs Python backend parity of outcomes;
   - ES fine-tuning from a clone.
   Each needs AGENTS.md "Research method" pre-registration and certification from offset 325.
2. desktop-1's batch: apply its fixed readings and update the register.
3. desktop-2: review its step-1 premise report before approving code.

## 6. mac-2's mistakes, so you don't repeat them
- **A brief went out with an unexpanded shell placeholder** (`$(PLAN)`) instead of the plan text. The subagent made no edits, but a cycle was wasted. Paste briefs inline, or point to a durable file path, and re-read the prompt before sending.
- **A wrong premise in a draft plan, caught only on self-review:** the research `gen 1 11` is end-exclusive (seeds 1–10), and the boring-frame counter was per process. Verify every numeric claim against the code before briefing.
- **The approved ES design (Run B′ from transfer-handoff §7.3) was not checked against env.js's start-score floor.** desktop-1 found that it never starts inside 400–800. Check that a curriculum actually puts mass where its hypothesis needs it before calling it a test of that hypothesis.
- **A PushNotification was attempted while the user was at the terminal.** It was not sent, which is harmless. Use notifications only when the user is away.
