# Retrospective: multi-machine training-efficiency work (2026-10-10, about 14:50–16:35 UTC)

Written by the outgoing orchestrator (`dino-game-3e`) for the next one (`mac-2`) and for the user. Its purpose is to sharpen how experiments are chosen and designed, not just to list what happened. Facts are cited from `handoff/2026-10-10T1549Z-orchestrator.md` and `handoff/2026-10-10T1626Z-orchestrator-transfer.md`.

---

## 1. What was done

| Outcome | Status |
|---|---|
| Cross-machine setup: 2 desktop sessions plus Mac subagents, worktrees per session, branch → main integration | worked |
| Cross-architecture determinism: x86 reproduces the Mac to 1e-9 over 800 generations | worked; it made the desktop's compute fully usable |
| `test/bit-identity.js`: a durable 6-scenario harness | merged |
| Milestone 2: gated window adaptation and geometric-mean fitness | merged, verified, **not yet used in training** |
| Milestone 1: tie-rate σ controller | built, **failed**, reverted |
| `--wasm` SIMD forward pass, 1.68× faster, bit-identical | built; x86 check pending |
| Seed survey: plateau escape is a seed lottery; median about 1,200 generations; K = 2–3 seeds at a 1,500-generation horizon | done; reframed the plateau problem |
| Second wall identified: 500–700 points (first pterodactyls); 6 of 7 certification failures are there | being diagnosed |
| Hygiene-trained model certified 71/75 against the shipped model's 72/75 (offsets 100–174) | tied |
| 7 research notes in `research/` | merged |

**Net:** good infrastructure (determinism, harness, a 1.68× speed-up) and a better problem statement. **No model better than the shipped one yet.** The cost was about 40 experiment runs, many of which couldn't have discriminated between the options they tested.

## 2. What went wrong, and why

### 2.1 The σ controller was built on an assumption the data already contradicted
- **What happened:** the controller assumed the tie fraction q falls steadily as σ grows (a Robbins–Monro argument from a research agent). Archived HANDOFF §6.1's own table showed ties *rising again* at σ ×3.375. The brief went out, desktop-2 implemented it, and 14 replays failed: σ ran away to its maximum and every candidate died at the first obstacle.
- **Root cause:** the derivation's premise was treated as established because it came with a "proof sketch". Nobody checked it against the evidence already on hand before spending implementation and compute.
- **Cheaper path that existed:** a static check, measuring q(σ) on frozen plateau snapshots across a σ grid, takes seconds with no trainer code. It would have shown the U-shape.

### 2.2 The first replay test was mis-sized, so everything was censored
- **What happened:** the 400-generation "plateau proxy" from the archived HANDOFF §9 was used as given. The shipped config escapes at about 776–792, so the known-good arm could never succeed within 400. All 26 runs at 400 generations (and many at 900) were censored, and the adoption criterion's second clause was vacuous.
- **Root cause:** there was no **positive control**, an arm that must succeed inside the horizon. The horizon came from an old proposal, not from known escape times.

### 2.3 Interpreting n = 1 and confounded checks
- The archived HANDOFF had already reclassified the stall rule's 776 → 252 gain as chaos. The decay check (step 1) was still run although it was known to be confounded (resumed mid-climb). It cost 6 minutes and returned "inconclusive", which was predictable.
- **Root cause:** the plan was followed without re-asking "what result would change a decision?". If no outcome changes a decision, don't run it, or run it only as a free side job.

### 2.4 The plateau was framed as a tie problem before the baseline's variance was known
- Much effort (mid ranks, stall rule, controller) went into tie mechanics. The seed survey then showed escape is mostly **seed luck**: about 25% by generation 900, independent of arm. The statistics agent's advice from the archived HANDOFF §8, "measure the variance once" (6 seeds of the incumbent), was deferred until after the controller work.
- **Root cause:** treatment comparisons were designed before knowing between-seed variance. That should always come first; it is cheap (plateau generations cost about 30 ms).

### 2.5 Errors in briefs that implementers had to catch
- The geometric-mean claim (≈ 0 when any stage fails) was false for a single stage: the value is (1/span)^(1/E).
- The window "shrink" could grow a window that was below adaptWindow's floor.
- Test scenarios used invalid values (`--start-min 1000` violates checkCurriculum).
- Saturated ties, where both members reach the cap, would have driven the controller; this was caught only at brief-writing time.
- **Root cause:** claims were copied from the archived HANDOFF into briefs without checking them against the code's constraints (`checkCurriculum`, `adaptWindow`). Every numeric or behavioural claim in a brief should be checked against code or data, or marked as unverified.

### 2.6 Permission and authority assumptions
- desktop-2 was asked to push while AGENTS.md said "Never push". It correctly refused until the user confirmed in its own session.
- The Edge/WSL browser test was started before asking whether browser use was available. The user stopped it.
- The orchestrator analysed inline until the user asked it to delegate, and its context grew faster than necessary.
- **Lessons:**
  - Check repo rules against the user's instructions *before* briefing peers. Surface conflicts to the user first.
  - Ask about environment capabilities before spending effort on them.
  - Delegate by default.

### 2.7 Tooling slips (each cost a retry)
- `pgrep -f "<pattern>"` in a wait loop matched its own shell. Wait on a **PID** (`kill -0 $pid`) instead.
- In zsh, `$F` holding several flags does **not** word-split. Use arrays or literal arguments.
- The `rm` hook blocks any command containing `$`, globs, `~` or braces. Keep `rm` commands literal and separate.
- The harness's first run overlapped a trainer because `pgrep` didn't stop the launch. The harness now refuses to start while a trainer runs.
- Cross-session delivery by bridge ID was "unconfirmed". Duplicate "did you get it?" messages were sent, and the briefs had in fact been received. Treat silence as unknown, but don't resend content blindly.

### 2.8 Research topic chosen by authority, not by the decision it serves
- The first research note focused on Karpathy at the user's request and found **essentially no ES content**. The ES-relevant material came from Salimans 2017.
- Later notes chosen by *open decision* were far more useful: WASM (led to 1.68×), rare-failure robustness, demo bootstrapping.
- **Lesson:** keep the user's preferred sources, but pick topics by the next decision to be made. Ask each note to end with "which decision this changes".

### 2.9 Held-out seed offsets are a finite resource
- Offsets 100–174 went on a comparison of two models that came out statistically tied (71 vs 72 of 75). That was informative, but 75 seeds can't show an improvement over 72/75: one-sided Fisher p is about 0.12 even at 0/75 failures, per research note 7. About 150 fresh seeds are needed.
- **Lesson:** size certifications for the effect being claimed. Plan offset usage in advance (175, then 325, …).

## 3. Assumptions not to make again

1. **"A knob has a monotone effect"** (σ → ties, decay → speed). Measure the response curve on frozen snapshots first.
2. **"A derivation or proof sketch in the handoff is correct."** Re-verify its premises against existing data before acting.
3. **"The proposed horizon or budget is adequate."** Derive it from a positive control's known success time.
4. **"A difference between two runs is a treatment effect."** Between-seed sd(ln T) is about 0.4–0.7. One run per arm detects only ≥ 5–7× time effects or ≥ 0.4 pass-rate effects.
5. **"The plateau is caused by ties."** It is mostly a search lottery. Ties are a symptom.
6. **"The fix should target the plateau."** The second wall (pterodactyls at 500–700) causes most certification failures and stalls more escaped seeds.
7. **"Peers will do what the orchestrator asks."** They follow their own user's rules and docs. Align the docs and the user's authorisation first.
8. **"A sent message was read."** Delivery is unconfirmed on Remote Control routes. Rely on replies.
9. **"75 fresh seeds can show an improvement over 72/75."** They can't. Use ≥ 150 for that claim.

## 4. Refined process for defining next steps

For every experiment, write a short pre-registration (in the brief or the handoff) before running it:

1. **Decision:** which concrete choice it informs (adopt flag X, change features, size K…). If no outcome changes a decision, skip the experiment.
2. **Premise check:** list the assumptions behind the approach, and check each against existing data or a seconds-scale static probe.
3. **Positive control:** include an arm known to succeed within the horizon. Size the horizon from it, plus margin.
4. **Variance first:** know the between-seed spread of the primary outcome for the incumbent before comparing arms.
5. **Seeds from power:** choose n per arm for the effect size worth detecting. Use paired seeds across arms.
6. **Primary metric and reading:** write down how each plausible result will be read, including "inconclusive".
7. **Advisor at design time:** consult before launching (it caught saturated ties, cross-architecture risk and the mis-sized test), not only before implementing.
8. **Cheapest discriminating test first:** static probe → short replay → full run → certification.
9. **Certification budget:** fresh offsets sized for the claim, recorded in the handoff.

Operating rules for the orchestrator:
- Delegate code and analysis by default.
- Keep briefs self-contained, with allowed files, a stop condition and verification steps.
- Verify every claim in a brief against code.
- Wait on PIDs, not process-name patterns.
- Write a timestamped handoff at each milestone.

## 5. What this implies for the next steps (`handoff/2026-10-10T1626Z-orchestrator-transfer.md` §7)

- **The pterodactyl wall is the priority, ahead of more plateau mechanics.** Before choosing between feature changes, a curriculum targeting 400–800 points, or behaviour-cloning init, run the cheap discriminators first:
  - the feature-sufficiency probe (in flight);
  - desktop-1's death breakdown (in flight);
  - a static check of whether the current policies ever duck under low pterodactyls.
- **Plateau:** stop tuning tie mechanics. Use seed racing (K = 2–3, about 1,500 generations) as the default exploration lever, and treat escape as time-to-event.
- **Run B′ and the layer sweep** must each pre-register their decision, positive control, horizon and seed count (§4) before launch. Run every experiment with `--wasm` once it is merged.
