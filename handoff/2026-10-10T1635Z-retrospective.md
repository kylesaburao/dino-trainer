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

## 6. Scientific method and pruning the search space without false negatives

The search space (algorithm knobs × curriculum × features × shapes × seeds) is too large to explore by trial runs. The goal is to design tests whose results **rule out whole regions** for now, and to record each ruling-out narrowly enough that a test that was wrong, under-powered or mis-scoped doesn't permanently close a valuable region. Today's errors (§2.1, §2.2) were exactly cases where a missing or invalid test let effort go into a region that a cheap test would have pruned, or would have shown had to stay open.

### 6.1 Working method: hypothesis-driven, strong inference
- **State competing hypotheses explicitly**, with predictions that differ. Example: the 500–700-point wall comes from one of
  - H1: missing information in the features;
  - H2: a search difficulty (the policy could represent it but ES hasn't found it);
  - H3: an untrained skill (it never ducks), because the training distribution rarely shows low pterodactyls.

  This is Platt's "strong inference" (Science, 1964): devise the *crucial experiment* whose outcome excludes one or more hypotheses, run it, and recurse.
- **Make every hypothesis falsifiable** by writing down beforehand what result would refute it. A hypothesis that survives every possible outcome is not being tested.
- **Each run must be able to discriminate.** Before a run, list the hypotheses still alive and the result each predicts. If every live hypothesis predicts the same result, the run is uninformative; redesign it or skip it.

### 6.2 Pruning tools (cheapest first)
1. **Bounds and back-of-envelope checks** (Fermi or Amdahl reasoning). These rule out directions with no runs at all.
   - Today: GPU and Rust were pruned by episode seriality and the MLP's share of runtime.
   - `actionRepeat` was bounded at 1.67×.
   - Cutting training episodes early can save at most about 8%.
2. **Static probes on frozen artefacts.** Measure a response curve (σ → tie fraction, features → separability) on saved θ, with no training.
   - Seconds to minutes.
   - It would have pruned the monotone-σ controller before any code was written.
3. **Oracle and upper-bound tests.** Give a component privileged help and see whether the *ceiling* moves.
   - Example: an oracle feature computed from forbidden state.
   - If even the oracle doesn't help, the direction is pruned conditionally.
   - If it does help, measure how much of that gain the allowed features recover.
4. **Necessary-condition tests.** If an approach can only work when Y holds, test Y cheaply first.
   - Example: behaviour-clone initialisation only pays off if a clone on these features survives past 700 points in rollouts.
5. **Ablation and sufficiency.** Remove one component, or isolate it, to find which knobs matter.
   - **Factorial or fractional-factorial screening** (Plackett–Burman) tests many knobs at once across few runs, at the cost of confounding higher-order interactions; say which interactions are assumed negligible.
6. **Dose–response sweeps instead of single points.** A single σ or decay setting can sit at a bad spot on a U-shaped curve and falsely null the knob.
7. **Sequential designs.** Successive halving or racing, using **time-to-event** outcomes (plateau escape, passing the wall) and early stopping of arms that are clearly losing, with a stated stopping rule.
8. **Paired designs.** Use the same seeds across arms (common random numbers) to cut variance, so fewer runs reach the same power.

### 6.3 Safeguards against false nulls

A pruned branch is a claim and can be wrong. Before declaring "doesn't work":

1. **Test validity (positive control inside the pruning test).** Could this test have shown success if the idea worked? The 400-generation replay couldn't: the known-good arm needs about 780 generations. So its null meant nothing.
2. **Power.** What effect size could the test detect? Report the null as "no effect larger than X detected", with a confidence interval, never as "no effect".
   - Use equivalence tests (TOST) when the claim really is "no meaningful difference".
3. **Scope.** Record the exact conditions:
   - shape, seeds, horizon, σ, decay and game version;
   - the code commit;
   - the outcome metric.

   The null holds **only within that scope**. Extrapolating it to other shapes or phases needs its own argument.
4. **Concept vs implementation.** Distinguish "this specific design or parametrisation failed" from "the idea fails". The tie-rate integral controller failed; the general idea of adjusting σ on ties did not (CMA-ES's one-shot flat-fitness bump with a reset is untested).
5. **Chaos and n.** One run per arm can't null an effect smaller than about 5–7× in time, given between-seed sd(ln T) of about 0.4–0.7.
6. **Reopen triggers.** Each pruned branch records what new evidence would reopen it. Revisit pruned branches whenever an upstream fact changes, such as new features, a new curriculum, another shape, or `--wasm` making longer horizons affordable.
7. **Don't let a proxy veto the real objective.** If the pruning test measures a proxy (escape gen, BC accuracy, separability), check that the proxy plausibly predicts the real outcome: a fresh held-out pass rate to 1M. Example: BC accuracy is not the test; rollout survival is. A poor clone doesn't null feature changes, because the solver uses hidden information (the imitation gap).

### 6.4 Pruned-branch register (status at 16:35Z; keep this table updated in each handoff)

| Branch | Verdict | Scope of evidence | Power / validity caveat | Reopen if… |
|---|---|---|---|---|
| Tie-rate σ controller, integral form (`ln σ += K(q − q*)`) | **pruned (this design)** | q* 0.2/0.35, σmax 0.3/1.0, 32,32 and 32,32,32, seeds 1–3, 400 generations, mid + decoupled 0.01 | Mechanism understood (q(σ) is U-shaped); not a power issue | a σ rule that targets only *unsaturated, above-first-obstacle* ties, or one with a reset (CMA flat-fitness style) |
| `--stall-rule` (skip mode) | **pruned at wd 0.01** | 32,32, seeds 1–3, 900 generations | Escaped once at wd 0.1 (n = 1, archived §6.1) | a non-skipping inflate mode; a different decay |
| Tie hygiene (mid + decoupled) as a *plateau-escape* lever | **no effect detected** | 32,32, 3 seeds per arm, 900 generations | Weak (n = 3, censored); only large effects could be seen | any arm whose escape rate clearly beats the 25%-by-900 baseline over ≥ 10 seeds |
| Decoupled wd 0.1 | **disfavoured** (slows the climb 3–6×) | Run A vs hygiene s3 and shipped, n = 1 each | Different seeds | paired runs show no difference |
| GPU / Rust port | **pruned by bounds** | Episodes up to 1.9M sequential frames; MLP is 80% of the cost | Bound argument, not measured | much shorter episodes (≤ 1e5 frames) or a much larger population |
| `actionRepeat` 2 | **pruned** | ≤ 1.67× bound; it breaks the solver's 1-frame guarantee | — | v2 minimum window frames change |
| Early termination of training candidates | **pruned** | ≤ 1–8% saving | — | — |
| Feature change in `policy.js` | **OPEN, under test** | Feature-sufficiency probes in flight | A BC rollout failure is not a null (imitation gap) | — |
| Behaviour-clone-from-solver initialisation | **OPEN** | Literature only (`research/demo-bootstrapped-es.md`) | Necessary condition: a clone survives past 700 in rollouts | — |
| Curriculum targeting the 400–800-point zone | **OPEN** | Gated curriculum merged, untrained | — | — |
| Seed racing (K = 2–3, ~1,500 generations) | **adopted as the default plateau lever** | 12 seeds, hygiene arm | Kaplan–Meier, CIs wide | escape-rate data on other shapes |
