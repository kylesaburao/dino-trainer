# Follow-up questions

Open questions the user has raised, kept here until each has a pre-registration or an answer. For each one, the entry says why it matters, what is already known, the cheapest test that would answer it, and what it depends on. Add new questions at the end. When a question is answered, link the result and keep the entry.

Process: AGENTS.md "Research method". A question becomes an experiment only when its answer changes a decision.

---

## Q1. Can a larger model trained on the policy be distilled into a smaller model (policy → large → small)?
Raised by the user, 2026-10-10. Status: **open; waits on the depth × width sweep** (`research/prereg-bc-sweep.md`).

### What it means here
- "Policy" is the teacher, `test/solver.js`: the exact solver that reads hidden state.
- **Today:** solver → student. A net (the shipped one is 44-32-32-3) is trained on the solver's hard labels, jump or not, frame by frame (`bc-gen` → `bc-train`).
- **Proposed:** solver → **large** clone → **small** student. The small net learns from the large clone rather than directly from the solver.

### Why it could help, when the solver is already a perfect teacher
The solver is exact, so a large intermediate can't add correctness. It can only change *what signal the small net sees*:
1. **Soft targets** (Hinton, Vinyals & Dean 2015, "Distilling the Knowledge in a Neural Network"; to be cited from source when pre-registered).
   - The large clone's logits say *how close* a frame is to the jump/no-jump boundary. The solver's hard labels don't.
   - A small net trained on soft targets may place its decision boundary better per parameter, and reach a given agreement with fewer FLOPs.
2. **The same input space as the student.** The large clone sees only the 44 legal features, as the student does. Where the solver's action depends on hidden state, its hard labels look like noise from the student's side: the imitation gap (`research/distillation-robustness.md`). The large clone's outputs are a function of the 44 features only, which may be an easier target.
   - Counterpoint: the feature-sufficiency work found the features nearly sufficient (one aliased state, "jump key held"), so this gain may be small.
3. **Cheap labels on any state.** The large clone can label states the small student visits. That is DAgger without querying the solver. But the solver can already be queried from student states on ground frames (robustness scan, deviation 5), so this mainly matters for mid-air states the solver can't label.

### What is already known
- Small nets trained directly on the solver already work in some cases. Probe B's 44-8-8-3 reached 1M on all its games (`research/feature-sufficiency.md`), but with a higher training loss (0.0058–0.0076 against 0.0044 for 32-32).
- So distillation pays off only if the sweep shows a gap: small nets that **fail** tier B, or need many more FLOPs to reach the agreement thresholds, when trained directly.

### Cheapest discriminating test (draft; pre-register before running)
- **Precondition, from the sweep:** a "large" configuration that passes tier B with high held-out agreement. There must also be a "small" configuration that is *marginal* when trained directly: it fails tier B on one seed, or never reaches T2. **If every small configuration already passes tier B on both seeds, Q1 has no decision to inform. Close it as "not needed at this scope".**
- **Arms**, all on the same demonstration states, with identical small shape, seeds and hyperparameters:
  - (a) small trained on the solver's hard labels, the sweep's own model;
  - (b) small trained on the large clone's soft targets, cross-entropy against the softmax at temperature τ ∈ {1, 2, 4}, chosen in advance;
  - (c) small trained on the large clone's hard argmax labels. This control separates the soft-target effect from the relabelling effect.
- **Measures:**
  - the sweep's time-independent metrics: held-out agreement against the **solver** on seeds 41–60, FLOPs to T1 and T2, and the tier A and tier B game bars;
  - the min top-2 logit margin, which matters for CSS faithfulness in small nets.
- **Readings:**
  - (b) beats (a) on tier B or effort, and (c) doesn't: soft targets help.
  - (c) ≈ (b) > (a): the gain comes from relabelling on the student's input space, i.e. the imitation gap.
  - No difference: direct cloning is enough at this scope. Close Q1.

### Implementation needs
`bc-train` has no soft-target mode. The test needs:
- a demo format with per-row target distributions, or the large clone's logits computed on the fly;
- a loss option in both backends, behind an off-by-default flag. Defaults must stay identical: the bc tests and the `py-s1` exactness control.

Don't build any of this before the precondition above holds.

### Depends on
`research/prereg-bc-sweep.md` results: which shapes pass tier B on both seeds, and the per-configuration effort.

---

## Q2. A teacher that ducks: "v3"
Raised by the user, 2026-10-10. Status: **open; not started.**

**Naming (the user's convention, 2026-10-10).** Everything up to this point is **"v2"**: the jump-only solver teacher, its demonstrations, the shipped clone and the sweep. A ducking solver, and training on it, is **"v3"**.
- This is a *project or teacher generation*, not a game-rules version. The game rules stay v2 (`createGame({ version: 'v2' })`, `?game=v2`).
- In code, run dirs and docs, write it as "teacher v3" or "pipeline v3". Never write it as `gameVersion: 'v3'`. A game version v3 would mean different rules, a different `HI_KEY` and new parity goldens, none of which this needs.

### Why the v2 teacher never ducks
- The fairness model (`core.js` `requirement()`, comment lines 255–290) puts every obstacle in one of two classes:
  - must be **grounded** during its overlap: high pterodactyls;
  - must be **airborne**: cacti, low and mid pterodactyls.
- The generator then spaces obstacles so that a jump-only plan always exists. It says so itself: "Fast-drop and ducking only add options, so the model is conservative."
- `test/solver.js` executes that jump-only plan, so it never presses DOWN. The clone copies it, so its duck output is never trained (class weight 0).

### What ducking would change
| Pterodactyl (hitbox y) | Standing | Ducking | Options |
|---|---|---|---|
| low, y = 108 (box 121–129) | hit | hit | must jump |
| mid, y = 82 (box 95–103) | hit | clear | **jump or duck** |
| high, y = 58 (box 71–79) | clear | clear | stay grounded |

(From the `tools/wall-diag.js` header; verify against `obstacleBox` and the duck box before use.)

**Claim to verify (reasoned from the code, not tested):** ducking under a mid pterodactyl should never be less safe than the planned jump.
- The overlap interval O_P sits inside the planned airborne interval A_P.
- Every neighbouring requirement is already disjoint from A_P, or separated from it by at least minGround.
- So staying grounded and ducking through O_P demands less of the timeline than the jump does.

Open details: when DOWN can be pressed and released relative to landing from the previous jump, `applyKeys` semantics, and whether the jump key may still be held.

Fast-drop (DOWN in mid-air) is a separate, later option. probe A found the ES model's late crashes were landings into fast, high pterodactyls that a fast-drop would have survived. Including it means changing the arc model, so keep it out of the first v3.

### Cheapest discriminating sequence (pre-register before running)
1. **Necessary condition, no training.** Write `createDuckSolver`: duck under every mid pterodactyl, otherwise behave like `createSolver`. It must reach 1M with **0 crashes** over thousands of seeds, the same bar the v2 solver meets.
   - A crash means the claim above is wrong somewhere. Diagnose it before going further.
   - Measure the share of frames with DOWN, so we know there is enough duck signal to learn.
2. **Learnability.** Run `bc-gen` with the v3 teacher; three classes are now present. Then `bc-train` at the shipped shape and the smallest shape that passed the v2 sweep. Run tier A, then the 1,000-game scan.
   - Check per-class recall for duck.
   - The 44 features include pterodactyl height class, so the decision should be observable. Verify that per frame: does the solver's duck decision depend on anything outside `observe()`?
3. **Comparison with v2, paired.** Same shapes and the same scan seeds. Compare scan failures, held-out agreement, effort-to-criterion, and the min top-2 margin. A third class may thin the margins, which matters for CSS faithfulness.
4. **Ship (only with the user's approval):** certification on fresh held-out offsets, paired with the shipped v2 clone, plus the browser check with CSS agreement.

### Decision it informs
Whether the shipped AI should become a v3 clone, which ducks and looks more like a human player. Shipping v3 needs:
- reliability on the same bars as v2 (0 failures in the scans and 150/150 certified);
- the user's approval.

A v3 clone that is equally reliable is a presentation and behaviour change, not a reliability gain.

### Depends on
Nothing blocking. It can start once the Mac is free of the v2 sweep, which is now in its last scans.
