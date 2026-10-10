# Handoff: the behaviour clone ships (2026-10-10, 16:53 UTC, orchestrator mac-2)

Read after `handoff/2026-10-10T1626Z-orchestrator-transfer.md` and the retrospective. The full decision record is in AGENTS.md, "Decision log", 2026-10-10.

## 1. User decisions (verbatim)
- On a question about contamination: the orchestrator explained that the clone's labels come from the privileged solver but its inputs are only the 44 allowed features. The user then said: **"Shipping the clone is allowed"**.
- **"Clearly document this pivot in architecture"**: AGENTS.md now has a "Model provenance" section, a clarified anti-cheat rule, and a "Decision log".

## 2. What changed on main
- `wt/wasm` was merged (`cdf6e53`). desktop-2's x86 checks were all IDENTICAL, with a 1.65–1.70× speed-up. The Mac browser check passed 38/38. `--wasm` is now in the AGENTS.md flag map.
- Research note 8, `research/validation-selection.md`, was merged. Selection bias is real, but fixing it gains only about 0.4 points of pass rate. Designs A–E are described there, not implemented.
- **Ship:** `model.js` and `model.css` now hold `runs/bc-v2-h32x32/best.json`, the clone `m_base_s1`, 44-32-32-3. `runs/bc-v2-h32x32/evals.jsonl` holds the 150-seed certification and the standard 25-seed eval.

## 3. Certification (pre-registered in `runs/exp-bc-clones/PREREG-cert-175-324.md`)
| Model | Offsets 175–324 |
|---|---|
| clone s1 (shipped) | 150/150 |
| clone s2 | 150/150 |
| ES gen 970 | 143/150 (5 early, 2 near 994k) |

McNemar one-sided p = 0.008.

**The next fresh held-out offset is 325.**

## 4. Preserved data (gitignored)
- `runs/exp-bc-clones/`: the clone params (`m_*.json`), wrappers (`model_base_s*.json`), certification outputs and the pre-registration.
- `runs/exp-bc-clones/scratch-features/`: a 1.7 GB copy of the old orchestrator's probe workspace, including the `B/d1–d4.bin` demonstration data.

## 5. Pruned-branch register changes (retrospective §6.4)
- **Behaviour-clone-from-solver:** **adopted and shipped.** Reopen if a clone failure mode appears that the solver lacks, or if the inputs, core or game version change (then retrain).
- **Feature change in `policy.js`:** pruned (unchanged from the 16:45Z addendum).
- **Pure-ES second wall:** research only now. desktop-1's diagnosis is in `wt/desktop-1` `74284d9` (tools/). The second wall depends on the seed:
  - one seed is blind to low pterodactyls, and its population ties;
  - the others jump into mid pterodactyls and climb slowly.

## 6. Open next steps (none of them block the shipped model)
1. Review and merge `wt/desktop-1` (`74284d9`, tools/ only).
2. Promote the BC pipeline from research scripts into `trainer/`. It currently needs Python with numpy, while the repo policy is no npm dependencies. Ask the user before adding a Python dependency to the main pipeline.
3. Decide whether ES research continues. Options: ES fine-tuning starting from the clone, or the layer sweep. Each needs pre-registration.
