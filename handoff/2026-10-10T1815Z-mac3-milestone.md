# Handoff: mac-3 milestone (2026-10-10, about 18:15 UTC)

mac-3 is the orchestrator, by the user's order relayed by mac-2. Read `handoff/2026-10-10T1711Z-orchestrator-transfer-mac3.md` first; this file updates it.

## 1. Merged and pushed to main since the transfer
| Commit | What |
|---|---|
| `d4a1224` | `wt/bc`: BC as a first-class trainer pipeline. Tests 72/72 with `PYTHON` set; smoke 17/17; bit-identity IDENTICAL, 7/7, against `4274a3d` |
| `5f0edc8`, `f040434` | Robustness scan, pre-registered and run (`research/prereg-bc-robustness.md`) |
| `7815cdd` | Research note 9, `research/distillation-robustness.md` |
| `da74010`, `7931881` | End-to-end pipeline check, pre-registered and run (`research/prereg-bc-pipeline-e2e.md`) |
| `69279e0` | `research/retrospective-research-value.md`: the user's question "did the research find anything that would have sped up development?" |

## 2. Results
- **Robustness scan** (seeds 30,000,000–30,001,999, full v2 games to 1M):

  | Model | Failures |
  |---|---|
  | shipped clone C1 | 0/2,000 |
  | replicate C2 | 0/2,000 |
  | Node clone N1 | 0/2,000 |
  | ES positive control | 18/500 |

  - **D1:** robustness work (DAgger, noise injection) is **pruned at scope**. The shipped clone's rate is < 0.15% (one-sided 95%).
  - **D2:** the Node backend is acceptable as the research default.
  - The oracle-disagreement tooling has never been exercised on a real clone death.
  - ES late deaths fall at 968k–983k: a high pterodactyl at speed 21.96, with the dino in the air.
- **End-to-end pipeline:** `bc-gen` (0.55 s; 2,096,268 rows; weight-1 rows equal to the prototype's) → `bc-train` Python (19 s) and Node (161 s). Both reach 25/25 validation and 0/1,000 in the scan (seeds 30,002,000–30,002,999). **Pipeline validated.**
- **Seeds used for scans:** 30,000,000–30,002,999. Use 30,003,000 or later next. **Held-out offsets are untouched: the next fresh one is still 325.**

## 3. Peer decisions made
- **desktop-1:** its stop-rule deviation was confirmed. The held-out stop counts as confirmation only, and the event is the first validation with valMedian ≥ 2000. Its batch (12 runs) is running; the result is pending.
- **desktop-2:** natural-ramp step 2 was approved.
  - `test/solver.js` stays the only teacher (the move to `trainer/teacher.js` was declined).
  - The code is done locally (`387e553`, 9 tests). The harness, smoke, determinism checks and cost measurement wait for desktop-1's batch.
  - Its final report must document the `--window-adapt` stratum for ramp episodes and give the ramp cost as a share of a generation.

## 4. Pruned-branch register updates (retrospective §6.4)
| Branch | Verdict | Scope | Reopen if… |
|---|---|---|---|
| Clone robustness (DAgger, noise injection) | **pruned at scope** | 44-32-32-3, v2, demos from seeds 1–40, full games from 0; 0/2,000 per clone, 3 clones | a field failure; a new game version or `observe()`; a smaller shape; CSS diverging from JS |
| ES fine-tuning from a clone, *for reliability* | **not needed** | as above | the same |
| BC pipeline (`bc-gen` → `bc-train`, both backends) | **validated** | 44-32-32-3, v2, default gen settings, seed 1 | changes to the core, `observe()` or the game version |

## 5. Next steps
1. **Clone research has no open decision left at the reliability level.** Possible next topics need a goal from the user:
   - smaller shapes, which only matter if model size or CSS-page speed matters;
   - data-efficiency curves;
   - a v1 clone.
2. Apply desktop-1's fixed readings when it reports, then gate desktop-2's harness run on the CPU being free. Merge `wt/desktop-2` only with the harness IDENTICAL plus tests and smoke.
3. Housekeeping: `/Users/kyle/Desktop/dino-game-bc` (wt/bc, merged) can be removed with `git worktree remove` once nothing references `runs/exp-bc-repro` there. The robustness scan copied the Node clone into `runs/exp-bc-robust/`.
