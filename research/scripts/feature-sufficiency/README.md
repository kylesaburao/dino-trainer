# Feature-sufficiency probe scripts

These are the scripts behind `research/feature-sufficiency.md`, copied from a session scratchpad that no longer exists. They are kept for reproducibility. They are not maintained tools.

- **Absolute paths:** the scripts and `.sh` drivers still contain absolute paths into the old scratchpad (`/private/tmp/claude-501/.../scratchpad/features/`). Change those paths before re-running anything.
- **Generated data:** the probe datasets (about 1.7 GB) were not kept. Re-generate them with `probeA.js` (crash windows) and `cloneB.js` (solver demonstrations).
- **Saved outputs:** the solver-clone models (`m_*.json`: 44-32-32-3 and 44-8-8-3 clones of `test/solver.js`, with feature variants), the rollout results (`rf_*`, `rw_*`), the training logs, and the crash summaries (`crash_early.json`, `crash_late.json`, `analyzeA.json`) are kept locally, outside git, in `runs/exp-bc-clones/` on the Mac.
