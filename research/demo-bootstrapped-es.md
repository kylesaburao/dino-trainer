# Bootstrapping ES from solver demonstrations (behaviour cloning, then ES fine-tuning)

Compiled 2026-10-10, branch wt/mac. No project code was changed and no trainer was run.

Labels:
- **STATED (source)**: read in the source text this session. Papers marked V in section 7 were downloaded and text-extracted locally (PDFKit), then grepped. Quotes are short and verbatim only inside quotation marks.
- **STATED (abstract/page)**: read only on the arXiv abstract page or a publisher page through a summarising fetch tool. Method details were not read.
- **STATED (search summary)**: reported by a web-search result summary only; the source was not opened.
- **INFERENCE**: my reasoning about this project. Not from any source.
- **UNVERIFIED**: could not be fetched or read.

Project facts (plateau ~54 points, median escape ~1,200 generations on 32,32, second wall at ~500-600 points, shipped model 72/75 on fresh offsets 100-174, next fresh offset 175) come from the task brief and `handoff/archive/2026-10-10T0742Z-HANDOFF.md` 4.2 and 4.5. I did not re-measure them. I read `test/solver.js`, `policy.js` and `trainer/cli.js` headers only to ground the design. A separate probe agent is testing whether 44 features suffice to clone the solver; none of its results are used here.

---

## 1. Summary: the decision-relevant takeaways

1. **No published precedent for exactly "BC then ES on a small MLP" was found.** The closest is Lorenc & Neruda 2025 (BC-pretrained Decision Transformer, then OpenAI-ES, Humanoid). STATED (source). Their result is mixed: pretraining forced a lower sigma and lr (0.01 vs 0.02 and 0.05), the model first got "broken", then recovered and "builds upon the seeded model". Search found no other direct ES+BC paper; absence of evidence, not proof. The BC-then-RL pattern itself is standard (AlphaGo, AlphaStar, DQfD, Go-Explore).
2. **BC from a fixed solver dataset has a known failure: compounding error.** STATED (source: Ross, Gordon & Bagnell 2011): for a policy with per-step error `eps` on the expert's state distribution, the extra cost can grow as `T^2 * eps`, and this is tight. Here the horizon `T` is very long (up to the 1M-point cap), so a tiny BC error matters. The fix in that literature is on-policy relabelling (DAgger) or noise injection (DART). INFERENCE for this project: one rare mis-timed jump near a pterodactyl is a crash, so "good BC accuracy" will not be enough; ES fine-tuning is exactly the phase that removes such errors.
3. **ES fine-tuning is the natural cure for covariate shift, because ES optimises the closed-loop score.** INFERENCE. BC supplies a start near the competent region (skipping the 54-point plateau and perhaps the 500-600 wall); ES then optimises the true objective on the states the policy actually visits. Go-Explore uses the same logic: its robustification phase uses a method that "is only optimized to maximize its own score, and not actually forced to accurately mimic the trajectory" (STATED, source).
4. **The privileged solver creates an "imitation gap" wherever its action depends on something the 44 features hide.** STATED (source: Weihs et al. 2021): with cross-entropy, the student learns `E[pi_teach(S) | obs]`, the teacher's policy with the hidden information averaged out. INFERENCE: solver actions that depend on the hidden spawn gap or lookahead beyond x<600 become soft or wrong labels in exactly those states. That may be benign (if one action dominates) or may be the second-wall states. The probe agent measures how big this is.
5. **Privileged-teacher pipelines handle the gap by (a) on-policy relabelling and (b) giving the student history, not by closing the gap.** STATED (source): Chen et al. 2019 train the student on-policy "using the privileged agent as an oracle"; Lee et al. 2020 use DAgger with a student that reads a history window so it can "(partially) recover" latent features. INFERENCE: the dino policy is memoryless (single frame, action repeat), so only (a) is available; no history fix without changing the evaluated policy.
6. **ES after BC needs a smaller step, and the paper that tried it saw an initial drop.** STATED (source: Lorenc & Neruda): same lr and sigma as from scratch "leads to a complete randomization of the model at the beginning". They lowered both to 0.01 and noted slower, "less exploration per iteration". INFERENCE: sigma 0.05-0.1 (the brief's proposal) is likely too large for an unregularised, saturated BC network; test a sweep (section 4).
7. **A BC-initialised policy that clears the cap on most seeds gives ES almost no gradient.** INFERENCE: with centred ranks and many tied scores at the cap, the antithetic pairs tie. This is the "ties" failure documented in HANDOFF 4.2, now at the top end instead of the bottom. Fine-tuning then needs the rare-failure tools in `research/rare-failure-robustness.md` (hard-seed mining), not plain mean fitness.
8. **Variants that keep the evaluated objective and evaluation unchanged:** BC init only (a different theta0); BC-regularised updates that vanish when switched off; dataset-conditioned restarts from solver states (Backplay-like). **Variants that change the optimised objective:** fitness = score + imitation-agreement term (DQfD/Nair/POfD pattern). Evaluation stays unchanged in both; only the second breaks the repo's "reward = capped score, no shaping" training rule. That line is a project-owner decision (section 5).
9. **DQfD and Nair et al. both keep the supervised term during RL and treat annealing as a design question.** STATED (source): DQfD keeps demonstrations in the replay buffer permanently and argues that "Even after pre-training, the agent must continue using the expert data"; Nair et al. use a Q-filter that "automatically disables the BC loss when a better action is found". INFERENCE: ES has no critic, so a Q-filter analogue would have to be an episode-level one (apply the BC term only when the solver's cumulative outcome on a state set beats the policy's). That is more machinery than the experiment needs on the first pass.
10. **Cheapest informative experiment:** BC init, then unchanged ES (no extra loss), small sigma sweep, 3+ trainer seeds per setting, compared with the baseline escape distribution and the shipped 72/75. Decide by generations to the first and second wall and by fresh held-out pass count; the statistics are in section 4 (150 fresh seeds with zero failures separates from 3/75 at one-sided Fisher p about 0.036; 75 seeds cannot).
11. **Use the solver only for training data.** The policy still reads only the 44 allowed features at decision time. The anti-cheat rules in `AGENTS.md` bind observation extraction, and the solver's privileged reads happen only in the data-generation script. INFERENCE: confirm that interpretation with the project owner before shipping any BC-derived model, since the shipped `model.js` would then embed behaviour distilled from privileged information.
12. **Main risk to guard against:** a BC init that scores well on in-distribution seeds but is brittle, then ES drifts or collapses (Lorenc & Neruda saw an early deterioration). Always keep the BC theta0 as a baseline in the eval table and checkpoint the best by validation score as the trainer already does.

---

## 2. Per-question findings

### Q1. BC as initialisation for ES / neuroevolution; reported uses and failure modes

**Direct ES precedent.**
- STATED (source): Lorenc & Neruda (arXiv 2501.13883) pretrain a Decision Transformer by behaviour cloning towards a feedforward SAC policy ("a feedforward model trained using SAC"), then train it with OpenAI-ES on Humanoid. The motivation given: for a large model, sampling "in a neighborhood of a randomly initialized model" may give too little information.
- STATED (source): costs they report. Virtual batch normalisation conflicts with pretraining, so they dropped it for the ES phase. Using from-scratch lr and noise std erased the pretraining; both were set to 0.01 (Table: noise 0.02 and lr 0.05 for the non-pretrained runs). Performance first "gets 'broken'" over the first few iterations, then improves "much sooner than when no pretraing is utilized". Rollouts of the initial and final models show similar gaits, so ES "builds upon the seeded model". Overall they conclude pretraining "showed multiple flaws".
- STATED (source): they hypothesise the drop arises because gradient training and ES "follows a different overall goal" (text cut off in my extraction; I did not read the rest of the hypothesis). The pretrained model and its SAC teacher scored "around 4" (their fitness units, unclear scale). Their conclusion is a single-environment, single-architecture result with a transformer, not directly transferable.
- Caveat: that paper's code and per-run numbers were not read; the above is from its text only.

**Hybrids of evolution and gradients.**
- STATED (abstract/page): Mustafaoglu, Pingali & Miikkulainen, Evolutionary Policy Optimization (arXiv 2504.12568) combine neuroevolution with policy gradients; claims better sample efficiency than PG or evolution alone on Pong and Breakout. STATED (search summary, not verified at source): the search tool described it as PPO pretraining, then seeding the population. I did not read the method, so I do not rely on that detail.
- STATED (search summary): MLGO warm-starts an RL policy by behaviour-cloning a compiler heuristic; not ES. Not used further.
- No paper on "ES from demonstrations" for small MLPs was found by two searches. UNVERIFIED that none exists.

**BC then RL, as analogy (not ES).**
- AlphaStar: STATED (abstract/page, DeepMind blog): the network was first trained by "supervised learning from anonymised human games"; the initial agent beat the built-in Elite AI in 95% of games (the blog calls this "around gold level for a human player"); these agents "seed a multi-agent reinforcement learning process", and the league agents ended well above the Grandmaster level (the blog's wording, paraphrased; not a verified quote). The blog also says competitors must not forget how to beat earlier ones, which is an anti-forgetting design, but that is population-level, not parameter regularisation.
- AlphaGo: STATED (search summary): an SL policy network trained on expert moves, then refined by self-play RL. Not opened at source.
- DQfD: STATED (source). Pre-training uses four losses, including a supervised large-margin classification loss, whose weight is set to zero on self-generated data; demonstrations are retained permanently in replay. The authors write that supervised learning alone "is not successful" on the demonstrations (their phrase "Simply doing supervised learning on the human demonstrations is not successful"), and that "The supervised loss is critical for the pre-training to have any effect."
- Nair et al. 2018: STATED (source). BC loss on demonstration batches added to the actor loss, Q-filter, and resets to demonstration states. Ablation: "Without the behavior cloning loss, the method is significantly" slower (text continues; I did not read the numbers).
- Go-Explore: STATED (source). Phase 2 robustification uses the Backward Algorithm of Salimans & Chen: start near the end of a demonstration, run PPO from there, back the start point up once the policy matches or beats the demonstration's reward. STATED (source): a single demonstration made this unreliable ("only 40% of our attempts" succeeded for level 1); using 10 demonstrations at once, all 5 robustification runs succeeded. Robustified policies reached 99.8% success rate in level 1 evaluations. Note this is demonstration-start-state curriculum, not weight initialisation.

**Compounding errors and covariate shift.**
- STATED (source: Ross, Gordon & Bagnell 2011, DAgger paper, Theorem 2.1, attributed there to Ross & Bagnell 2010): if `E_{s~d_pi*}[l(s,pi)] = eps`, then `J(pi) <= J(pi*) + T^2 * eps`; "this bound is tight". DAgger (stationary deterministic policy, no-regret online learning) is stated to reach error linear in `T` under the policy's own induced distribution (Theorems 3.3 and 3.4, with sample requirements `O(T^2 log(1/delta))` as stated). The mixing schedule `pi_i = beta_i * pi* + (1 - beta_i) * pi_i_hat` is used; they typically use `beta_1 = 1`.
- UNVERIFIED: Ross & Bagnell 2010 ("Efficient reductions for imitation learning", AISTATS) itself was not opened; it is cited here only through the DAgger paper's text.
- STATED (abstract/page): DART (Laskey et al. 2017) injects noise into the supervisor during demonstration so the data contains recovery, tunes the noise to approximate the learner's error, and reports being up to 3x faster than DAgger on Humanoid. I read the abstract only.
- INFERENCE for this project: a solver rollout that never crashes visits only perfect states. A BC student that mis-times one jump enters states with no training data (e.g. mid-air next to an obstacle it should have cleared earlier). Expect good per-step accuracy and poor long-horizon survival. Two cheap mitigations: add action noise to the solver while collecting (DART-style), and relabel states the student visits (DAgger-style). Both need the solver to produce an action from an arbitrary game state, which `test/solver.js` may not do (see section 3.2).

### Q2. DAgger with a privileged expert; the imitation gap

**What the privileged-teacher papers do.**
- Learning by Cheating, STATED (source): stage 1 trains the privileged agent from expert demonstrations; stage 2 trains the sensorimotor agent "off-policy by imitating the privileged agent on the same set of states", then "on-policy, using the privileged agent as an oracle that provides adaptive on-demand supervision in any state reached by the sensorimotor student." Claimed advantage: the privileged agent is a "white box" that can label any state the student reaches, so DAgger-like training is automatic.
- Lee et al. 2020 (Science Robotics, venue as recalled, not re-verified), STATED (source): the privileged teacher is a TRPO-trained policy with terrain knowledge; the proprioceptive student is trained by supervised loss on teacher actions and a latent embedding, using DAgger ("training data is generated by rolling out trajectories by the student policy. For each visited state, the teacher policy computes its embedding and action vectors"). Their stated hypothesis: the latent features "can be (partially) recovered from a time series of proprioceptive observations". Their ablation (Fig. E-G) compares a student trained directly (no privileged training) with a privileged-trained student; I read the caption, not the numbers.
- Asymmetric actor-critic (Pinto et al. 2017), STATED (abstract/page): critic gets full simulator state, actor only images. This is not imitation; it shows the other standard use of privileged state, in the value function. ES has no critic, so it is not applicable here.

**The imitation gap.**
- STATED (source: Weihs et al. 2021, NeurIPS): "when a student attempts to imitate an expert that is privileged with information not available to the student, the student learns a version of pi_teach in which this privileged information is marginalized out." Proposition 1: for the policy class containing all `f`-restricted policies, the cross-entropy-optimal student satisfies `pi_IL(o) = E_mu[pi_teach(S) | f(S) = o]`. Their 1D-Lighthouse example shows a student that is uniformly random over many states even with perfect optimisation. Their remedy, ADVISOR, "adaptively weights imitation and RL losses" and imitates only where the gap is small.
- STATED (source): they state the student "should imitate the teacher's policy only in settings where the teacher's policy can, in principle, be exactly reproduced by the student. Otherwise the student should learn via 'standard' RL." They also note the common practice of "continuing to train the agent using pure RL". They report experiments mainly on navigation with shortest-path experts and say more work is needed elsewhere.

**Relevance to the solver and the hidden spawn gap.**
- STATED (source: `test/solver.js` header): the solver "reads the planner, which the policy must never do", takes off "on the whole frame nearest the centre of its takeoff window, using that obstacle's frozen requirement (release frame)". It scans all obstacles within `tau <= 80` frames, including ones the policy may not observe.
- INFERENCE (needs the probe's confirmation): the solver's action at frame `t` is a function of the full game state. Part of that function is invisible to the 44 features (the planner's requirement for upcoming obstacles, anything at x >= 600 that `placeFairly` parked). In those states the BC student's CE-optimal output is the conditional mean of the solver's one-hot actions given the observation, i.e. a soft distribution. After argmax, it picks the modal action. Whether this is harmful depends on how often the modal action is also the right one. Frame-level jump timing within a takeoff window is the most likely place for this: a window several frames wide is mostly recoverable from `relx`, speed and type; the question is how often the solver's window centre depends on hidden state.
- Cross-reference: `research/observation-design.md` Q4 (INFERENCE there, not re-checked here) argues dino is Markov in the visible state, with the invisible spawn sequence and anything at x >= 600 the non-Markov remnants, and finds no benefit from frame stacking or recurrence when velocities are supplied. That is consistent with the argument here: the gap, if any, comes from the hidden items, and history would not fix it.
- INFERENCE: the gap is not an obstacle to using BC as an init, because the aim is only to land in a basin ES can then improve with the true objective. It is an obstacle to using BC as the end product. This matches the Weihs framing: imitate where reproducible, optimise the real objective elsewhere.
- INFERENCE: DAgger-style relabelling only helps if the solver can label arbitrary visited states. The solver holds internal state (`hold`) and depends on `game.keys.jump`; relabelling student states needs a stateless variant or a reconstruction of `hold` from the game state. See 3.2.

### Q3. Fine-tuning with ES without destroying the clone

**What is stated.**
- Lorenc & Neruda (STATED, source) is the only ES+BC evidence found: both lr and sigma reduced to 0.01 from 0.05 and 0.02 respectively (lr also from a SGDM optimizer, not Adam), with population 20,000, and weight_decay_factor 0.995 in all rows of their table. They report an initial drop in performance even with the reduced values.
- Safe Mutations (Lehman, Chen, Clune & Stanley 2018), STATED (abstract/page): mutation operators that limit behaviour change when weights are perturbed; SM-G scales each weight's mutation by the sensitivity of the network's outputs to it (output gradients, not error gradients). Their claim is for a simple GA on deep and recurrent nets. I did not read the details or any ES variant. Applying it to ES is an inference.
- Salimans et al. 2017 (STATED, abstract/page) describes the baseline ES; their sigma/lr choices are from scratch and were not examined for a pretrained start.
- STATED in-repo (`trainer/cli.js` header and HANDOFF 5.2): the trainer already has `--decay decoupled`, `--sigma-scale fanin`, `--sigma-anneal` and a `--stall-rule`. Defaults are sigma 0.1, weightDecay 0.005. Those are knobs for this experiment; I did not test them.

**What is not found.**
- No source found quantifying "the ES sigma needed from a competent policy vs from random". UNVERIFIED that a general rule exists. The only datapoint is Lorenc & Neruda's 0.01 vs 0.02 (and lr 0.01 vs 0.05) on a different problem, with a different scale of parameters.

**Project-specific reasoning (INFERENCE).**
- Relative noise: a perturbation of size sigma per coordinate on a 44-32-32-3 network (about 2,700 parameters) is a random step of norm about `sigma * sqrt(P)`. A BC network trained with plain CE tends to larger weights and sharper decision boundaries than a random init, so the same sigma is relatively smaller, but near a decision boundary (the takeoff frame) even small noise flips actions. The relevant failure is a one-frame change in jump timing, not weight norm.
- HANDOFF 4.2 shows from random init the exploration that escapes the plateau is the tie-driven random walk; from a competent init the same randomness is a hazard rather than a help. So "sigma smaller" is the natural first sweep; I propose 0.01, 0.02, 0.05, 0.1 (section 4). The brief's 0.05-0.1 range is the top end.
- Regularisation toward theta_BC options, ordered by cost:
  1. none (pure ES, rely on small sigma and rank-based selection); 
  2. decoupled weight decay pulling to zero (existing flag) is the wrong anchor: it pulls towards 0, not towards theta_BC. A decay towards theta_BC would need new code (`theta -= lr * wd * (theta - theta_BC)`); 
  3. add an agreement term (section 5).
- Selection already guards the shipped model: `best.json` is chosen by validation score, so a transient drop costs time, not quality. The BC theta0 should be evaluated first, so it is the floor.

### Q4. Mixed objectives (task score plus imitation term, annealed)

**Practice in the literature.**
- DQfD, STATED (source): `J(Q) = J_DQ + lambda1 * J_n + lambda2 * J_E + lambda3 * J_L2`, with the large-margin supervised loss `J_E` applied to demonstration data in both phases and not to self-generated data. Demonstrations are never overwritten. Their ablation is quoted above ("the agent must continue using the expert data"). This is a persistent, not annealed, imitation term.
- Nair et al., STATED (source): "a method of annealing away the effect of the demonstrations when the learned policy is better than the demonstrations" (the Q-filter); BC loss is applied only where "the demonstrator action is better than the actor action" by the critic. They also use fixed weights (lambda values listed in the text, e.g. `lambda1 = 10^-3`, `lambda2 = 1.0/N_D` in the setting I read) and resets from demonstration states. Their text says "Without the behavior cloning loss, the method is significantly" worse (I did not read the rest of the sentence).
- POfD (Kang, Jie & Feng, ICML 2018): STATED (search summary only): it enforces occupancy-measure matching between the policy and the demonstrations and "induces implicit dynamic reward shaping". The source was not opened. UNVERIFIED in detail.
- Gao et al. 2018 (NAC), STATED (abstract/page): losses of supervised and RL parts "conflict" in prior methods; NAC uses one objective for both phases. Not directly relevant.
- Weihs et al. (ADVISOR), STATED (source): weights imitation vs RL adaptively where the gap is small; relevant to an annealing schedule driven by how reproducible the teacher is.
- Backward Algorithm / Backplay (Salimans & Chen; Resnick et al.; Florensa et al.): STATED (abstract/page) for Salimans & Chen: "resetting to a state from a demonstration", reward maximised directly, with no imitation loss. This is already covered in HANDOFF 4.5 and is a no-shaping, start-state curriculum.

**Where the repo's rule bites (the line).** The repo rules, from `AGENTS.md`: "The reward is the capped score, with no shaping"; "Evaluation uses the unmodified game"; "Any curriculum is training-only." Cases:

| Variant | Optimised quantity | Evaluation | Breaks "no shaping"? |
|---|---|---|---|
| A. BC init only, then unchanged ES | capped score | unchanged | No |
| B. A plus smaller sigma / lr / decay-to-anchor | capped score | unchanged | No (optimiser settings only) |
| C. Training-only starts from solver states (Backplay-like, `startScore` already exists) | capped score from a start | unchanged | Same kind as the existing curriculum; already allowed ("Any curriculum is training-only") |
| D. Fitness = score + lambda * agreement(policy, solver), annealed to 0 | modified fitness | unchanged | Yes, it adds a term to the training fitness |
| E. BC gradient added to the ES update (Adam on `g_ES + lambda * g_BC`), annealed | score plus an imitation loss | unchanged | Arguable: the fitness is the same but the update is not the score gradient; treat as D |
| F. Reward-shaped solver (e.g. bonus for matching jump window) | shaped | unchanged | Yes, clearly |

INFERENCE: A, B and C keep both the evaluated objective and the training objective as capped score. D and E keep evaluation unchanged but not the optimised objective. The `AGENTS.md` wording is about the reward, so D/E need a project-owner decision on whether "no shaping" applies to the training fitness or only to what is reported. This note recommends starting with A (and B) and treating D/E as a fallback if A stalls.

Practical issue with D (INFERENCE): ES uses centred ranks of per-candidate fitness. Agreement is dense and varies smoothly, while capped score is mostly a plateau value. Summing them before ranking lets agreement dominate wherever the score ties. That is the intended effect early, and a hazard late (agreement with a solver that sees hidden state is not the objective). Annealing lambda to 0 on a generation schedule is crude; Nair's Q-filter analogue would be to drop the term when the policy's held-out score beats a threshold. I did not find ES-specific evidence either way.

### Q5. Concrete minimal design

See sections 3 and 4.

---

## 3. Proposed design

Everything below is a proposal. None has been run. Nothing here changes `core.js` or `policy.js` (so the core hash and existing checkpoints stay valid), and nothing touches evaluation.

### 3.1 Data (training-only)

- New script outside `core.js`/`policy.js` (e.g. `trainer/demo-data.js`): run the solver (`test/solver.js`) on the unmodified v2 game over training-range seeds (not validation, not held-out `EVAL_SEEDS`, and not the fresh offsets 100-174 or 175+). At each policy decision step (honouring `actionRepeat`, and the idle-to-start rule in `policy.js`), record `(obs44, solver_action_index)`. Obs must come from the same extraction function the policy uses (`policy.js`), so the recorded features are exactly what training sees.
- Data per episode length: the full 1M points is 1M-scale frames; sample a subset (e.g. segments from points 0-1k, around the first pterodactyls at 500-600, and a spread up to the high-speed regime). Segment mix is a design choice; the probe's feature-sufficiency results should set it.
- Class balance: actions are heavily "none". Weight classes (inverse frequency or fixed weights) and consider up-weighting the frames around takeoff, since those labels are the decisive ones. This is the "weighted cross-entropy" of the brief.
- Optional DART-style noise (INFERENCE): collect some episodes with the solver's action randomly replaced with small probability, labelling with the solver's clean action at the next state if the solver can act from that state (see 3.2). This gives recovery data. Skip in the first pass if the solver cannot relabel.

### 3.2 DAgger relabelling feasibility (check before building)

INFERENCE, from reading `test/solver.js` lines 1-40 only: the solver is a closure with state (`hold`), reads `game.keys.jump`, and needs the full game object (obstacles including off-screen, `requirement()` for each). To relabel states a BC student visits, it would need to be re-driven from that state, or reconstructed as a stateless function of the game state. That needs a solver change (test code), which is out of scope for this note. Without it, do pure BC on solver trajectories (plus optional action-noise episodes if the solver tolerates being run from a perturbed state). If the probe shows the solver crashes or stalls from off-distribution states, DAgger is unavailable and ES is the only remedy for covariate shift.

### 3.3 BC training (plain JS, no dependencies)

- Network exactly as `policy.js`: 44-32-32-3 tanh MLP with the same parameter layout (use `initParams`, `numParams` and the `Policy` forward pass; write the backward pass by hand).
- Loss: weighted softmax cross-entropy over the 3 logits against the solver action. Note the policy takes argmax of the output (verify in `policy.js` before writing the loss; the action mapping `ACTIONS` is three discrete actions).
- Optimiser: Adam, mini-batch, small L2 or decoupled decay to keep weights moderate, early stopping on a held-out episode split (split by seed, not by frame).
- Output a `best.json`-compatible file (`format`, `layers`, `params`, `actionRepeat`, `gameVersion`) so it can be evaluated with `trainer/cli.js eval`.
- The in-repo precedent for the size of the work: trainer already has Adam in `es.js`; reuse its update code if it is exported.

### 3.4 Using theta_BC as the ES start

Two routes (INFERENCE, based on the CLI header and `cli.js` lines 250-292, not tested):
1. **New flag** `--init-from <best.json>`: at fresh-start time (around `trainer/cli.js:292`, where `initParams` is called), load `params` from the file instead of random init, after checking `layers` and `actionRepeat` match. Smallest code change; no checkpoint forging. Adam state starts fresh, which is what a fresh start does anyway.
2. **Resume machinery**: write a checkpoint from theta_BC and `--resume`. Riskier: the checkpoint carries a `coreHash`, optimizer state, generation counter and config; forging it is fragile. Not recommended.

Run settings for the first pass (from the brief, with the sweep added in 4): `--hidden 32,32`, `--rank-ties mid` (`--rank-ties mid`), `--decay decoupled --weight-decay 0.01`, explicit `--dir runs/<new>` (not `runs/main`), no `--auto-export` (it overwrites the shipped `model.js` and `model.css` on a new best).

---

## 4. Experiment plan and decision rule

All runs below are proposals; none was executed.

**Stage 0: data and BC quality (cheap; overlaps the probe).**
- Report per-class held-out accuracy and, more importantly, closed-loop score of theta_BC alone: the median over the 25 held-out seeds at the usual cap, plus the per-seed list. Also the death-point histogram (does it die at the first wall, the pterodactyl wall, late?).
- Stop rule: if theta_BC is not already past the 54-point plateau on most seeds, then BC from these features cannot replace the escape, and the idea reduces to a possible different start for ES. Report and stop.

**Stage 1: ES from theta_BC, no extra loss (variant A/B).**
- Sigma sweep {0.01, 0.02, 0.05, 0.1} (the brief's 0.05-0.1 are the top two), at least 3 trainer seeds each, same other settings. Add a control: random init with identical settings and the same seeds, to compare against the stated baseline escape (median ~1,200 generations).
- Measure per run (fields are in the existing `metrics.*` and `evals.jsonl`):
  - first generation where the held-out median passes ~100 points ("escape skipped" if already true at generation 0);
  - first generation where it passes ~1,000 points (the pterodactyl wall; "second wall skipped" if true at generation 0 or within a few generations);
  - generations to the cap median;
  - the dip: minimum validation score over the first 50 generations relative to generation 0 (the Lorenc & Neruda effect);
  - fraction of antithetic pairs tied (HANDOFF diagnostics), especially once theta_BC's score is near cap.
- Fresh held-out: for the best run per setting, evaluate on a range never used for any decision, starting at offset 175 (offsets 100-174 gave the shipped model 72/75; do not reuse them for selection).

**Statistics for the fresh set (computed here, standard hypergeometric):** one-sided Fisher exact p for "0 failures in n fresh seeds" vs the shipped 3 failures in 75: n=75 gives p about 0.122; n=150 gives p about 0.036; n=300 gives p about 0.0077; n=600 gives p about 0.0013. So 75 fresh seeds cannot show an improvement over the shipped model even with zero failures; use at least 150, preferably 300, for any claim of "better than shipped". To claim only "not worse", compare failure counts with the rule-of-three upper bound `3/n` (Hanley & Lippman-Hand 1983, cited in HANDOFF 4.5; I did not re-read it).

**Decision rule (proposal).**
- *Adopt BC init as the default recipe* if, across at least 3 seeds at the best sigma, (i) all runs reach the 1,000-point wall in under ~1/4 of the baseline median generations (about 300), (ii) all runs reach the cap median in fewer generations than the baseline median, and (iii) the fresh held-out failures at n >= 150 are no worse than the shipped model's rate (Fisher p > 0.05 for "worse").
- *Keep it only as a wall-skipping tool* if (i) and (ii) hold but (iii) fails: the fast start is real but the final model is less robust; then run the longer fine-tune with the rare-failure tools.
- *Drop it* if theta_BC is not past the first wall (Stage 0 stop rule), or if the dip in Stage 1 wipes out the cloned behaviour at every sigma (the run looks like random-init at the same generation count).
- If Stage 1 stalls with ties at the cap, go to Stage 2 (hard-seed mining from `research/rare-failure-robustness.md`), not to variant D.
- Variant D/E (fitness with an agreement term) only after a project-owner decision on the "no shaping" rule, and only if Stage 1 plateaus before the cap with a non-trivial dip.

All thresholds here (the 300 generations, the 1/4 factor, the n values) are proposals, not measured quantities.

---

## 5. Where the evaluation line sits (for the project owner)

- Unchanged evaluation holds for every variant in the table in Q4: the held-out score is still the median capped score of the final policy in the unmodified game.
- Variants A to C keep the training objective as capped score. They change only the starting point, the optimiser settings, or the start states.
- Variants D to F change the training fitness. They are the ones the "no shaping" rule targets. Decide explicitly whether the rule is about the objective or only about reporting.
- Separate question: the solver's actions come from privileged information. `AGENTS.md` anti-cheat rules constrain observations at decision time, which the BC student obeys. A second reading is that a policy distilled from privileged information is itself "informed" by forbidden data. The privileged-teacher literature treats this as the intended setup (Chen et al. 2019; Lee et al. 2020), but the repo owner should say whether a BC-derived `model.js` is acceptable to ship. If not, BC can still be used as a fast-start tool whose ES-trained result is the only shipped artifact; the ES phase can then remove behaviours that do not help the real score (INFERENCE, not guaranteed).

---

## 6. Risks

1. **Imitation gap hides in the decisive states.** If the solver's takeoff frame depends on hidden state in the pterodactyl states, BC is soft exactly where it matters; theta_BC may stall at the second wall. Mitigation: Stage 0 reports death points; the probe's feature-sufficiency result gates the whole plan.
2. **Compounding error.** BC without recovery data may crash early on its own state distribution. Mitigation: ES fine-tuning; optional noise-injected data if the solver can relabel.
3. **Ties at the cap.** A competent init makes ranks tie and ES stalls (INFERENCE). Mitigation: mid ranks (already planned), hard-seed mining, and a check on the fraction of tied pairs.
4. **Initial deterioration and sigma sensitivity.** Reported by Lorenc & Neruda; mitigation: sigma sweep, keeping theta_BC as the floor, `best.json` selected by validation.
5. **Seed and overfitting leakage.** The BC data must come from training seeds only; held-out `EVAL_SEEDS`, the validation seeds and fresh offsets 100-174/175+ must not be used for data or selection. A BC model trained on seeds that overlap the evaluation range would overstate results.
6. **Solver not a stateless oracle.** Relabelling arbitrary states may not be possible without changing test code (out of scope here).
7. **Evidence base is thin.** Only one ES+BC paper was found, on a different architecture and task, and several sources were read at abstract level only (section 7). Treat the literature as motivation, not as predictions for this project.
8. **Cost.** Full-length (1M-point) solver episodes are long to simulate and to evaluate; use subsets for data and a lower cap for screening runs, with the full cap only for finalists. (Costs not measured here.)
9. **Process:** the trainer must not be run with `--auto-export` on experiments, and the CLI defaults `--dir runs/main` and `--hidden 16` are stale (per `AGENTS.md`); pass both explicitly.

---

## 7. References

All accessed 2026-10-10. Status: **V** = full text downloaded and grepped; **A** = arXiv abstract or page read through a summarising tool only; **S** = a search-result summary only, source not opened; **U** = not fetched.

| ID | Reference | URL | Status |
|---|---|---|---|
| R1 | Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger), AISTATS 2011 | https://arxiv.org/abs/1011.0686 | V |
| R2 | Ross & Bagnell, "Efficient Reductions for Imitation Learning", AISTATS 2010 (cited by R1 for the `T^2 eps` bound) | no URL fetched | U |
| R3 | Chen, Zhou, Koltun & Krahenbuhl, "Learning by Cheating", 2019 | https://arxiv.org/abs/1912.12294 | V |
| R4 | Lee, Hwangbo, Wellhausen, Koltun & Hutter, "Learning Quadrupedal Locomotion over Challenging Terrain", 2020 (venue, Science Robotics, not re-verified) | https://arxiv.org/abs/2010.11251 | V |
| R5 | Weihs et al., "Bridging the Imitation Gap by Adaptive Insubordination" (ADVISOR), NeurIPS 2021 | https://arxiv.org/abs/2007.12173 | V (first author list not fully read; search summary names Weihs and Jain) |
| R6 | Pinto, Andrychowicz, Welinder, Zaremba & Abbeel, "Asymmetric Actor Critic for Image-Based Robot Learning", 2017 | https://arxiv.org/abs/1710.06542 | A |
| R7 | Lehman, Chen, Clune & Stanley, "Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients", GECCO 2018 | https://arxiv.org/abs/1712.06563 | A |
| R8 | Hester et al., "Deep Q-learning from Demonstrations" (DQfD) (venue not re-verified) | https://arxiv.org/abs/1704.03732 | V |
| R9 | Nair, McGrew, Andrychowicz, Zaremba & Abbeel, "Overcoming Exploration in RL with Demonstrations", 2018 | https://arxiv.org/abs/1709.10089 | V |
| R10 | Kang, Jie & Feng, "Policy Optimization with Demonstrations" (POfD), ICML 2018 | https://proceedings.mlr.press/v80/kang18a.html | S |
| R11 | Gao, Xu, Lin, Yu, Levine & Darrell, "Reinforcement Learning from Imperfect Demonstrations" (NAC), 2018 | https://arxiv.org/abs/1802.05313 | A |
| R12 | Ecoffet, Huizinga, Lehman, Stanley & Clune, "Go-Explore", 2019 | https://arxiv.org/abs/1901.10995 | V |
| R13 | Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration", 2018 | https://arxiv.org/abs/1812.03381 | A |
| R14 | Lorenc & Neruda, "Utilizing Evolution Strategies to Train Transformers in Reinforcement Learning", 2025 | https://arxiv.org/abs/2501.13883 | V |
| R15 | Mustafaoglu, Pingali & Miikkulainen, "Evolutionary Policy Optimization", 2025 | https://arxiv.org/abs/2504.12568 | A |
| R16 | DeepMind, "AlphaStar: Mastering the real-time strategy game StarCraft II" (blog) | https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/ | A |
| R17 | Silver et al., "Mastering the game of Go with deep neural networks and tree search", Nature 2016 | https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/ | S |
| R18 | Laskey, Lee, Fox, Dragan & Goldberg, "DART: Noise Injection for Robust Imitation Learning", CoRL 2017 | https://arxiv.org/abs/1703.09327 | A |
| R19 | Salimans, Ho, Chen, Sidor & Sutskever, "Evolution Strategies as a Scalable Alternative to RL", 2017 | https://arxiv.org/abs/1703.03864 | A |
| R20 | Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep RL and Demonstrations" (DAPG), RSS 2018. Abstract page did not state the BC-init mechanism; not used for any claim. | https://arxiv.org/abs/1709.10087 | A |
| R21 | MLGO (compiler inlining, BC-warm-started RL), mentioned only in a search summary, not used | https://arxiv.org/abs/2101.04808 | S |

Counts: 21 entries. **V (verified from full text): 8** (R1, R3, R4, R5, R8, R9, R12, R14). **A (abstract/page only): 9** (R6, R7, R11, R13, R15, R16, R18, R19, R20). **S (search summary only): 3** (R10, R17, R21). **U (not fetched): 1** (R2). Entries with no usable claim in this note: R20, R21 (listed for completeness).

In-repo sources: `AGENTS.md`; `handoff/archive/2026-10-10T0742Z-HANDOFF.md` 4.2 and 4.5; `test/solver.js` header (lines 1-40); `policy.js` header and constants; `trainer/cli.js` header and lines 84-300; `research/es-practice.md`, `research/rare-failure-robustness.md` (heads only).
