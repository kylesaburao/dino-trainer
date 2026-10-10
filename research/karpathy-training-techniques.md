# Karpathy's neural-network training techniques, read against ES on a small tanh MLP

Compiled 2026-10-10. Branch wt/mac. No project code was changed and no trainer was run.

Conventions used throughout:
- **STATED** marks what a fetched source says (paraphrase, or a short quote where the text was fetched).
- **INFERENCE** marks my own reasoning about this project. It is not from the source.
- **UNVERIFIED** marks anything I could not confirm from the source text in this session (see section 5).
- Project facts (plateau at ~54 points, ‖θ‖ 8 → 38, ~50% of hidden units with |pre|>2, σ≈0.1, Adam lr 0.03) come from the task brief and `HANDOFF.md` 4.2-4.5. I have not re-measured them.
- The literature already listed in HANDOFF 4.5 (restart distributions, curricula, novelty search, ARS, CMA-ES, EvoJAX and so on) is not repeated here. Salimans 2017, Such 2017, Safe Mutations, Decoupled Weight Decay and µP appear in HANDOFF 4.5 only as names. I re-read them for training-practice details.

Main caveat: **Karpathy has written very little about ES.** His public work is overwhelmingly about backprop-trained networks. Almost every recipe below is gradient-based advice. The ES-specific evidence that survives verification comes from Salimans et al. 2017. Each mapping states whether the advice plausibly transfers.

---

## 1. Summary: the most decision-relevant takeaways

1. **Karpathy's own ES content is essentially nil.** The Pong post (2016) predates the OpenAI ES paper (2017) and never names ES. It only recommends cross-entropy-method style "guess and check" as a baseline to try before policy gradients (STATED). A tweet about ES that I half remember could not be found (UNVERIFIED). Treat his material as a source of diagnostics and habits, not of ES algorithms.

2. **Weight decay in ES has a direct, source-stated rationale.** Salimans et al. apply weight decay so that "parameters" do not "grow very large compared to the perturbations" (paraphrase of the paper's wording, fetched). That is exactly the project's ‖θ‖ 8 → 38 problem. With a fixed σ the relative perturbation σ/|w| falls as ‖θ‖ grows, so the search becomes effectively less exploratory late in training.
   - INFERENCE: the observed slowing of the late climb under decay 0.1 may be the same mechanism. A cheap check is to track the relative perturbation size (see the table).

3. **Keep the perturbation constant in function space, not in weight space.** Salimans et al. give a width-invariance argument. If you duplicate every feature, ES behaves identically provided σ and lr are both divided by 2 (fetched; section 3.2 of the paper).
   - INFERENCE: for a tanh MLP the pre-activation perturbation grows like σ·√fan_in·rms(input). A σ that is right at width 32 is therefore too large at width 128 and too small at width 8. Parametrise σ per layer as σ0/√fan_in (or perturb relative to the init scale) so the sweep over widths does not need a σ re-tune. This is the only concrete σ-transfer idea I found with a source-based argument. It is not a proven rule.

4. **Parameter-space noise can produce policies that ignore their input.** Salimans et al. report that on Atari, randomly perturbed parameters often encoded "always take one specific action". Virtual batch normalisation fixed this by making the policy more sensitive to small input changes at init (STATED). MuJoCo actions were discretised into 10 bins to improve exploration (STATED).
   - INFERENCE: this resembles the 54-point plateau, where perturbations change some actions but rarely the crash frame. A fixed-statistics input normalisation (a VBN-like step: standardise the 44 features with statistics from a fixed batch of observations) is the nearest cheap analogue. The init and plateau are the places to test it.

5. **Initialise the output layer so the starting policy is not degenerate.** Recipe: set the final bias to match the data statistics, and check the initial loss against the theoretical value. The makemore notebook scales the last layer down ("make less confident") (STATED, via the notebook code).
   - INFERENCE: for an argmax policy a small last layer means the biases decide the action, and the HANDOFF init policy is 52% none / 47% duck, with ducking blocking jumps. Setting the output biases or shrinking the last layer deliberately (for example, jump bias slightly positive) is a cheap init experiment. Whether it helps is not known.

6. **Monitor the update-to-weight ratio; the CS231n target is about 1e-3.** The statement is for SGD-trained nets (STATED). Adam-ES does not behave like that: with lr 0.03 the per-coordinate step is about 0.03 whatever the signal (HANDOFF measured mean |m|/√v of 0.22-0.31), so the ratio is ~0.1-0.2 for weights of magnitude ~0.2. A 1e-3 target does not carry over as a target.
   - INFERENCE: it is still a useful per-layer diagnostic (log10 of ‖Δθ_layer‖/‖θ_layer‖ per generation), mainly to compare layers and widths, and to see how the ratio shrinks as ‖θ‖ grows.

7. **Saturation diagnostics are standard, and the project's number is high.** The makemore part 3 notebook prints, per tanh layer, the percentage of activations with |tanh|>0.97 (STATED, notebook code). Its example output shows saturation of a few percent (for example 2.78%) after the fix to init scale. The project's ~50% of units with |pre|>2 (tanh(2)≈0.964) is on a comparable threshold and far higher.
   - Backprop motivation (vanishing gradients through a saturated tanh) does not apply to ES. The ES-relevant effect is INFERENCE: a saturated unit is insensitive to small weight perturbations, so many antithetic pairs give identical actions (ties).

8. **A regularisation and decay recipe exists that is cheap to copy.** nanoGPT defaults: AdamW, lr 6e-4, weight decay 0.1 applied only to tensors with ≥2 dimensions (not biases), betas (0.9, 0.95), linear warmup then cosine decay to lr/10, gradient clipping at 1.0 (STATED, code). Note the project's decay 0.1 coincides numerically with nanoGPT's. The project applies it decoupled (θ *= 1 − lr·wd), and nanoGPT's AdamW does the same, so the value is only loosely comparable.
   - INFERENCE: the options to test are (a) exclude biases from decay, (b) lr/σ cosine decay for the late, rare-crash phase, (c) a short warmup.

9. **"Overfit a small batch first" has an ES analogue.** Recipe: before anything else, overfit a few examples to near-zero loss (STATED).
   - INFERENCE: run ES on 1-3 fixed training seeds and confirm the score reaches the cap. If a given width/depth/σ/lr cannot do that, no sweep result on the full seed set is interpretable. This makes the sweep cheaper, because each architecture can be screened in minutes.

10. **Search hyperparameters randomly and on a log scale, not on a grid.** Both the Recipe and CS231n say this (STATED). It applies directly to the planned sweep over (width, depth, σ, lr, decay).
    - INFERENCE: sample σ0 and lr log-uniformly per architecture, with a short fixed budget, and use successive halving on the survivors (HANDOFF already lists halving as literature).

11. **Do not rely on adaptive σ.** Salimans et al. say they saw no benefit from adapting σ and treat it as fixed (STATED). That is on Atari and MuJoCo, where episodes are short compared with this project's 1M-point episodes. Applying it to a hazard-dominated late phase is open (see section 4).

12. **Do not expect gradient-based advice to port blindly.** Init scales for gradient flow (He, Xavier, the tanh gain of 5/3 in makemore, the nanoGPT 0.02 / √(2·n_layer) residual scaling) are chosen so backward signals neither vanish nor explode. ES has no backward pass. The forward-statistics part of the argument (keep pre-activation variance about 1 per layer) still applies, because it governs saturation and the size of function-space perturbations.

---

## 2. Per-source notes

### 2.1 "A Recipe for Training Neural Networks", Karpathy, 25 Apr 2019
https://karpathy.github.io/2019/04/25/recipe/

STATED (fetched):
- Phases: become one with the data; build an end-to-end training and evaluation skeleton with simple baselines; overfit; regularise; tune; squeeze out the last gains.
- Skeleton: fixed random seed; turn off augmentation; evaluate on the full test set; verify the loss at init (expect "-log(1/n_classes)" for softmax); initialise the final bias to the data statistics (for example, a 1:10 class ratio gives p=0.1 at start; regression mean 50 gives bias 50); an input-independent baseline (zero the inputs); **overfit a single batch** of a few examples; visualise exactly what goes into the network; use gradients to check dependencies.
- Overfit phase: "Don't be a hero" (take the simplest known model). Adam with lr 3e-4 is "safe" as a baseline. Add complexity one piece at a time. **Disable lr decay at first** (constant lr), because epoch-based schedules can drive the rate to zero too early.
- Regularise: more data first; augmentation; pretraining; smaller input dimensionality or model size; smaller batch size; dropout (sparingly); increase weight decay; early stopping.
- Tune: random search over grid search.
- Gains: ensembles (about 2% on anything, per the post); train longer, because nets often improve after validation loss plateaus.

Relevance: skeleton checks and the order of operations (get a trustworthy eval first; make a tiny problem solvable before scaling). Most regularisation advice is about overfitting to a dataset. This project's training seeds, validation seeds and held-out seeds already follow the same split, so the analogue is fine.

### 2.2 CS231n notes: Neural Networks Part 3 (learning and evaluation)
https://cs231n.github.io/neural-networks-3/ (Karpathy was the lead instructor and author of these notes; authorship is not displayed on the fetched page, so treat the attribution as UNVERIFIED from the page itself.)

STATED (fetched):
- Update-to-parameter ratio: "a rough heuristic is that this ratio should be somewhere around 1e-3". Lower suggests lr too low; higher, too high.
- Babysitting: loss curve shape and noise level depend on batch size and lr.
- Gradient checks: centred differences, relative error (above 1e-2 usually means a bug), double precision.
- Activation and gradient histograms: for tanh you want activations spread across the range, not all near zero and not "all neurons being completely saturated".
- First-layer weights should look smooth and diverse; noisy ones suggest an unconverged net, a poor lr or weak regularisation.
- Hyperparameters: search on a log scale (for example `10 ** uniform(-6, 1)`); random search is more efficient than grid search (citing Bergstra and Bengio).
- Regularisation: "Don't let the regularization overwhelm the data".

### 2.3 CS231n notes: Neural Networks Part 2 (setup)
https://cs231n.github.io/neural-networks-2/

STATED (fetched):
- Zero-centre inputs; compute statistics on training data only.
- Initialise with small random numbers, but not identical (symmetry breaking). Scale by √fan_in so neuron output variance is about 1; He init uses variance 2/n for ReLU. Biases are commonly zero.
- Batch normalisation reduces sensitivity to bad init.
- L2 weight decay shrinks every weight linearly each step. Max-norm constraints (typically norm bound around 3-4) keep weights from "exploding" even at a too-high lr.

Relevance: max-norm is a hard cap on per-neuron weight norms and is an alternative to decay for the ‖θ‖ growth issue (INFERENCE; untested here).

### 2.4 "Deep Reinforcement Learning: Pong from Pixels", Karpathy, 31 May 2016
https://karpathy.github.io/2016/05/31/rl/

STATED (fetched):
- 2-layer policy net with 200 hidden units, 80x80 difference-frame input, RMSProp, batches of 10 episodes, trained about 3 nights on a MacBook. Discount 0.99; returns standardised (mean subtracted, divided by std) to control variance.
- Calls policy gradients "brute force": the agent must "accidentally stumble into rewarding situations" at least once. Credit assignment spans about 20 frames in Pong.
- Mentions a "stochastic hill-climbing 'guess and check'" baseline loosely inspired by evolution (cross-entropy method). ES is **not** named (post predates the OpenAI paper).
- Random initialisation of W1, W2, no scheme specified; biases omitted.

Relevance: standardising returns is the same idea as the project's centred-rank shaping. The plateau (find a rewarding state by chance) is the same exploration problem he describes.

### 2.5 Zero to Hero lecture series and "makemore Part 3: Activations & Gradients, BatchNorm"
Series page: https://karpathy.ai/zero-to-hero.html (fetched; lists the lectures and describes Part 3 as covering "activation and gradient statistics" and diagnostic tools plus BatchNorm).
Lecture: https://youtu.be/P6sfmUTpUmc (title only confirmed; the video content could not be fetched).
Notebook: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb (read through raw.githubusercontent.com).

STATED from the notebook (code and comments, partially read: roughly the first 200k of 495k characters):
- Init: a `(5/3)/sqrt(fan_in)` gain for tanh appears (commented out in places), and a "semi-principled kaiming init" log line. The final layer is scaled by 0.1 ("last layer: make less confident").
- Saturation diagnostic: per tanh layer, mean, std, and `(t.abs() > 0.97).float().mean()*100` percent saturated.
- Update-to-data ratio recorded as `((lr*p.grad).std() / p.data.std()).log10()` for each parameter over training.
- Step lr decay: lr 0.1 then 0.01 after a set step.

UNVERIFIED: the spoken explanations, the specific "log10 ratio about -3" reference line in the plot, and the claim that the 5/3 gain keeps tanh outputs at std about 1. These are things I recall from the lecture and cannot cite from the text I read.

### 2.6 nanoGPT (train.py and model.py)
https://github.com/karpathy/nanoGPT (raw files read: train.py, model.py)

STATED (code, read from raw.githubusercontent.com):
- Defaults: learning_rate 6e-4, weight_decay 1e-1, beta1 0.9, beta2 0.95, grad_clip 1.0, warmup_iters 2000, lr_decay_iters 600000 (comment: should be about max_iters per Chinchilla), min_lr 6e-5 (comment: about lr/10).
- Schedule: linear warmup, then cosine decay to min_lr, then constant min_lr.
- Init: Linear and Embedding weights normal(0, 0.02), biases zero; the residual projection weights (`c_proj.weight`) get std 0.02/√(2·n_layer).
- Weight decay is applied to parameters with dim≥2 only; biases and LayerNorm parameters are excluded.

### 2.7 llm.c and nanochat
- llm.c: https://github.com/karpathy/llm.c. A README-level fetch only. It states the project is plain C/CUDA GPT-2/3 pretraining and shows an lr sweep (3e-5, 1e-4, 3e-4, 1e-3). No default hyperparameters were visible. Nothing here changes the picture from nanoGPT.
- nanochat: https://github.com/karpathy/nanochat. README only: uses AdamW and Muon; says hyperparameters such as lr adjustments and weight decay are calculated automatically from a single `--depth` setting. Values were not visible. The idea that one scalar (depth) sets the other hyperparameters is the same goal as the project's planned sweep, but I did not verify the rule it uses. To find it, read `nanochat/optim.py` and the training script (not done).

### 2.8 "Software 2.0", Karpathy, 2017
https://karpathy.medium.com/software-2-0-a64152b37c35. Direct fetch returned HTTP 403. A web search confirmed the URL exists, that it was published around late 2017, and the core thesis (the program lives in the weights; you specify the goal and a search finds the program). I could not read the article text, so I cite nothing specific from it. Low relevance to the problems above.

### 2.9 Salimans, Ho, Chen, Sidor, Sutskever, "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", arXiv:1703.03864 (v1 10 Mar 2017, v2 7 Sep 2017)
https://arxiv.org/abs/1703.03864 (abstract page fetched; PDF text of v2 also read)

STATED:
- Antithetic (mirrored) sampling, always evaluating pairs ε and −ε, to reduce variance.
- Rank transformation of returns (fitness shaping): "removes the influence of outlier individuals" and reduces the tendency to fall into local optima early (paraphrase).
- Weight decay on the policy parameters, to stop them growing "very large compared to the perturbations".
- No benefit seen from adapting σ; fixed hyperparameter.
- Parameter-space exploration: for ES to improve, "it is crucial" that some perturbations beat others. Perturbed Atari policies often "always took one specific action"; virtual batch normalisation fixed this. Discretising continuous actions helped on some MuJoCo tasks.
- MuJoCo policies: two 64-unit hidden layers with tanh, the same architecture as the TRPO comparison. ES reached TRPO's final performance within roughly a 10x sample penalty on the hardest tasks.
- Dimension argument: duplicating features does not make the problem harder as long as σ and lr are both divided by 2. In practice slightly better results with larger networks. The authors suggest that it is the intrinsic dimension that matters.
- Episode length capped at m steps, adjusted during training (to keep CPU use high).

### 2.10 Such et al., "Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative...", arXiv:1712.06567 (v1 18 Dec 2017, v3 20 Apr 2018); Uber AI Labs blog "Welcoming the Era of Deep Neuroevolution", 18 Dec 2017
https://arxiv.org/abs/1712.06567; https://www.uber.com/blog/deep-neuroevolution/

STATED: a simple GA trains networks of over 4 million parameters and can beat ES and DQN on several Atari games; it parallelises well. Combined with novelty search it solves deceptive tasks where reward-maximising methods (including ES) fail. The blog says ES, without added exploration, can get stuck indefinitely in local optima (an Atari example where ES converged to never surfacing for oxygen).

### 2.11 Lehman, Chen, Clune, Stanley, "Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients", arXiv:1712.06563
https://arxiv.org/abs/1712.06563. STATED: random mutation of many weights tends to break existing behaviour; SM-G scales each weight's mutation by the sensitivity of the outputs to that weight (needs gradients of outputs). Relevant to the idea of non-isotropic σ, but it needs a differentiable policy and the project's MLP is differentiable (INFERENCE; untested).

### 2.12 Loshchilov and Hutter, "Decoupled Weight Decay Regularization", arXiv:1711.05101 (ICLR 2019)
https://arxiv.org/abs/1711.05101. STATED: L2 and weight decay are equivalent for SGD only; for Adam they differ; decoupling makes the best decay value much less dependent on the lr. INFERENCE: in the project, coupled L2 inside Adam is rescaled by the adaptive denominator, so the HANDOFF finding (decoupled decay bounded ‖θ‖, coupled did not) is consistent with this paper.

### 2.13 Greg Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (µP), arXiv:2203.03466
https://arxiv.org/abs/2203.03466. STATED: under the Maximal Update Parametrization many optimal hyperparameters stay stable as width grows; tune on a small proxy and transfer. INFERENCE: µP is derived for gradient-based training (SGD/Adam). Whether its width-scaling rules (init scales and lr multipliers per layer) carry over to ES with Adam on the ES gradient estimate is unproven. The √fan_in σ rule in takeaway 3 is a related but independent heuristic.

### 2.14 He, Zhang, Ren, Sun, "Delving Deep into Rectifiers...", arXiv:1502.01852 (Feb 2015)
https://arxiv.org/abs/1502.01852. STATED: an initialisation method designed for rectifiers (variance 2/n). The CS231n notes summarise it. For tanh the corresponding rule is Xavier/Glorot (variance about 1/n), which I did not fetch (cited by name only; UNVERIFIED).

---

## 3. Mapping table

"Transfers to ES?" is my INFERENCE unless a source is cited. Experiments are sized for the CPU being busy; each one is a short run with a handful of seeds, and should be compared against an unchanged-baseline run of the same length.

| # | Technique (source) | Project problem | Transfer to ES on a small tanh MLP | Cheap experiment |
|---|---|---|---|---|
| 1 | Weight decay to keep θ small relative to perturbations (Salimans 2017; nanoGPT wd 0.1 on dim≥2 only) | ‖θ‖ 8 → 38; slowed late climb | Strong. Source explicitly gives this rationale for ES. Biases excluded from decay in nanoGPT (untested here). | Log σ/rms(w) per layer vs generation in an existing run. Then a short run with decay on weights only vs weights plus biases. |
| 2 | σ scaled per layer by 1/√fan_in (derived from Salimans dimension argument; compare µP) | σ and lr do not transfer across width/depth | Plausible (derivation: pre-activation perturbation ≈ σ·√fan_in·rms(input)). Not proven for lr. | Measure the std of the pre-activation change induced by a σ-perturbation at widths 16, 32, 64 with current σ; then rerun with σ0/√fan_in and compare the first 100 generations. |
| 3 | Update-to-weight ratio per layer (CS231n ~1e-3; makemore log10 ratio) | Same, plus Adam step size ~lr regardless of signal | Diagnostic only. The 1e-3 target is for SGD and does not apply to Adam-ES (HANDOFF measured |m|/√v ≈ 0.2-0.3). | Log log10(‖Δθ_layer‖/‖θ_layer‖) per generation per layer, for 32,32 and 32,32,32. |
| 4 | Saturation histogram (makemore: percent with abs(tanh)>0.97) | ~50% of units with abs(pre)>2 | Backprop reason (vanishing gradient) does not apply. Reduced sensitivity to perturbation (ties) does. | Add a per-layer saturation print to an existing eval script (read-only); compare saturation vs tie rate by generation. |
| 5 | Init for sane starting outputs (Recipe: final bias; makemore: scale last layer by 0.1) | Init policy 52% none / 47% duck; ducking blocks jumps | Plausible. With argmax, small last layer makes biases decide the action. | Short runs with output bias initialised to prefer none or jump vs the current init; measure generations to leave the 40.475 constant-policy score. |
| 6 | Input standardisation / VBN (CS231n zero-centre; Salimans VBN for degenerate perturbed policies) | 54-point plateau; ties between antithetic pairs | Plausible. VBN reason is from Salimans' own ES experience. Features here are already roughly scaled; check their means and variance first. | Compute per-feature mean/std over a fixed set of observations. If some features have tiny variance, test a fixed (not running) standardisation; compare plateau escape time over a few seeds. |
| 7 | Overfit a tiny set first (Recipe) | Interpreting the architecture sweep | Direct analogue. | For each candidate width/depth, run ES on 1-3 fixed training seeds for a fixed budget; discard configurations that cannot reach the cap. |
| 8 | Random log-scale search, not a grid (Recipe; CS231n) | Planned sweep over layer sizes and counts | Directly applicable (search method, not model). | Sample (σ0, lr, decay) log-uniformly for each architecture with a short budget; keep top third. |
| 9 | lr warmup then cosine decay to lr/10 (nanoGPT) | Late climb, rare crashes; early Adam steps ±lr | Uncertain. Warmup could damp the early random walk, which HANDOFF says was the plateau-escaping mechanism, so it may hurt. Decay late is more plausible. | Two arms: decay lr (and optionally σ) by cosine in the last 30% of a run vs constant; compare held-out min over seeds. |
| 10 | Adapt σ (Wierstra via Salimans) | Late phase, hazard-dominated | Salimans saw no gain, but on short-horizon tasks. Open. | Fixed σ vs a one-off drop of σ by 2x when validation stalls; compare the same budgets. |
| 11 | Max-norm constraint (CS231n) | ‖θ‖ growth | Alternative to decay: hard cap on per-unit incoming weight norm. Untested for ES. | Clip rows of each weight matrix to a norm bound after each update; compare ‖θ‖ and saturation. |
| 12 | Gradient clipping at 1.0 (nanoGPT) | None clear | Probably not needed: ES gradient estimates are rank-shaped (bounded) and Adam normalises scale. | Skip unless instability is observed. |
| 13 | Population-based exploration / novelty search (Such 2017, Uber blog) | Plateau | Already in HANDOFF 4.5 (novelty search listed). Noted as stated by the Uber blog: ES may stall without added exploration. | None new here. |
| 14 | Safe-mutation-style non-isotropic σ using output sensitivity (Lehman 2017) | Ties and saturation; σ transfer | Needs output gradients wrt weights. A forward-only proxy (scale σ by 1/rms of the incoming activation) is the same family as row 2. Untested. | Defer until row 2 is tried. |
| 15 | Ensembles (Recipe) | Rare late crashes | An ensemble of policies voting on the action is different from a single exported net. The export format (one MLP, CSS evaluation) forbids it as shipped. | Not applicable without changing the product. |

---

## 4. Open questions

- What does `‖Δθ‖/‖θ‖` per layer look like over a full 1M-point run, and does the point where it drops coincide with the late slowdown? (Needs only logging, not a new algorithm.)
- Does the saturation level (50%) matter for ES at all, or is it benign for argmax policies? Is a saturated hidden layer actually correlated with antithetic ties, or is it just the common end state?
- Is Karpathy's tanh gain of 5/3 (or Xavier) the right init scale for a 44-input, 3-output argmax MLP, and does it change the plateau time? The forward-statistics reasoning applies but the gradient-flow reasoning does not.
- Does a σ ∝ 1/√fan_in rule actually flatten the optimum σ across widths 16-128, or do depth effects (accumulating perturbation through layers) dominate? µP-style per-layer rules might need a depth term.
- Salimans found no benefit from adapting σ. Does that hold in a regime where the remaining failures are rare crashes late in a 1M-point episode?
- The 5/3 gain, the "-3" log10 line in the makemore ratio plot, and the spoken claims of the lectures: read the transcript or watch the lecture to confirm before relying on them.
- nanochat's rule for deriving hyperparameters from `--depth`: not read. If it exists as code, it may be the best Karpathy-authored reference for the sweep problem.
- A Karpathy tweet about ES (2017) may exist; I could not find or verify one.

---

## 5. References (accessed 2026-10-10)

Verification status: **VERIFIED** means I fetched the page or file and the title, author, date and URL match what is shown. **UNVERIFIED** means the content (or an item) could not be read or confirmed.

Verified (18):
1. Karpathy, "A Recipe for Training Neural Networks", 25 Apr 2019. https://karpathy.github.io/2019/04/25/recipe/ . VERIFIED.
2. Karpathy, "Deep Reinforcement Learning: Pong from Pixels", 31 May 2016. https://karpathy.github.io/2016/05/31/rl/ . VERIFIED.
3. CS231n, "Neural Networks Part 3: Learning and Evaluation". https://cs231n.github.io/neural-networks-3/ . VERIFIED content; author attribution not shown on the fetched page.
4. CS231n, "Neural Networks Part 2: Setting up the Data and the Loss". https://cs231n.github.io/neural-networks-2/ . VERIFIED content; same attribution caveat.
5. nanoGPT `train.py`. https://github.com/karpathy/nanoGPT/blob/master/train.py (read via raw.githubusercontent.com/karpathy/nanoGPT/master/train.py). VERIFIED.
6. nanoGPT `model.py`. https://github.com/karpathy/nanoGPT/blob/master/model.py (read via raw). VERIFIED.
7. Karpathy, "Neural Networks: Zero to Hero". https://karpathy.ai/zero-to-hero.html . VERIFIED (lecture list and descriptions).
8. makemore part 3 notebook. https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb . VERIFIED for the parts read (about 200k of 495k characters).
9. nanochat README. https://github.com/karpathy/nanochat . VERIFIED (README only).
10. llm.c README. https://github.com/karpathy/llm.c . VERIFIED (README only).
11. makemore README. https://github.com/karpathy/makemore . VERIFIED (minor).
12. Salimans et al., "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", arXiv:1703.03864, 2017. https://arxiv.org/abs/1703.03864 . VERIFIED (abstract page and PDF text).
13. Such et al., "Deep Neuroevolution...", arXiv:1712.06567, 2017/2018. https://arxiv.org/abs/1712.06567 . VERIFIED (abstract).
14. Stanley and Clune (Uber AI Labs), "Welcoming the Era of Deep Neuroevolution", 18 Dec 2017. https://www.uber.com/blog/deep-neuroevolution/ . VERIFIED.
15. Lehman et al., "Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients", arXiv:1712.06563. https://arxiv.org/abs/1712.06563 . VERIFIED (abstract).
16. Loshchilov and Hutter, "Decoupled Weight Decay Regularization", arXiv:1711.05101, ICLR 2019. https://arxiv.org/abs/1711.05101 . VERIFIED (abstract).
17. Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", arXiv:2203.03466. https://arxiv.org/abs/2203.03466 . VERIFIED (abstract).
18. He et al., "Delving Deep into Rectifiers...", arXiv:1502.01852, 2015. https://arxiv.org/abs/1502.01852 . VERIFIED (abstract).

Unverified (4):
19. Karpathy, "Software 2.0", Medium, 2017. https://karpathy.medium.com/software-2-0-a64152b37c35 . UNVERIFIED: fetch returned 403; URL and thesis only confirmed from web-search snippets.
20. Karpathy, "Yes you should understand backprop" (Medium, 2016). https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b . UNVERIFIED: fetch returned 403; not cited for any claim, listed only because it discusses saturated and dead units and is worth reading.
21. makemore part 3 lecture video, https://youtu.be/P6sfmUTpUmc . UNVERIFIED beyond its title (page text could not be fetched); the notebook (item 8) is the evidence used.
22. A Karpathy tweet or comment about ES (2017). UNVERIFIED: not found by search; no claim in this document depends on it.

Count: 18 verified, 4 unverified.
