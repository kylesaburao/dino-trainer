# Width/depth hyperparameter transfer for the ES trainer, and how to sweep layer sizes

Compiled 2026-10-10, branch wt/mac. No project code was changed and no trainer was run.

Labels:
- **STATED (source)**: read in the source text. The papers were downloaded and text-extracted locally (macOS PDFKit) and grepped. Quotes are short and verbatim only inside quotation marks.
- **STATED (via summary)**: fetched through a summarising tool, not as raw text.
- **INFERENCE**: my reasoning about this project. Not from any source.
- **TOY**: a number I computed myself with a throwaway Node script (scratchpad, not committed). It uses Gaussian inputs, **not the project's 44 features**, and a single RNG stream, so treat it as an order-of-magnitude illustration.
- **UNVERIFIED**: not confirmed from source text in this session.

Project facts (tanh MLP 44 → hidden → 3, argmax; OpenAI-ES, antithetic, global σ 0.1, centred ranks, Adam lr 0.03, pop 128; init N(0, 1/fan_in); weight RMS ~0.7 per layer late; ~50% of hidden units saturated; plateau escape on 32,32 about 1/3 of seeds by gen 900) come from the task brief and `HANDOFF.md` section 4.3 ("Weights" bullet: final RMS 0.69-0.77 in every layer in the baseline, 0.45-0.50 in the curriculum model, against an init of 0.15-0.18; |pre|>2 for 48% and 58% of hidden units). I did not re-measure them. The `--sigma-scale fanin` mode (`trainer/cli.js` line 56, `trainer/es.js` `sigmaVector`) scales layer l's weight σ by sqrt(44/fan_in_l); I did not read it beyond that. I found no per-layer learning-rate option in the CLI grep; **I did not verify that there is none**.

This note does not repeat `research/karpathy-training-techniques.md` (it already gives the Salimans duplicate-feature argument and the "σ per layer as σ0/√fan_in" idea, item 3 of its summary), `research/es-practice.md` (tie handling, σ-schedules, restart rules; its item 6 says no library default scales σ with network width; item 12 covers the es-starter `normc(0.01)` output init) or `research/observation-design.md`. It cites them where they bear on the question.

---

## 1. Summary: the decision-relevant takeaways

1. **µP's width rule for Adam is a per-layer learning-rate rule, not only an init rule.**
   - STATED (Tensor Programs V, Table 3): hidden weights have init variance 1/fan_in and Adam LR ∝ 1/fan_in. Output weights have Adam LR ∝ 1/fan_in (and init variance 1/fan_in², i.e. a small output layer). Input weights and all biases have Adam LR Θ(1). The same paper's Table 8 is an equivalent form.
   - STATED: with µP the optimal LR is "stable" across width; with standard parametrisation it is not (Figures 1 and 3).
   - The project's single absolute lr 0.03 is the standard-parametrisation behaviour that µP says does not transfer.

2. **What µP does not give: depth.**
   - STATED (TP5, conclusion): "initialization does not transfer well across depth, and depth transfer generally still does not work for post-layernorm Transformers". The paper also writes that transfer across depth is "fragile" (appendix G.2.2).
   - STATED (Tensor Programs VI, Table 1; Bordelon et al. abstract): the depth rules (branch multiplier a/√depth, Adam LR η/√depth) are for **residual** networks. Bordelon: "in this scheme, hyperparameters do not transfer across depths".
   - STATED (TP6, section 2): for plain MLPs with no skip connections it points to Jelassi et al., depth^(-3/2) learning-rate scaling, which "guarantees stability after the initial gradient step", and says it is unclear beyond that step (the Jelassi paper itself is UNVERIFIED, seen only through TP6).
   - Conclusion: **no published rule transfers lr or σ across depth for a plain tanh MLP.** Treat depth as an axis to sweep, not to extrapolate.

3. **µP was derived for backprop gradients. Whether its 1/fan_in Adam rule or a 1/√fan_in rule is right for ES is an open question that one cheap measurement settles.**
   - INFERENCE: Adam moves every coordinate by about lr, whatever fan-in. The change in a unit's pre-activation is Σ_j Δw_j·a_j. If the signs of Δw_j align with the inputs (gradient-like, which is what µP's 1/fan_in assumes) it grows like lr·fan_in·rms(a). If the signs are incoherent (noise-like) it grows like lr·√fan_in·rms(a).
   - With pop 128 in a ~2.5k-6k dimensional space the ES gradient estimate is mostly noise, so the incoherent regime may dominate, especially on a plateau.
   - Discriminator: measure the RMS pre-activation change per generation, per layer, at width 32 vs 64 under one global lr. ×2 supports 1/fan_in. ×√2 supports 1/√fan_in. Section 3 turns this into a stage-0 check.

4. **The one source-backed σ rule is "σ_l ∝ 1/√fan_in", and the existing `--sigma-scale fanin` already implements it.**
   - STATED (Salimans et al. 2017, section 3.2): duplicating every feature leaves ES unchanged "as long as we divide the standard deviation of the noise by two, as well as the learning rate". Note that this is for a doubling of the **input** fan-in, with σ and lr scaled together.
   - INFERENCE: a weight perturbation of std σ_l shifts a pre-activation by σ_l·√fan_in·rms(a). Making that width-invariant gives σ_l ∝ 1/√fan_in. The first layer's fan_in is 44 in every shape, so it needs no change; `fanin` mode leaves it at 0.1.
   - TOY: under `fanin` the RMS change in the logits for a random perturbation is nearly flat across the seven shapes (0.57-0.65 in the first run, 0.56-0.62 in a second). Under a global 0.1 it is 0.46-0.71.

5. **Lehman et al. 2018 (SM-G) gives the qualitative argument for per-layer σ but no scaling law.**
   - STATED: "perturbing a weight in the layer immediately following the inputs has rippling consequences as activation propagates through subsequent layers, unlike perturbation in layers nearer to the output."
   - SM-R rescales the whole perturbation. SM-G uses output gradients per weight. Neither is a fan-in rule, and SM-G needs gradients that this project does not have.
   - I found no per-layer or layer-wise σ in the evosax or EvoJAX papers (grep of their text), and es-practice item 6 records the same for their code.

6. **The weight-RMS ~0.7 in every layer is what Adam's per-coordinate normalisation predicts, so it is not evidence about init.**
   - INFERENCE: with pure-noise gradients Adam's accumulated displacement is about lr·√T per coordinate. 0.03·√900 ≈ 0.9, against the observed 0.69-0.77 (and 0.45-0.50 with the curriculum). The init RMS (0.15-0.18) is forgotten after roughly (0.17/0.03)² ≈ 32 generations of random walk.
   - This is consistent with, but does not prove, a noise-dominated update. It also says that tuning the **init** std will not change the late state much. The lr, the σ and a weight-decay-like restoring force do.

7. **Late saturation follows from the same arithmetic and gets worse with fan-in.**
   - INFERENCE: pre-activation std ≈ w_rms·√fan_in·rms(a). At w_rms 0.7 and fan_in 32-64 that is 4-6·rms(a).
   - TOY (Gaussian inputs, i.i.d. weights at RMS 0.7): 59-69% of hidden pre-activations have |pre|>2 in all seven shapes, matching the 48-58% in HANDOFF in order of magnitude.
   - To hold the pre-activation scale fixed as fan-in grows, the drifted weight RMS has to scale as 1/√fan_in, i.e. lr_l ∝ 1/√fan_in in the incoherent regime (takeaway 3).

8. **A forward-signal story for why 3 hidden layers may plateau where 2 do (INFERENCE; no direct evidence for ES).**
   - STATED (Pennington et al. eqns 4-5; Schoenholz et al.): for tanh the order/chaos boundary is χ = σ_w²∫Dh φ'(√q*·h)² = 1. By eqn 5, N(0,1/fan_in) weights with zero biases (σ_w=1, σ_b=0) give χ = 1 at q* = 0, a critical point with decaying activation variance.
   - TOY: init logit RMS falls with depth (shape [32]: 0.49, [32,32,32]: 0.38; [64]: 0.68, [64,64,64]: 0.41). The fraction of argmax decisions flipped by one `fanin` perturbation rises (0.36-0.41 for one hidden layer, 0.47-0.53 for three).
   - So the same relative perturbation changes the action more often in the deeper net, and the useful signal (obstacle-dependent logit difference) is smaller relative to it.
   - STATED (Pennington): Gaussian i.i.d. weights cannot reach dynamical isometry at large depth, orthogonal weights can for sigmoidal nets. All of that is about backprop speed. Whether it affects ES rank signals is untested here.

9. **Successive halving on score is the wrong promotion rule for this problem.**
   - STATED (Hyperband, ASHA): promote the top 1/η of configurations by the intermediate score, η = 3 by default.
   - INFERENCE: during a plateau every run scores about 54, so rung 0 would be a coin flip, and it would kill slow starters such as 32,32,32. Use plateau escape as a binary or time-to-event outcome with confidence intervals, and use scores only after escape.
   - Budgets should be frames or wall-clock, not generations: plateau generations are cheap and late generations are expensive, so "gen 900" costs very different amounts in different shapes.

10. **With a ~1/3 escape rate, 5-10 seeds per shape cannot rank shapes.** My power calculation (normal approximation, two-sided α 0.05, power 0.8):
    - 1/3 vs 2/3 needs about 35 seeds per arm;
    - 1/3 vs 0.9 needs about 11;
    - 1/3 vs 1/2 needs about 137.
    - Zero escapes in 8 seeds is evidence against a 1/3 rate (P = 0.04) but is compatible with up to a 32% true rate (Wilson upper bound).
    - STATED (Henderson 2018): two groups of 5 seeds with the same configuration differed significantly (t-test p = 0.0016). STATED (Agarwal 2021): folk wisdom is 20-30 runs, and bootstrap CIs on medians in Atari 100k need "closer to 50-100 runs".
    - So the sweep must be staged so that most seeds are spent on the few shapes that survive.

11. **For the continuous knobs use random search with values shared across shapes, not a grid.**
    - STATED (Bergstra and Bengio 2012): random search finds models as good or better than grid search in a small fraction of the compute, because "only a few of the hyper-parameters really matter" and which ones differs across data sets.
    - INFERENCE: share one list of sampled (s, ρ) values across all shapes. Then "did the best setting move with width?" is directly readable, which is the transfer question.

12. **PBT (Jaderberg 2017) fits as a production trainer after the sweep, not as the sweep.** (INFERENCE)
    - STATED: exploit copies weights and hyperparameters of a top-20% member into a bottom-20% member, and explore perturbs hyperparameters by ×1.2 or ×0.8.
    - Copying an escaped member's weights into a stuck run erases the per-shape escape statistic this sweep wants to measure.

---

## 2. Findings per question

### 2.1 Question 1: µP, Tensor Programs V, the `mup` library, depth work

**What the paper's table says (STATED, Tensor Programs V, arXiv 2203.03466, Tables 3, 8, 9).** Rows are parameter classes, with "fan_in" the layer's input dimension; the fan_out of a bias is its dimension and its fan_in is 1. Parentheses in the original are the standard-parametrisation entry.

Table 3 ("µP and SP for General Neural Networks"):

| | Input weights and all biases | Output weights | Hidden weights |
|---|---|---|---|
| Init variance | 1/fan_in | 1/fan_in² (SP: 1/fan_in) | 1/fan_in |
| Adam LR | 1 | 1/fan_in (SP: 1) | 1/fan_in (SP: 1) |
| SGD LR | fan_out | 1/fan_in | 1 |

- Table 8 is stated to be an equivalent form with a multiplier: output init variance 1 with a 1/fan_in output multiplier, and Adam LR 1 for output.
- The text extraction of the table rows is imperfect (columns run together). I checked the Adam hidden entry against the `mup` README, which says the Adam learning rate of hidden weights is `globalLR / p.infshape.width_mult()`, where width_mult is fan_in / base_fan_in (STATED (via summary)). They agree.
- The paper's own reading: "a multiplier of order 1/fan_in should accompany any weight that maps an infinite dimension to a finite one" (Table 8 discussion).
- STATED (footnote 9): Glorot-style init (variance 1/(fan_in+fan_out)) "would scale asymptotically the same as µP" for the init, but with layer/batch norm it blows up logits where µP does not.
- This project has finite input (44 features), infinite hidden and finite output (3). In µP terms the 44→h layer is "input" (lr Θ(1)), the h→h layers are "hidden" (lr ∝ 1/width), the h→3 layer is "output" (lr ∝ 1/width, init small).

**What transfers (STATED).**
- Table 2 lists the µTransferable classes: optimiser (learning rate, momentum, Adam betas, schedule), init variance, parameter multipliers, "all of the below can also be specialized to per-layer hyperparameters".
- Table 1 and section 6: learning rate and similar transfer; regularisation hyperparameters (dropout, weight decay) "don't work well"; width, depth and batch size are the training-scale axes across which the others transfer.
- Section 6.1: "HPs generally transfer across scale dimensions if some minimum width (e.g. 256), depth (e.g., 4)..." with the caveats quoted in takeaway 2. The minimum widths in the paper's experiments are far above 16-64.
- The same section says that in practice (their section 7.3 results) fixing the init std while tuning the other hyperparameters works well when transferring across depth.

**The `mup` library (STATED (via summary), microsoft/mup README).**
- `set_base_shapes` requires that base and target models "have the same depth".
- `MuReadout` replaces the output layer. `MuAdam` divides the hidden-weight lr by the width multiplier.
- The coord check: track activation size across widths (it should stay roughly stable), use a large learning rate, and a blow-up or shrink-to-zero signals a bug. `readout_zero_init=True` is recommended.
- The README does not give an explicit lr rule for output or bias parameters in the text I received.

**Depth (STATED).**
- Tensor Programs VI (arXiv 2310.02244): for residual networks whose blocks have a single layer, branch multiplier a/√depth, and Adam learning rate η/√depth. "If each block is deeper... we find fundamental limitations". Section 9 says block depth ≥ 2 "is defective".
- Bordelon et al. (arXiv 2309.16620): 1/√L residual-branch scaling on top of µP gives transfer across width and depth for ResNets and ViTs on CIFAR-10 and ImageNet. The same paper observes that, outside this scheme, "hyperparameters do not transfer across depths". It also notes that transfer across width and depth was achieved on non-residual networks with the Automatic Gradient Descent algorithm of Bernstein et al., which is a different optimiser (UNVERIFIED, seen only in the related-work paragraph).
- **Which rules apply to a plain tanh MLP with no residuals? None of the depth rules.** The 1/√L factor multiplies a residual branch, and this network has none. The width rules (Table 3) are derived for MLPs and apply per layer. For depth, the closest is Jelassi et al., depth^(-3/2), first step only (UNVERIFIED).

**What does not carry over to ES (INFERENCE).**
- Derivation: µP matches the size of Θ(1) feature updates to the size of the backprop gradient's alignment with activations. ES has no such alignment structure: its "gradient" is Σ(rank-weight × noise)/(nσ), and each coordinate is mostly noise at this population size.
- So the 1/fan_in Adam rule is a candidate, not a result. See the coherence argument in takeaway 3.
- Tanh saturation is also outside µP's assumptions: µP is a statement about infinite-width scaling, not about whether the finite-width network has gone into the saturated regime (takeaway 7).

### 2.2 Question 2: the ES analogue and per-layer σ

**Setup (STATED, Salimans 2017 eq. and section 3.2).** The ES gradient estimate is (1/nσ)Σ F_i ε_i with θ+σε. Salimans remarks that the ES "gradient" resembles finite differences and might scale poorly with dimension, then argues the relevant quantity is intrinsic dimension: duplicating all features leaves ES unchanged "as long as we divide the standard deviation of the noise by two, as well as the learning rate". "We did not see benefit from adapting σ during training."

**Derivation of the per-layer scaling (INFERENCE; the algebra is mine).** Let layer l have fan-in f_l, activations a (RMS r_a, i.e. ‖a‖ = √f_l·r_a), weight noise std σ_l and bias noise std σ_b.
- A weight perturbation shifts each pre-activation by std σ_l·√f_l·r_a.
- A bias perturbation shifts it by σ_b (no fan-in factor).
- To make the shift width-invariant: σ_l = s/√f_l for weights and σ_b = s·r_a for biases, with one dimensionless s. This is the `fanin` mode with s = 0.1·√44 ≈ 0.66 for weights. The mode keeps the bias σ at 0.1, which is about 0.15/r_a times the weight-induced shift s·r_a. Whether that ratio is right is a tuning question, not a derivation.
- **First layer:** f = 44 in every shape, so σ_1 = 0.1 in `fanin` mode. The first layer's input has r_a = the feature RMS (features are sparse and up to 2.3), so its effective shift is σ_1·√44·r_x, with r_x not necessarily 1.
- **Hidden layers:** the shift Δh in layer l propagates to the next layer's pre-activations through W_{l+1}, whose entries have std ~1/√f_{l+1}. The induced change is Δh_next ≈ rms(tanh'(h))·rms(Δh)·O(1) (an O(1) factor, independent of width), so each layer's contribution to the logit change stays width-invariant under the same rule. This uses the init scale of W_{l+1}. After drift to RMS 0.7 the propagation gain is no longer O(1) (it is ≈ 0.7·√f·rms(tanh')), but saturated units have tanh' ≈ 0, which partly offsets it.
- **Output layer:** logit shift = σ_out·√f_out·r_a. Same rule. The argmax is invariant to scaling all logits equally, so what matters is the **relative** perturbation, Δz divided by the spread of z across inputs, not the absolute σ_out.
- **Biases:** µP (Table 3) treats biases as vector-like with Θ(1) Adam LR and init, consistent with a width-independent bias σ. This is consistent with µP, not derived from it.
- **Depth (INFERENCE + TOY):** independent per-layer perturbations add in quadrature, so L layers give roughly √L times one layer's effect in the **linear** regime. But in the tanh net the logit scale also shrinks with depth (takeaway 8). In my toy the absolute RMS logit change under `fanin` was flat with depth (0.56-0.62), while the flip fraction rose. A √(2/L_w) depth factor on top of `fanin` (L_w weight layers) cut the absolute logit change too far (0.39 for [64,64,64], vs 0.61 for [64]) and brought the flip fraction only to 0.43 vs 0.41. **I do not propose a depth factor on σ.**

**TOY table** (400 Gaussian inputs, x ~ N(0, I_44), 20 perturbation draws, one init per shape; "global" is σ 0.1 on all weights and biases; "fanin" is weights σ_l = 0.1·√(44/f_l), biases 0.1). Columns: std across inputs of the logit difference z0 - z1 at init; RMS change of the 3 logits; argmax-flip fraction.

| shape | logit-diff std at init | RMS Δlogit global | RMS Δlogit fanin | flip global | flip fanin |
|---|---|---|---|---|---|
| 16 | 0.57 | 0.46 | 0.57 | 0.34 | 0.39 |
| 32 | 0.71 | 0.52 | 0.57 | 0.40 | 0.41 |
| 64 | 1.13 | 0.71 | 0.65 | 0.37 | 0.36 |
| 32,32 | 0.79 | 0.55 | 0.63 | 0.42 | 0.45 |
| 64,64 | 0.69 | 0.66 | 0.60 | 0.49 | 0.45 |
| 32,32,32 | 0.69 | 0.49 | 0.57 | 0.49 | 0.49 |
| 64,64,64 | 0.57 | 0.68 | 0.59 | 0.54 | 0.51 |

(Second run with another RNG state, `fanin` only: RMS Δlogit 0.61, 0.62, 0.61, 0.61, 0.59, 0.58, 0.56 and flip 0.41, 0.36, 0.40, 0.39, 0.45, 0.47, 0.53 for the same shape order. The two runs agree to about ±0.05, which is the noise level of this toy.) The point is only the trend: at σ ≈ 0.1 a single perturbation flips 35-55% of argmax decisions on random Gaussian inputs, in every shape. How that maps to the real, sparse observations is unknown.

**Other literature (what I looked for and found).**
- Lehman et al. 2018 (STATED): the layer-sensitivity quote in takeaway 5. SM-R "can rescale the perturbation as a whole, but it cannot granularly rescale each dimension". SM-G normalises each weight's perturbation by its gradient sensitivity.
- Such et al. 2017 (Uber, deep GA; text grep): the mutation power σ appears as a single scalar per run (0.00224, 0.005, 0.002 in one table; annealed to 0.001 in one place). No per-layer rule found in the text I searched.
- evosax and EvoJAX papers: no per-layer σ (text grep). es-practice item 6 records that no library default scales σ with width, and that ClipUp fixes the whole-vector radius ‖σ0‖ (so per-coordinate σ0 = r/√d under uniform σ).
- NES / CMA per-layer step sizes: I did not find a source that applies per-layer step sizes to a neural network; CMA-ES adapts a full covariance and NES scales λ and the σ learning rate with dimension (es-practice item 6, citing Wierstra et al.). **UNVERIFIED here for per-layer use.**
- Net: the 1/√fan_in rule is the only width-invariance argument I can state, and the only source for it is the Salimans duplicate-feature argument plus my own algebra above. It is a hypothesis for the sweep.

### 2.3 Question 3: initialisation for deeper tanh MLPs, and the depth plateau

**Facts (STATED).**
- Pennington, Schoenholz, Ganguli 2017 (arXiv 1711.04735): for a deep linear network, orthogonal init gives dynamical isometry (all Jacobian singular values 1). "Gaussian weights behave very differently." For nonlinear nets: ReLU cannot achieve isometry, sigmoidal (tanh-like) networks "can achieve isometry, but only with orthogonal weight initialization". They state that even critical Gaussian initialisations lead to a failure of isometry at large depth (for ReLU and hard-tanh, where the argument is explicit). Their depth-200, width-400 CIFAR-10 experiment (Fig. 4): orthogonal tanh with σ_w² = 1.05 "trains several orders of magnitude faster than the least isometric ReLU network".
- Their order-chaos picture: χ = σ_w²∫Dh φ'(√q*·h)² (eqn 5) and χ > 1 is chaotic (gradients explode, "forward signal propagation expands and folds space"), χ < 1 ordered ("contracts space"). The mean squared singular value of the Jacobian is χ^L.
- Schoenholz et al. 2017 (arXiv 1611.01232): depth scales ξ_q and ξ_c limit how deep signal can propagate; ξ_c diverges at the order/chaos transition; random networks are trainable "precisely when information can travel through them"; arbitrarily deep nets are trainable only near criticality. For tanh, ξ_q (single-input variance scale) is "consistently finite and significantly shorter" than ξ_c.
- Saxe, McClelland, Ganguli 2014 (arXiv 1312.6120): for **deep linear** networks, random orthogonal init gives "depth independent learning times" like greedy layerwise pre-training; scaled Gaussian does not.
- LeCun vs Xavier/Glorot: LeCun init is N(0, 1/fan_in); Xavier/Glorot is variance 2/(fan_in+fan_out). For equal-width hidden layers they coincide. They differ only at the 44→h layer (Xavier 2/(44+h), vs LeCun 1/44; for h=32: 0.026 vs 0.023) and the h→3 layer (Xavier 2/(h+3), vs 1/h; h=32: 0.057 vs 0.031). Arithmetic, not a source. Neither original paper was fetched here (Glorot and Bengio 2010, LeCun et al. 1998: UNVERIFIED).

**Why the project's default init is where it is (INFERENCE from STATED formulas).**
- By eqn 5, at σ_w = 1, σ_b = 0 and q* → 0, χ = 1. So LeCun-with-zero-bias is on the critical line, at the corner where activations decay (q* = 0). Tanh near zero is linear, so a freshly initialised 3-hidden-layer tanh net is close to a product of 4 Gaussian matrices, and Pennington says Gaussian products have a widening singular-value spectrum with depth.
- Practical consequence for an ES policy: the input-dependent part of the logits is dominated by a few amplified directions of the random product, and the logit scale falls with depth (TOY, takeaway 8). The decisions the policy needs ("duck/jump when an obstacle is near") depend on small input-dependent logit differences, which are both smaller and less diverse in the deeper net.

**Why a 3-layer net might plateau where a 2-layer one does, in a gradient-free setting (INFERENCE only; I found no source on ES plateau vs depth).**
- H1 (forward signal): smaller relative input-dependent logit spread at init (TOY above) means each ES perturbation changes decisions more erratically (flip 0.47-0.53 vs 0.36-0.41) while carrying less of the structure the rank signal needs.
- H2 (credit through more layers): a useful policy change needs coordinated changes in more matrices. Under antithetic ES with ranks, a coordinated change in k layers is harder to get by random direction than in k-1.
- H3 (tie-driven diffusion, from HANDOFF/es-practice item 1): on the plateau most pairs tie, so the ranks are partly index noise, and the escape is a random walk in θ. Adam's step is ~lr per coordinate in every shape, so the same walk covers a larger share of function space in the shallow net.
- H4 (dimension): the 3-hidden-layer shape has ~40% more parameters (44·32+32·32+32·32+32·3 ≈ 3.6k vs 2.5k), which alone is unlikely to explain it (Salimans: intrinsic, not nominal, dimension matters).
- **Cheap tests, no trainer changes needed beyond what the sweep does:** (a) record at init and at gen 0 the logit-difference spread across a fixed probe set of recorded states, per shape; (b) record the argmax-flip rate under the run's own perturbation; (c) run the orthogonal-init arm (section 3).
- Orthogonal init for the h→h layers is the standard remedy in the backprop literature (Saxe, Pennington). Evidence that it helps ES is nil in what I read. It is one arm, not a recommendation.

### 2.4 Question 4: sweep methodology

**Algorithms (STATED).**
- Hyperband (Li et al. 2018, arXiv 1603.06560): successive halving across several brackets that trade the number of configurations against budget per configuration; η = 3 is the default, R = 81 gives s_max = 4 and 5 brackets (their example). Brackets exist because the right amount of early-stopping aggressiveness is unknown. It allocates budget from intermediate performance, so it relies on early scores being informative about final ones (INFERENCE: the project's plateau violates this).
- ASHA (Li et al. 2020, arXiv 1810.05934): asynchronous version; promote a configuration to the next rung as soon as it is in the top 1/η of those evaluated so far at its rung; avoids the straggler problem of synchronous SHA. Useful here because ES runs have wildly different wall-clock per generation across shapes. Its benefit is parallelism; the project trains on one machine's worker threads, so I would not rely on it (INFERENCE).
- PBT (Jaderberg et al. 2017, arXiv 1711.09846): exploit = truncation selection (bottom 20% copy weights and hyperparameters from a uniformly sampled top-20% member) or binary tournament; explore = perturb each hyperparameter by ×1.2 or ×0.8, or resample from the prior.
- Bergstra and Bengio 2012 (JMLR 13): random search beats grid search in the same budget because "for most data sets only a few of the hyper-parameters really matter" and different ones matter on different data sets.

**Fit to this problem (INFERENCE).**
- Costs are asymmetric: the plateau generations are cheap (short episodes at ~54 points) and the late generations expensive (episodes up to the 1M cap). Rung budgets in generations therefore misprice the rungs. Budget in frames simulated (the trainer already benchmarks frames/s) or wall-clock.
- The early-rung outcome is bimodal, not graded (HANDOFF: plateau 55-88 points until a sudden climb). Rank-by-score at rung 0 selects on noise, and a late escaper (the 32,32,32 control sat on a plateau until gen ~1000-1100 in the v1-era run, per HANDOFF section 1 item 5) is killed by any aggressive η.
- So treat early rungs as **binary event or time-to-event**: "escaped by budget B_k" with a Wilson interval, or a Kaplan-Meier curve with censoring at the budget, compared by log-rank (standard survival analysis; not from the sources above).

**Seeds and statistics (STATED).**
- Henderson et al. 2018 (arXiv 1709.06560): 10 trials split into two sets of 5 gave "statistically different distributions just from varying random seeds" (their Fig. 5: average two-sample t-test t = -9.09, p = 0.0016). They note common practice of N < 5 trials or of reporting the top-N of several. They propose bootstrap CIs and power analysis.
- Agarwal et al. 2021 (arXiv 2108.13264): "Statistical concerns cannot be satisfactorily addressed with few runs"; claims with ≤ 3 runs raise eyebrows, folk wisdom is 20-30, but 95% CIs on medians suggest "closer to 50-100 runs in Atari 100k". They recommend the interquartile mean (25% trimmed mean: "discards the bottom and top 25% of the runs") with stratified bootstrap CIs, and performance profiles. IQM is "more robust to outliers" than the mean and "has considerably less bias than median".
- Atari 100k is not this problem: the outcomes there are continuous per-game scores. Here the plateau-escape rate is a proportion, so binomial methods apply to the escape rate, and IQM and bootstrap apply to post-escape scores.

**Seeds needed for the escape rate (my calculation, two-proportion normal approximation with pooled variance under H0, two-sided α = 0.05, power 0.8; no continuity correction, so slightly optimistic):**

| escape rate A vs B | seeds per arm |
|---|---|
| 0.05 vs 0.33 | 30 |
| 0.10 vs 0.33 | 49 |
| 0.10 vs 0.50 | 20 |
| 0.33 vs 0.50 | 137 |
| 0.33 vs 0.67 | 35 |
| 0.33 vs 0.90 | 11 |
| 0.50 vs 0.90 | 20 |

Wilson 95% intervals for context: 4/12 escapes is [0.14, 0.61]; 8/24 is [0.18, 0.53]; 12/36 is [0.20, 0.50]; 0/8 is [0.00, 0.32]; 0/12 is [0.00, 0.24]; 0/24 is [0.00, 0.14]. If the true rate were 1/3, P(0 escapes in n seeds) is 0.20 (n=4), 0.088 (n=6), 0.039 (n=8), 0.008 (n=12).

The takeaway is that the sweep can reliably (a) discard shapes or settings that are dead (0 of 12 or 0 of 16), and (b) detect large effects (≥ 2×). It cannot rank shapes whose escape rates differ by under ~0.3 unless it spends 35+ seeds per arm on them.

---

## 3. Proposed parametrisation (PROPOSAL, TO BE TESTED; nothing here is a verified result)

**Notation.** Layers l = 1..K+1 (K hidden layers), fan_in f_l (f_1 = 44), weights W_l, biases b_l. Reference values are today's: σ_ref = 0.1, lr_ref = 0.03, and the reference fan-in 44.

**P0: current behaviour.** Init N(0, 1/f_l); global σ 0.1 on weights and biases; global lr 0.03.

**P1: width-invariant perturbation (existing `--sigma-scale fanin`).**
- Weight σ_l = 0.1·√(44/f_l). The first layer stays 0.1. For f_l = 32: 0.117; f_l = 64: 0.083.
- Bias σ_b = 0.1 for all layers.
- Justification: section 2.2 (STATED Salimans width argument + INFERENCE algebra). The TOY table shows RMS Δlogit nearly flat across shapes.

**P2: per-layer learning rates, two candidates, chosen by a measurement before any long run.**
- P2a (µP-Adam analogue, assumes coherent updates): lr_l = 0.03·(44/f_l) for weights of layers 2..K+1 (hidden and output); first-layer weights and all biases 0.03. Source: STATED TP5 Table 3 (hidden and output Adam LR ∝ 1/fan_in, input and bias Θ(1)). The reference width is 44, not 32 or 64, to keep the first layer's lr at today's value.
- P2b (noise-dominated analogue): lr_l = 0.03·√(44/f_l) for weights of layers 2..K+1; first-layer weights and all biases 0.03. Equivalent to keeping lr_l/σ_l = 0.3 constant (the current ratio) in every layer. Source: INFERENCE, takeaway 3.
- P2c = P0's single lr = 0.03 (baseline, standard parametrisation).
- If you implement P2b by dividing each coordinate's Adam output by σ_l-relative factors, it is the same as running Adam in normalised coordinates φ with θ_l = σ_l·φ_l and global σ and lr. (INFERENCE; mathematically trivial.)
- Biases: keep lr = 0.03 and σ = 0.1 (Table 3 Θ(1)).

**P3: init.**
- Default: keep N(0, 1/f_l) (LeCun, weights; zero biases). Takeaway 6: the init is forgotten after tens of generations once the walk starts, so a different init value is expected to matter only for the escape phase.
- Arm "orth": for hidden→hidden layers only (not the 44→h or h→3 layers) use a random orthogonal matrix with gain 1 (σ_w = 1, Pennington's critical point). For K = 1 there is no hidden→hidden layer, so the arm only applies to K ≥ 2.
- Arm "small-out": output layer N(0, (0.01)²/f)-like scale, following es-starter's `normc(0.01)` (es-practice item 12) and µP's 1/fan_in² output init (TP5 Table 3). HANDOFF measured the project's init policy as 52% none / 47% duck, so this is a behaviour-level experiment, not only a scale one.
- No depth-dependent σ or lr multiplier is proposed (section 2.2: the TOY shows it undershoots).

**P4: what must be measured before trusting P2 (stage 0 of the protocol).**
- The coordinate check (mup README, STATED (via summary)): for each shape, per layer, RMS change of the pre-activations over a fixed probe set of recorded states, per generation, in the first 20-50 generations. It should be about the same across widths if the rule is right.
- Two widths (32 and 64, single hidden layer, same seeds) under the single global lr 0.03: if the layer-2 (output) change grows ×2 from 32 to 64, P2a; if ×√2, P2b; if ×1, the update is not scaling with fan-in at all and the premise fails.
- Also report: logit-difference spread, argmax-flip rate under the run's perturbation, and fraction |pre|>2, per layer, at gen 0, 50, 200.

**What would falsify the proposal.** (1) The best (s, ρ) found at width 32 is not near the best at 64 under P1+P2 either (the check in section 4). (2) The saturated fraction still grows with width under P1+P2. (3) The deeper shapes' escape rates stay at ~0 even with the "orth" arm; then the depth issue is not the init and needs a different lever (population, σ schedule, tie handling per es-practice).

---

## 4. Proposed sweep protocol (PROPOSAL, TO BE TESTED)

**Design principles.**
- Shapes: {16; 32; 64; 32,32; 64,64; 32,32,32; 64,64,64}, all on game v2, the project's seed ranges kept disjoint (training, validation; never the held-out `EVAL_SEEDS` for selection).
- Same list of random-search points for every shape, so transfer is directly readable (Bergstra and Bengio; takeaway 11).
- Budgets in frames (or wall-clock), not generations.
- Escape is a binary / time-to-event outcome until a shape has escaped. Score statistics only after.
- Define the escape criterion **before** looking at data, from the plateau level (HANDOFF: 54-88 points) as, for example, "validation median ≥ 3× the plateau level" (I leave the number to the user; it should sit well above 88 and well below the 1M cap).
- Log every run's per-generation validation score so time-to-escape can be reconstructed at any budget.

**Stage 0: coordinate and signal check (no sweep, 2 seeds per cell, tens of generations).**
- The P4 measurements on [16], [32], [64] and [32,32], [64,64], [32,32,32], [64,64,64].
- Decides P2a vs P2b vs P2c as the lr rule going forward. If both of P2a and P2b look equal, carry both as arms.

**Stage 1: continuous-parameter transfer on one deep and one shallow family (random search).**
- Sample ~12 points (s, ρ) from a log-uniform box around the current values: s ∈ [0.33, 1.3] (weights σ_l = s/√f_l, today 0.66) and ρ = lr/σ-ratio multiplier ∈ [0.3, 3]× today (INFERENCE: ranges to be revised after stage 0).
- Run them on [32] and [64] (one hidden layer: shortest plateau, cleanest width contrast), 4 seeds each, budget B_1 chosen so the baseline escapes in about half its runs (the project's 44-32-3 run reached held-out 10000 at gen 550, 1.6 min, per HANDOFF line ~668, so [32] is the cheap place to calibrate).
- Output: whether the best (s, ρ) at 32 and 64 coincide. This is the width-transfer test. If it fails under P1+P2, fall back to tuning s and ρ per width on the shapes that matter.

**Stage 2: shape race at rung 0 (the "escape race").**
- Configs: 7 shapes × {P0, P1+P2 (winner of stage 0/1)} = 14; plus the "orth" arm on the 3 shapes with K ≥ 2 and "small-out" on 2 shapes = 19 configs.
- Seeds: 8 per config in the first batch (152 runs); budget B_2 in frames equal to the cost of ~300 generations of the 32,32 baseline.
- Observations: escaped (yes/no) by B_2, and time of escape.
- Promotion/elimination (replaces top-1/η by score):
  - Drop a config if its Wilson upper bound at its current n is below the baseline's escape rate point estimate: e.g. 0 of 12 (upper 0.24) when the baseline shows 1/3.
  - Keep a config in the race while its Wilson interval overlaps the leader's.
  - Use η = 3 across rungs: about 19 → 6 → 2.
- This is **seed racing**: add seeds in batches of 4, recompute intervals, and stop a config early when it is dead or clearly leading. A dead config stops after 8-12 seeds, saving the budget for the ones that matter.

**Stage 3: mid rung (~6 surviving configs).**
- Seeds topped up to 24 per config (total 144 runs); budget B_3 = 3× B_2.
- Compare configs by (a) escape rate with Wilson intervals; (b) Kaplan-Meier time-to-escape, censored at B_3, with the log-rank test for pairs of interest; (c) among escaped runs, the validation-score trajectory, summarised by the IQM with stratified bootstrap 95% CI (Agarwal et al.).
- Seeds are paired across configs by the **environment** seed (training seeds and validation seeds), though not by init since the shapes differ. Use common random numbers within a shape across settings, which reduces variance for the s and ρ comparisons.

**Stage 4: finalists (2 configs, plus the baseline).**
- 36 seeds each (the table's 1/3 vs 2/3 case needs 35 per arm), full target budget, using the project's real metric: median held-out score over the 25 held-out seeds, capped at target, reported once at the end (`eval`, not used for selection).
- Report the proportion of seeds that reach the cap, IQM and median over seeds with 95% stratified bootstrap CIs, and a performance profile (fraction of runs above each score threshold; Agarwal).
- Pre-register the comparison (which pair, which metric), because with many configs and several metrics the chance of a spurious "winner" is large (Henderson: seeds alone produced p = 0.0016 differences).

**Cost sketch (INFERENCE).** Stage 2 is cheapest where it matters: the plateau is short in frames. Stage 4's runs are the expensive ones (late generations), so only ~3 configs × 36 seeds reach it. I did not estimate wall-clock; the per-frame cost in HANDOFF section 2.2 and the 15-minute 44-32-32-3 run to 500k (HANDOFF line ~681) are the inputs for that.

**What this protocol deliberately does not do.**
- No PBT: copying weights between runs removes the per-seed escape statistic (takeaway 12). It is useful as a production-training lever once a shape is chosen (a restart-from-best policy, in the spirit of es-practice's restart section).
- No Bayesian optimisation: the budget is dominated by seed variance, not by the smoothness of the response surface, and there are only two continuous dimensions plus a categorical shape.
- No claim that 24 seeds identifies the best of several close shapes; the table shows it does not.

---

## 5. References (accessed 2026-10-10)

"Text extracted" means the PDF was downloaded and its text layer read locally and grepped. "Via summary" means a summarising tool fetched it. "Cited only" means I saw the work only as a citation inside another source.

Verified (text extracted unless noted):
1. Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen, Gao. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv 2203.03466. https://arxiv.org/abs/2203.03466 (PDF https://arxiv.org/pdf/2203.03466). Tables 1-3, 8, 9, sections 3, 6, 6.1, G.2.2.
2. Yang, Yu, Zhu, Hayou. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. arXiv 2310.02244. https://arxiv.org/abs/2310.02244. Table 1, section 2, section 9.
3. Bordelon, Noci, Li, Hanin, Pehlevan. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit. arXiv 2309.16620. https://arxiv.org/abs/2309.16620.
4. microsoft/mup README. https://github.com/microsoft/mup (raw README fetched via summary).
5. Pennington, Schoenholz, Ganguli. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. arXiv 1711.04735. https://arxiv.org/abs/1711.04735.
6. Schoenholz, Gilmer, Ganguli, Sohl-Dickstein. Deep Information Propagation. arXiv 1611.01232 (ICLR 2017). https://arxiv.org/abs/1611.01232.
7. Saxe, McClelland, Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv 1312.6120. https://arxiv.org/abs/1312.6120.
8. Salimans, Ho, Chen, Sidor, Sutskever. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv 1703.03864. https://arxiv.org/abs/1703.03864. Section 3.2.
9. Lehman, Chen, Clune, Stanley. Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients. arXiv 1712.06563 (GECCO 2018). https://arxiv.org/abs/1712.06563.
10. Such, Madhavan, Conti, Lehman, Stanley, Clune. Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning. arXiv 1712.06567. https://arxiv.org/abs/1712.06567. Only the mutation-power values and the absence of a per-layer rule were checked (text from an earlier session, re-grepped now).
11. Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. arXiv 1603.06560. https://arxiv.org/abs/1603.06560.
12. Li, Jamieson, Rostamizadeh, Gonina, Ben-Tzur, Hardt, Recht, Talwalkar. A System for Massively Parallel Hyperparameter Tuning (ASHA). arXiv 1810.05934. https://arxiv.org/abs/1810.05934.
13. Jaderberg et al. Population Based Training of Neural Networks. arXiv 1711.09846. https://arxiv.org/abs/1711.09846.
14. Bergstra, Bengio. Random Search for Hyper-Parameter Optimization. JMLR 13 (2012). https://jmlr.org/papers/v13/bergstra12a.html (PDF https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).
15. Henderson, Islam, Bachman, Pineau, Precup, Meger. Deep Reinforcement Learning That Matters. arXiv 1709.06560. https://arxiv.org/abs/1709.06560.
16. Agarwal, Schwarzer, Castro, Courville, Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. arXiv 2108.13264. https://arxiv.org/abs/2108.13264.
17. evosax and EvoJAX papers: text from an earlier session, grepped for per-layer or layer-wise σ with no match (only a sentence about layer-wise data structures in evosax). Their code is covered by `research/es-practice.md`, not re-inspected.

Unverified (not fetched, or cited only):
18. Jelassi, Hanin, Ji, Reddi, Bhojanapalli, Kumar. Depth^(-3/2) learning-rate scaling for MLPs (title not fetched). Cited only inside Tensor Programs VI ("Jelassi et al. [11]") and Bordelon et al.
19. Bernstein et al. 2020/2023. Automatic Gradient Descent. Cited only inside Bordelon et al.
20. Poole et al. 2016, Exponential expressivity in deep neural networks through transient chaos. Cited only inside Schoenholz et al.
21. Glorot and Bengio 2010, Understanding the difficulty of training deep feedforward neural networks. Not fetched (init formula quoted from memory).
22. LeCun, Bottou, Orr, Müller 1998, Efficient BackProp. Not fetched (LeCun init quoted from memory).
23. Wierstra et al. 2014 (NES) and Nesterov and Spokoiny 2011, per-dimension scaling of λ and the σ learning rate. Cited only through es-practice.md and Salimans.

Counts: 17 entries verified (16 papers or documents text-extracted or fetched, of which 1 is via summary [mup README], plus the evosax/EvoJAX grep entry counted once); 6 entries unverified (items 18-23). A web search for "µP applied to evolution strategies" returned only gradient-based µP work, so I found no ES-specific µP paper; absence in one search is not proof it does not exist.
