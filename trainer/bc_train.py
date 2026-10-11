"""Python/numpy backend for `node trainer/cli.js bc-train --backend python`.

usage: python bc_train.py --demos demos.bin --obs-dim 44 --hidden 32,32 --epochs 15
                          --batch 1024 --lr 0.002 --seed 1 --out result.json

A port of research/scripts/feature-sufficiency/trainB.py (variant base), kept
expression for expression (dtypes, rng call order) so it reproduces that
prototype's params exactly on the same rows. The algorithm is the one in
trainer/bc.js (the node backend). Rows: float32 [obs-dim features, label, weight].
Labels are policy.js action indices (0 none, 1 jump, 2 duck). Class weights cover
all three classes, as in bc.js (absent classes get 0). Prints one JSON line per
epoch ({epoch, loss, recall: {none, jump[, duck]}, seconds}; duck only when the
demos have label-2 rows) and writes {layers, params, log} to --out; Node owns best.json.
--snapshot-dir DIR (optional) also writes DIR/epoch-NN.json ({layers, params, epoch};
NN and epoch 1-based) after each epoch's last optimiser step: a float64 copy only, so it
draws no RNG and changes no computation.
Distillation and quantization (off by default; trainer/bc.js header has the definitions):
--teacher-logits FILE (float64 rows x 3, computed by Node from the teacher) with
--distill-mode soft|hard [--tau T --alpha A]; --quant-bits N [--quant-scheme row|layer|pow2]
(QAT: fake-quantized weights in forward and backward, straight-through to the float
weights; the output params are the float64 q * scale, and the output gains `scales`;
pow2 also fake-quantizes the biases, one 2^-k per layer recomputed every step, and the
output gains `k` instead of `scales`).
Neither draws RNG. Distilled rows add `distillLoss`; `loss` stays the CE vs the solver.
--extra-demos FILE (repeatable, optional) appends FILE's rows after --demos, in order.
--transition-weight W (optional; absent or 1 = skipped) multiplies the weight of duck-transition
rows (onset and release, on the solver labels; trainer/bc.js transitionMask) by W after the
class weights.
The host-side parts (arguments, rows and weights, init, quantization, the flat layout) live in
trainer/bc_common.py, shared with other engines; this file is the numpy training step.
Needs numpy (the repo's .venv; the PYTHON env var overrides the interpreter)."""
import json, os, time
import numpy as np
from bc_common import arg_parser, Data, layer_sizes, init_params, quantize, quantize_pow2, flat_params, quant_extras, recall, write_snapshot


def main():
    a = arg_parser().parse_args()
    d = Data(a)
    X, y, w_all, N, Q, y_solver, distill = d.X, d.y, d.w_all, d.N, d.Q, d.y_solver, d.distill
    epochs = a.epochs
    QB, SCHEME = a.quant_bits, a.quant_scheme
    layers = layer_sizes(a, X)
    rng = np.random.default_rng(a.seed)
    Ws, bs = init_params(rng, layers)
    params = Ws + bs
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    lr0, B, t = a.lr, a.batch, 0
    L = len(Ws)
    POW2 = bool(QB) and SCHEME == 'pow2'

    def forward(xb, W=None, b=None):
        W = Ws if W is None else W
        b = bs if b is None else b
        hs = [xb]
        for i in range(L):
            z = hs[-1] @ W[i].T + b[i]
            hs.append(np.tanh(z) if i < L - 1 else z)
        return hs

    def fake_quant():
        """The (W, b) lists the forward pass uses: float32 fake-quantized under QAT, else the floats."""
        if not QB: return Ws, bs
        if POW2:
            qs = [quantize_pow2(Ws[i], bs[i], QB) for i in range(L)]
            return [w.astype(np.float32) for w, _, _ in qs], [b.astype(np.float32) for _, b, _ in qs]
        return [quantize(W, QB, SCHEME)[0].astype(np.float32) for W in Ws], bs

    def distill_grad(zl, p, idx, yb, wb):
        """Distilled output gradient (before the batch-weight scale), the solver-label CE sum and
        the objective sum; definitions in the trainer/bc.js header."""
        ar = np.arange(len(idx))
        ys = y_solver[idx]
        hce = -np.log(p[ar, ys] + 1e-12)
        if Q is None:   # hard: yb holds the teacher's labels
            obj = -np.log(p[ar, yb] + 1e-12)
            g = p.copy(); g[ar, yb] -= 1
        else:
            tau, alpha = a.tau, a.alpha
            zs = zl / tau; zs = zs - zs.max(1, keepdims=True); pt = np.exp(zs); pt /= pt.sum(1, keepdims=True)
            q = Q[idx]
            sce = -(q * np.log(pt + 1e-12)).sum(1)
            obj = alpha * hce + (1 - alpha) * tau * tau * sce
            g = (1 - alpha) * tau * (pt - q)
            if alpha > 0:
                oh = np.zeros_like(p); oh[ar, ys] = 1
                g = alpha * (p - oh) + g
        return g, float((wb * hce).sum()), float((wb * obj).sum())

    if a.snapshot_dir: os.makedirs(a.snapshot_dir, exist_ok=True)
    log = []
    steps = epochs * (N // B)
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(N); tot = 0.0; dtot = 0.0
        for s in range(N // B):
            idx = perm[s * B:(s + 1) * B]
            xb, yb, wb = X[idx], y[idx], w_all[idx]
            if POW2: Wf, bf = fake_quant()
            else: Wf, bf = [quantize(W, QB, SCHEME)[0].astype(np.float32) for W in Ws] if QB else Ws, bs
            hs = forward(xb, Wf, bf)
            z = hs[-1]; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            if not distill:
                tot += float((wb * -np.log(p[np.arange(B), yb] + 1e-12)).sum())
                g = p; g[np.arange(B), yb] -= 1; g *= (wb / wb.sum())[:, None]
            else:
                g, hsum, osum = distill_grad(hs[-1], p, idx, yb, wb)
                tot += hsum; dtot += osum
                g *= (wb / wb.sum())[:, None]
            gW = [None] * L; gb = [None] * L
            for i in reversed(range(L)):
                gW[i] = g.T @ hs[i]; gb[i] = g.sum(0)
                if i > 0: g = (g @ Wf[i]) * (1 - hs[i] ** 2)
            grads = gW + gb
            t += 1; lr = lr0 * 0.5 * (1 + np.cos(np.pi * t / steps))
            for j, (pp, gg) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * gg; v[j] = 0.999 * v[j] + 0.001 * gg * gg
                pp -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        if a.snapshot_dir:
            write_snapshot(a.snapshot_dir, epochs, ep, layers, flat_params(Ws, bs, QB, SCHEME))
        # train-set recall per class (subsample)
        sub = rng.choice(N, min(N, 200000), replace=False)
        pred = (forward(X[sub], *fake_quant()) if POW2 else forward(X[sub], [quantize(W, QB, SCHEME)[0].astype(np.float32) for W in Ws] if QB else None))[-1].argmax(1)
        row = dict(epoch=ep, loss=float(tot / w_all.sum()), recall=recall(pred, y_solver[sub], d.has_duck), seconds=time.time() - t0)
        if distill: row['distillLoss'] = float(dtot / w_all.sum())
        log.append(row)
        print(json.dumps(row), flush=True)
    result = dict(layers=layers, params=flat_params(Ws, bs, QB, SCHEME), log=log)
    result.update(quant_extras(Ws, bs, QB, SCHEME))
    if d.transitions is not None: result['transitions'] = d.transitions
    json.dump(result, open(a.out, 'w'))


if __name__ == '__main__':
    main()
