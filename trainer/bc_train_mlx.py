"""MLX (Apple GPU) engine for bc-train / bc-train-batch --backend mlx (research/plan-gpu-trainer.md §4).

usage: python bc_train_mlx.py --demos demos.bin --obs-dim 44 --hidden 32,32 --seeds 1,2,3
                              --out-files r1.json,r2.json,r3.json [every bc_train.py option]

Trains N models of ONE shape at once (N = number of --seeds), stacked on a leading model axis:
W_i (N, out, in), b_i (N, 1, out), one batched matmul per layer, the whole step in one
mx.compile function. The algorithm is bc_train.py's, expression for expression, with the
gradient written by hand as there (no autodiff), so QAT's straight-through estimator is the
same by construction. Everything host-side is trainer/bc_common.py: each model keeps its own
np.random.default_rng(seed) on the CPU with bc_train.py's draw order (init, then per epoch the
permutation and the recall subsample), so --seed k means the same init and batch order as the
numpy backend; only float32 accumulation order differs (GPU). Saved params, scales and k are
computed in float64 on the CPU by bc_common.flat_params / quant_extras, so a QAT result is
exactly on its grid. QAT's per-step fake quantization runs on the GPU in float32 (training only).
Prints one JSON line per model per epoch (bc_train.py's row plus "model": k, 0-based) and writes
each model's result ({layers, params, log[, scales|k, transitions], engine}) to its --out-files
entry. --snapshot-dirs D0,D1,... (optional) writes each model's epoch-NN.json as bc_train.py does.
Needs mlx (the repo's .venv-mlx; PYTHON_MLX overrides the interpreter) and numpy.
Precision: MLX's GPU float32 matmul defaults to TF32 (10-bit mantissa; measured max relative
error ~9e-4 against float64 on the M5 Pro, against 3e-7 for numpy float32), which made per-step
gradients 1,000x noisier than the numpy backend's. MLX_ENABLE_TF32=0 (set below, before mlx is
imported) gives full float32 (2.5e-7); the result's engine record carries it."""
import json, os, time
os.environ['MLX_ENABLE_TF32'] = '0'
import numpy as np
import mlx.core as mx
from bc_common import arg_parser, Data, layer_sizes, init_params, flat_params, quant_extras, recall, write_snapshot

RECALL_SUB = 200000


def parse():
    ap = arg_parser()
    for act in ap._actions:   # --out / --seed are per model here: --out-files, --seeds
        if act.dest in ('out', 'seed'): act.required = False
    ap.add_argument('--seeds', required=True, help='comma-separated, one model per seed')
    ap.add_argument('--out-files', required=True, help='comma-separated, one per seed')
    ap.add_argument('--snapshot-dirs', default=None, help='comma-separated, one per seed')
    a = ap.parse_args()
    a.seed_list = [int(s) for s in a.seeds.split(',')]
    a.out_list = a.out_files.split(',')
    if len(a.out_list) != len(a.seed_list): raise SystemExit('--out-files needs one file per seed')
    a.snap_list = a.snapshot_dirs.split(',') if a.snapshot_dirs else None
    if a.snap_list is not None and len(a.snap_list) != len(a.seed_list): raise SystemExit('--snapshot-dirs needs one dir per seed')
    if a.snapshot_dir: raise SystemExit('use --snapshot-dirs (one per seed)')
    return a


def fake_quant_fn(QB, SCHEME, L):
    """(W list, b list) -> the (W, b) the forward pass uses, all float32 on the GPU, stacked (N, ...)."""
    if not QB: return lambda Ws, bs: (Ws, bs)
    qmax = float(2 ** (QB - 1) - 1)

    def rnd(x, scale_up, scale_down):
        return mx.clip(mx.floor(x * scale_up + 0.5), -qmax, qmax) * scale_down

    if SCHEME == 'pow2':
        def fq(Ws, bs):
            outW, outb = [], []
            for i in range(L):
                m = mx.maximum(mx.max(mx.abs(Ws[i]), axis=(1, 2)), mx.max(mx.abs(bs[i]), axis=(1, 2)))   # (N,)
                safe = mx.where(m > 0, m, 1.0)
                k = mx.floor(mx.log2(qmax / safe))
                k = k + mx.where(safe * mx.power(2.0, k + 1) <= qmax, 1.0, 0.0)   # bc_common.pow2_exponent's corrections
                k = k - mx.where(safe * mx.power(2.0, k) > qmax, 1.0, 0.0)
                k = mx.where(m > 0, k, 0.0)[:, None, None]
                up, down = mx.power(2.0, k), mx.power(2.0, -k)
                outW.append(rnd(Ws[i], up, down)); outb.append(rnd(bs[i], up, down))
            return outW, outb
        return fq

    def fq(Ws, bs):
        out = []
        for W in Ws:
            mxv = mx.max(mx.abs(W), axis=2, keepdims=True) if SCHEME == 'row' else mx.broadcast_to(mx.max(mx.abs(W), axis=(1, 2), keepdims=True), W.shape[:2] + (1,))
            sc = mx.where(mxv > 0, mxv / qmax, 1.0)
            out.append(mx.clip(mx.floor(W / sc + 0.5), -qmax, qmax) * sc)
        return out, bs
    return fq


def forward(xb, Ws, bs, L):
    hs = [xb]
    for i in range(L):
        z = hs[-1] @ mx.swapaxes(Ws[i], 1, 2) + bs[i]
        hs.append(mx.tanh(z) if i < L - 1 else z)
    return hs


def stack_params(inits, L):
    """Per-model (Ws, bs) from bc_common.init_params -> the stacked list [W_0..W_L-1, b_0..b_L-1]."""
    NM = len(inits)
    return [mx.array(np.stack([inits[k][0][i] for k in range(NM)])) for i in range(L)] + \
        [mx.array(np.stack([inits[k][1][i] for k in range(NM)])[:, None, :]) for i in range(L)]


def make_grad(L, fq, Xd, Yd, Wd, YSd, Qd, distill, tau, alpha):
    """grad(P, idx) -> (grads, tot, dtot): bc_train.py's per-step gradient (before Adam) for the
    stacked models on the rows idx (NM, B); tot / dtot the per-model loss sums of bc_train.py."""
    hard = distill and Qd is None
    ar = mx.arange(3)

    def grad(P, idx):
        Ws, bs = P[:L], P[L:]
        xb, yb, wb = Xd[idx], Yd[idx], Wd[idx]                       # (NM, B, ...)
        Wf, bf = fq(Ws, bs)
        hs = forward(xb, Wf, bf, L)
        zl = hs[-1]
        z = zl - mx.max(zl, axis=2, keepdims=True); p = mx.exp(z); p = p / mx.sum(p, axis=2, keepdims=True)
        oh = (yb[..., None] == ar).astype(mx.float32)
        scale = (wb / mx.sum(wb, axis=1, keepdims=True))[..., None]
        if not distill:
            tot = mx.sum(wb * -mx.log(mx.sum(p * oh, axis=2) + 1e-12), axis=1)
            dtot = mx.zeros_like(tot)
            g = (p - oh) * scale
        else:
            ohs = (YSd[idx][..., None] == ar).astype(mx.float32)
            hce = -mx.log(mx.sum(p * ohs, axis=2) + 1e-12)
            if hard:   # yb holds the teacher's labels
                obj = -mx.log(mx.sum(p * oh, axis=2) + 1e-12)
                g = p - oh
            else:
                zs = zl / tau; zs = zs - mx.max(zs, axis=2, keepdims=True); pt = mx.exp(zs); pt = pt / mx.sum(pt, axis=2, keepdims=True)
                q = Qd[idx]
                sce = -mx.sum(q * mx.log(pt + 1e-12), axis=2)
                obj = alpha * hce + (1 - alpha) * tau * tau * sce
                g = (1 - alpha) * tau * (pt - q)
                if alpha > 0: g = alpha * (p - ohs) + g
            tot = mx.sum(wb * hce, axis=1); dtot = mx.sum(wb * obj, axis=1)
            g = g * scale
        gW = [None] * L; gb = [None] * L
        for i in reversed(range(L)):
            gW[i] = mx.swapaxes(g, 1, 2) @ hs[i]; gb[i] = mx.sum(g, axis=1, keepdims=True)
            if i > 0: g = (g @ Wf[i]) * (1 - hs[i] * hs[i])
        return gW + gb, tot, dtot
    return grad


def main():
    a = parse()
    d = Data(a)
    X, y, w_all, N, Q, y_solver, distill = d.X, d.y, d.w_all, d.N, d.Q, d.y_solver, d.distill
    QB, SCHEME, epochs, B = a.quant_bits, a.quant_scheme, a.epochs, a.batch
    layers = layer_sizes(a, X)
    L = len(layers) - 1
    NM = len(a.seed_list)
    rngs = [np.random.default_rng(s) for s in a.seed_list]
    P = stack_params([init_params(r, layers) for r in rngs], L)
    M = [mx.zeros_like(p) for p in P]; V = [mx.zeros_like(p) for p in P]
    Xd, Yd, Wd, YSd = mx.array(X), mx.array(y.astype(np.int32)), mx.array(w_all), mx.array(y_solver.astype(np.int32))
    Qd = mx.array(Q) if Q is not None else None
    fq = fake_quant_fn(QB, SCHEME, L)
    grad = make_grad(L, fq, Xd, Yd, Wd, YSd, Qd, distill, float(a.tau), float(a.alpha))

    def step(P, M, V, idx, lr, c1, c2):
        grads, tot, dtot = grad(P, idx)
        M2 = [0.9 * m + 0.1 * gg for m, gg in zip(M, grads)]
        V2 = [0.999 * v + 0.001 * gg * gg for v, gg in zip(V, grads)]
        P2 = [pp - lr * (m / c1) / (mx.sqrt(v / c2) + 1e-8) for pp, m, v in zip(P, M2, V2)]
        return P2, M2, V2, tot, dtot

    cstep = mx.compile(step)
    if a.snap_list:
        for s in a.snap_list: os.makedirs(s, exist_ok=True)
    host = lambda: ([np.array(P[i]) for i in range(L)], [np.array(P[L + i])[:, 0, :] for i in range(L)])
    logs = [[] for _ in range(NM)]
    nb = N // B
    steps = epochs * nb
    t = 0
    wsum = float(w_all.sum())
    for ep in range(epochs):
        t0 = time.time()
        perm = mx.array(np.stack([r.permutation(N)[:nb * B] for r in rngs]).astype(np.int32))
        tot = mx.zeros((NM,)); dtot = mx.zeros((NM,))
        for s in range(nb):
            t += 1
            lr = np.float32(a.lr * 0.5 * (1 + np.cos(np.pi * t / steps)))
            c1, c2 = np.float32(1 - 0.9 ** t), np.float32(1 - 0.999 ** t)
            P, M, V, st, sd = cstep(P, M, V, perm[:, s * B:(s + 1) * B], mx.array(lr), mx.array(c1), mx.array(c2))
            tot = tot + st; dtot = dtot + sd
            mx.eval(P, M, V, tot, dtot)
        Wh, bh = host()
        per = [(Wh_k, bh_k) for Wh_k, bh_k in zip(zip(*Wh), zip(*bh))]
        if a.snap_list:
            for k in range(NM): write_snapshot(a.snap_list[k], epochs, ep, layers, flat_params(list(per[k][0]), list(per[k][1]), QB, SCHEME))
        tot_h, dtot_h = np.array(tot), np.array(dtot)
        Wf, bf = fq(P[:L], P[L:])
        for k in range(NM):
            sub = rngs[k].choice(N, min(N, RECALL_SUB), replace=False)
            zk = forward(Xd[mx.array(sub.astype(np.int32))][None], [w[k:k + 1] for w in Wf], [b[k:k + 1] for b in bf], L)[-1]
            pred = np.array(mx.argmax(zk[0], axis=1))
            row = dict(epoch=ep, loss=float(tot_h[k] / wsum), recall=recall(pred, y_solver[sub], d.has_duck), seconds=time.time() - t0)
            if distill: row['distillLoss'] = float(dtot_h[k] / wsum)
            logs[k].append(row)
            print(json.dumps(dict(row, model=k)), flush=True)
    Wh, bh = host()
    engine = dict(name='mlx', version=mx.__version__, device=str(mx.default_device()), models=NM, tf32=False)
    for k in range(NM):
        Wk, bk = [w[k] for w in Wh], [b[k] for b in bh]
        result = dict(layers=layers, params=flat_params(Wk, bk, QB, SCHEME), log=logs[k])
        result.update(quant_extras(Wk, bk, QB, SCHEME))
        if d.transitions is not None: result['transitions'] = d.transitions
        result['engine'] = engine
        with open(a.out_list[k], 'w') as f: json.dump(result, f)


if __name__ == '__main__':
    main()
