"""MLX engine equivalence checks (research/plan-gpu-trainer.md §5); needs .venv-mlx (mlx + numpy).

E0  one step: for each feature mode, the stacked MLX gradient (float32, GPU) of 2 models against a
    float64 numpy recomputation of bc_train.py's gradient (same init, same rows, the fake-quantized
    weights from bc_common's float64 rules). Bar: max |d| / max |ref| <= 1e-4 per tensor, and the
    per-model loss sums within 1e-5 relative.
usage: .venv-mlx/bin/python test/bc-mlx-check.py <jump demos.bin> <duck demos.bin> <teacher logits (float64 n x 3) for the jump demos>
Only the first 2,048 rows of the jump demos and logits are read; the duck demos are read whole."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trainer'))
import bc_train_mlx as E   # first: it turns MLX's TF32 matmul off before mlx is imported
import numpy as np
import mlx.core as mx
from bc_common import arg_parser, Data, layer_sizes, init_params, quantize, quantize_pow2

ROWS, B = 2048, 1024
jump, duck, logits = sys.argv[1:4]
tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'bc-mlx-check-%d' % os.getpid())
os.makedirs(tmp, exist_ok=True)


def head(src, n, width, dtype):
    out = os.path.join(tmp, os.path.basename(src) + '.%d' % width)
    np.fromfile(src, dtype=dtype, count=n * width).tofile(out)
    return out


J, T = head(jump, ROWS, 46, np.float32), head(logits, ROWS, 3, np.float64)
D = duck   # all of it: the first rows of a duck file hold no duck transitions to weigh


def ref_grad(d, a, Ws, bs, idx, QB, SCHEME, dt=np.float64):
    """bc_train.py's gradient in dtype dt (float64: the reference; float32: numpy's own rounding,
    the noise floor) for one model (W, b float32 inputs)."""
    L = len(Ws)
    if QB and SCHEME == 'pow2':
        qs = [quantize_pow2(Ws[i], bs[i], QB) for i in range(L)]
        Wf, bf = [q[0] for q in qs], [q[1] for q in qs]
    elif QB:
        Wf, bf = [quantize(W, QB, SCHEME)[0] for W in Ws], [b.astype(np.float64) for b in bs]
    else:
        Wf, bf = [W.astype(np.float64) for W in Ws], [b.astype(np.float64) for b in bs]
    Wf, bf = [w.astype(dt) for w in Wf], [b.astype(dt) for b in bf]
    xb, yb, wb = d.X[idx].astype(dt), d.y[idx], d.w_all[idx].astype(dt)
    hs = [xb]
    for i in range(L):
        z = hs[-1] @ Wf[i].T + bf[i]
        hs.append(np.tanh(z) if i < L - 1 else z)
    zl = hs[-1]; z = zl - zl.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    ar = np.arange(len(idx))
    if not d.distill:
        tot, dtot = float((wb * -np.log(p[ar, yb] + 1e-12)).sum()), 0.0
        g = p.copy(); g[ar, yb] -= 1
    else:
        ys = d.y_solver[idx]
        hce = -np.log(p[ar, ys] + 1e-12)
        if d.Q is None:
            obj = -np.log(p[ar, yb] + 1e-12); g = p.copy(); g[ar, yb] -= 1
        else:
            tau, alpha = a.tau, a.alpha
            zs = zl / tau; zs = zs - zs.max(1, keepdims=True); pt = np.exp(zs); pt /= pt.sum(1, keepdims=True)
            q = d.Q[idx].astype(dt)
            obj = alpha * hce + (1 - alpha) * tau * tau * -(q * np.log(pt + 1e-12)).sum(1)
            g = (1 - alpha) * tau * (pt - q)
            if alpha > 0:
                oh = np.zeros_like(p); oh[ar, ys] = 1; g = alpha * (p - oh) + g
        tot, dtot = float((wb * hce).sum()), float((wb * obj).sum())
    g *= (wb / wb.sum())[:, None]
    gW, gb = [None] * L, [None] * L
    for i in reversed(range(L)):
        gW[i] = g.T @ hs[i]; gb[i] = g.sum(0)
        if i > 0: g = (g @ Wf[i]) * (1 - hs[i] ** 2)
    return gW + gb, tot, dtot


MODES = [
    ('plain 2x32', J, ['--hidden', '32,32']),
    ('plain 16x8', J, ['--hidden', ','.join(['8'] * 16)]),
    ('soft distill tau2 a.5', J, ['--hidden', '8,8', '--teacher-logits', T, '--distill-mode', 'soft', '--tau', '2', '--alpha', '0.5']),
    ('soft distill tau1 a0', J, ['--hidden', '8,8', '--teacher-logits', T, '--distill-mode', 'soft']),
    ('hard distill', J, ['--hidden', '8,8', '--teacher-logits', T, '--distill-mode', 'hard']),
    ('qat int4 row', J, ['--hidden', '8,8', '--quant-bits', '4', '--quant-scheme', 'row']),
    ('qat int4 layer', J, ['--hidden', '8,8', '--quant-bits', '4', '--quant-scheme', 'layer']),
    ('qat int4 pow2', J, ['--hidden', '8,8', '--quant-bits', '4', '--quant-scheme', 'pow2']),
    ('qat int8 pow2', J, ['--hidden', '32,32', '--quant-bits', '8', '--quant-scheme', 'pow2']),
    ('duck + transition weight 3', D, ['--hidden', '6', '--transition-weight', '3']),
]
fails = 0
for name, demos, extra in MODES:
    a = arg_parser().parse_args(['--demos', demos, '--obs-dim', '44', '--out', os.devnull] + extra)
    d = Data(a)
    layers = layer_sizes(a, d.X); L = len(layers) - 1
    seeds = [1, 2]
    inits = [init_params(np.random.default_rng(s), layers) for s in seeds]
    perms = np.stack([np.random.default_rng(10 + s).permutation(d.N)[:B] for s in seeds]).astype(np.int32)
    if d.transitions:   # the batch must contain weighted rows, or the mode tests nothing
        a2 = arg_parser().parse_args(['--demos', demos, '--obs-dim', '44', '--out', os.devnull] + extra[:2])
        changed = int((Data(a2).w_all[perms] != d.w_all[perms]).sum())
        print('      transition-weighted rows in the two batches: %d' % changed)
        if not changed: print('FAIL  no transition rows in the batch'); fails += 1
    Qd = mx.array(d.Q) if d.Q is not None else None
    grad = E.make_grad(L, E.fake_quant_fn(a.quant_bits, a.quant_scheme, L), mx.array(d.X), mx.array(d.y.astype(np.int32)), mx.array(d.w_all),
                       mx.array(d.y_solver.astype(np.int32)), Qd, d.distill, float(a.tau), float(a.alpha))
    G, tot, dtot = grad(E.stack_params(inits, L), mx.array(perms))
    worst, lossd, floor = 0.0, 0.0, 0.0
    for k in range(len(seeds)):
        R, rt, rd = ref_grad(d, a, inits[k][0], inits[k][1], perms[k], a.quant_bits, a.quant_scheme)
        R32, _, _ = ref_grad(d, a, inits[k][0], inits[k][1], perms[k], a.quant_bits, a.quant_scheme, np.float32)
        for j, r in enumerate(R):
            m = np.array(G[j][k]).reshape(r.shape)
            worst = max(worst, np.abs(m - r).max() / max(np.abs(r).max(), 1e-30))
            floor = max(floor, np.abs(R32[j] - r).max() / max(np.abs(r).max(), 1e-30))
        lossd = max(lossd, abs(float(np.array(tot)[k]) - rt) / abs(rt), abs(float(np.array(dtot)[k]) - rd) / max(abs(rd), 1e-30) if d.distill else 0.0)
    ok = worst <= 1e-4 and lossd <= 1e-5
    fails += not ok
    print('%s  E0 %-28s max rel grad err %.2e (numpy float32: %.2e)  loss rel err %.2e' % ('PASS' if ok else 'FAIL', name, worst, floor, lossd))
print('E0: %d/%d modes pass' % (len(MODES) - fails, len(MODES)))
sys.exit(1 if fails else 0)
