"""RNG-stream gate (research/plan-gpu-trainer.md Phase 0, V0.2): prints the sha256 of each draw
bc_train.py makes from np.random.default_rng(seed), in its order (init normals per layer,
permutation per epoch, then the recall choice), for a few seeds. Run under two interpreters
(.venv and .venv-mlx) and diff: the MLX engine draws on the CPU with the shared code, so the
same --seed means the same init and batch order only if every hash matches.
usage: python test/rng-streams.py [rows]"""
import hashlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trainer'))
import numpy as np
from bc_common import init_params

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2062427
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
for seed in (1, 2, 7):
    for layers in ([44, 32, 32, 3], [44, 6, 3]):
        rng = np.random.default_rng(seed)
        Ws, bs = init_params(rng, layers)
        out = ['init ' + h(np.concatenate([w.ravel() for w in Ws]))]
        for ep in range(2):
            out.append('perm%d %s' % (ep, h(rng.permutation(N))))
            out.append('choice%d %s' % (ep, h(rng.choice(N, min(N, 200000), replace=False))))
        print('seed %d layers %s: %s' % (seed, '-'.join(map(str, layers)), ' '.join(out)))
print('numpy', np.__version__)
