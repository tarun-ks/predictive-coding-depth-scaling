"""Does the depth exponent survive a change of dataset to CIFAR-10?

The training-side exponent of the main sweeps rests on MNIST and Fashion-MNIST, which are
the same kind of data: 28x28 grayscale, linearly separable to a large extent. CIFAR-10
is a different distribution and a genuinely harder task, so it is the dataset change
that a reader is most likely to ask for.

The architecture is held: the reference residual MLP, the reference parameterisation,
the reference optimiser. Only the input dimension (3072 rather than 784) and the data
change. Absolute accuracy is NOT expected to be competitive -- a width-32 MLP is a
poor CIFAR model -- and it does not need to be, because the budget criterion is a
fraction of DEPTH-MATCHED backpropagation on the same architecture and data.

The diameter-floor diagnostic (the measured budget must exceed L-2) applies here and must be checked before
any exponent is quoted: if the measured budget sits well below L-2, the benchmark is
not exercising its depth and the exponent means nothing. That check is what
disqualified the convolutional variant, and it is reported for CIFAR either way.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.metrics import mse_ce_accuracy
from pcalm.model import init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import batch_order
from analysis.cifar_data import load_cifar10
from analysis.generic_pc import (al_energy_shifted, constraint_residuals, dense_block,
                                 forward, free_init, method_grad, zero_duals_like)

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
          1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288]
INPUT_DIM = 3072


def build(depth, seed, width):
    params = list(init_params(jax.random.PRNGKey(seed), depth=depth, width=width,
                              input_dim=INPUT_DIM, output_dim=10, dtype=jnp.float32))
    return params, model_scales(width, depth, INPUT_DIM), skip_mask(depth), [dense_block] * depth


def lambda_max(params, scales, skips, blocks, x, y, phi, iters=200):
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    gf = jax.grad(lambda f: al_energy_shifted(params, scales, skips, blocks, x, y, f,
                                              d0, 1.0, phi))
    hvp = jax.jit(lambda v: jax.jvp(gf, (f0,), (v,))[1])
    ks = jax.random.split(jax.random.PRNGKey(12345), len(f0))
    v = [jax.random.normal(k, z.shape) for k, z in zip(ks, f0)]
    nrm = lambda a: float(jnp.sqrt(sum(jnp.sum(t * t) for t in a)))
    n = nrm(v); v = [t / n for t in v]
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = float(sum(jnp.sum(a * b) for a, b in zip(v, w)))
        nw = nrm(w)
        if nw == 0: break
        v = [t / nw for t in w]
    return lam * x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--train-subset", type=int, default=50000)
    ap.add_argument("--test-subset", type=int, default=10000)
    ap.add_argument("--ref-mult", type=float, default=8.0)
    ap.add_argument("--budgets", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BS, phi, L = 64, jax.nn.relu, a.depth
    x_tr, y_tr, x_te, y_te = load_cifar10(train_subset=a.train_subset,
                                          test_subset=a.test_subset, seed=a.seed)
    x_tr, x_te = jnp.asarray(x_tr), jnp.asarray(x_te)
    params0, scales, skips, blocks = build(L, a.seed, a.width)
    xb0, yb0 = x_tr[:BS], jnp.asarray(y_tr[:BS])
    lam = lambda_max(params0, scales, skips, blocks, xb0, yb0, phi)
    slr = 1.0 / lam
    lr = 1e-3 * math.sqrt(a.width / L)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip(): done.add(json.loads(line)["budget"])
    if a.budgets:
        budgets = [int(t) for t in a.budgets.split(",")]
    elif a.method == "bp":
        budgets = [0]
    else:
        budgets = [t for t in LADDER if t <= a.ref_mult * L]

    with out.open("a") as fh:
        for T in budgets:
            if T in done: continue
            t0 = time.time()
            params = [p for p in params0]; opt = adam_init(params)

            @jax.jit
            def update(p, o, xb, yb):
                g = method_grad(p, scales, skips, blocks, xb, yb, a.method,
                                state_lr=slr, rho=1.0, alpha=1.0,
                                budget=max(T, 1), inner_steps=1, phi=phi)
                return adam_apply(p, g, o, lr)

            for idx in batch_order(x_tr.shape[0], BS, a.seed, True):
                params, opt = update(params, opt, x_tr[idx], jnp.asarray(y_tr[idx]))
            tot, n = 0.0, 0
            for i in range(0, x_te.shape[0], BS):
                e = min(i + BS, x_te.shape[0])
                lg = forward(params, scales, skips, blocks, x_te[i:e], phi)[-1]
                _, _, acc = mse_ce_accuracy(lg, jnp.asarray(y_te[i:e]))
                tot += float(acc) * (e - i); n += e - i
            row = dict(dataset="cifar10", depth=L, seed=a.seed, method=a.method,
                       budget=T, width=a.width, state_lr=slr, lambda_max=lam,
                       learning_rate=lr, train_subset=a.train_subset,
                       test_acc=tot / n, finite=bool(np.isfinite(tot / n)),
                       wall_sec=round(time.time() - t0, 2))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[cifar {a.method} L={L} s={a.seed}] T={T} acc={100*tot/n:.2f}% "
                  f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
