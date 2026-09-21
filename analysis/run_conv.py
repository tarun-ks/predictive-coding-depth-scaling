"""ITEM 3: does the depth exponent survive an architecture change?

Conv variant on analysis/generic_pc.py (which reproduces the reference dense
cell to 0.0000 pp -- see validate_generic.py). Accuracy is not expected to be
competitive; the only question is the depth exponent.

Parameterisation transfers the reference rule faithfully:
  dense:  1/sqrt(fan_in) | 1/sqrt(width*depth) | 1/width
  conv :  1/sqrt(9)      | 1/sqrt(9C*depth)    | 1/C
eta_h uses the paper's own analytic rule eta_h = 1/lambda_max, with lambda_max
from power iteration on the AL energy Hessian in the activities. There is no
frozen table for conv, so the SAME estimator is used at every conv depth, which
keeps the depth comparison internally consistent. (On dense cells this estimator
runs ~3.5% above the paper's offline value; a constant offset shifts the
intercept, not the exponent.)
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.optim import adam_apply, adam_init
from pcalm.training import batch_order
from analysis.generic_pc import (al_energy_shifted, conv_block, constraint_residuals,
                                 dense_block, flatten_dense_block, forward,
                                 free_init, method_grad, zero_duals_like)

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
C, SPATIAL, K = 8, 7, 3


def build(depth, seed):
    blocks = [conv_block] + [conv_block] * (depth - 2) + [flatten_dense_block]
    skips = tuple([False] + [True] * (depth - 2) + [False])
    POOLED = 2  # 7x7 -> 2x2 via window 3 stride 3
    fan0, fanh, fanlast = K * K * 1, K * K * C, C * POOLED * POOLED
    scales = ([1.0 / math.sqrt(fan0)]
              + [1.0 / math.sqrt(fanh * depth)] * (depth - 2)
              + [1.0 / fanlast])
    ks = jax.random.split(jax.random.PRNGKey(seed), depth)
    params = [jax.random.normal(ks[0], (K, K, 1, C))]
    for i in range(1, depth - 1):
        params.append(jax.random.normal(ks[i], (K, K, C, C)))
    params.append(jax.random.normal(ks[depth - 1], (10, C * 2 * 2)))
    return params, scales, skips, blocks


def lambda_max(params, scales, skips, blocks, x, y, phi, iters=150):
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
    ap.add_argument("--train-subset", type=int, default=12800)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L, BS = a.depth, 64
    phi = jax.nn.relu
    x_tr, y_tr, x_te, y_te = load_dataset("mnist", train_subset=a.train_subset,
                                          test_subset=10000, seed=a.seed,
                                          data_dir=a.data_dir, input_dim=784, output_dim=10)
    def prep(v):
        v = jnp.asarray(v).reshape(-1, 28, 28, 1)
        p = 28 // SPATIAL
        return jax.lax.reduce_window(v, 0.0, jax.lax.add, (1, p, p, 1), (1, p, p, 1),
                                     "VALID") / (p * p)
    x_tr_c, x_te_c = prep(x_tr), prep(x_te)
    params0, scales, skips, blocks = build(L, a.seed)
    xb0, yb0 = x_tr_c[:BS], jnp.asarray(y_tr[:BS])
    lam = lambda_max(params0, scales, skips, blocks, xb0, yb0, phi)
    slr = 1.0 / lam
    width_eq = C * SPATIAL * SPATIAL           # activities per layer, dense analogue of width
    lr = 1e-3 * math.sqrt(width_eq / L)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip(): done.add(json.loads(line)["budget"])
    budgets = [0] if a.method == "bp" else [t for t in LADDER if t <= 4 * L]
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

            for idx in batch_order(x_tr_c.shape[0], BS, a.seed, True):
                params, opt = update(params, opt, x_tr_c[idx], jnp.asarray(y_tr[idx]))
            tot, n = 0.0, 0
            for s in range(0, x_te_c.shape[0], BS):
                e = min(s + BS, x_te_c.shape[0])
                lg = forward(params, scales, skips, blocks, x_te_c[s:e], phi)[-1]
                _, _, acc = mse_ce_accuracy(lg, jnp.asarray(y_te[s:e]))
                tot += float(acc) * (e - s); n += e - s
            row = dict(arch="conv", depth=L, seed=a.seed, method=a.method, budget=T,
                       channels=C, spatial=SPATIAL, state_lr=slr, lambda_max=lam,
                       learning_rate=lr, train_subset=a.train_subset,
                       test_acc=tot / n, finite=bool(np.isfinite(tot / n)),
                       wall_sec=round(time.time() - t0, 2))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[conv {a.method} L={L} s={a.seed}] T={T} acc={100*tot/n:.2f}% "
                  f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
