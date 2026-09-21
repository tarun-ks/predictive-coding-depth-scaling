"""Power-iteration estimate of lambda_max used by the paper's eta_h = 1/lambda_max rule.

Read-only instrumentation: imports the repo's energy verbatim and never mutates it.
Purpose is to extend the frozen eta table (configs/eta_best_by_cell.csv, depths 8..128)
to the depths this study needs (4 and 256) using the paper's own analytic rule rather
than a tuned value. Validated against the table at the depths it does cover.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from pcalm.data import load_dataset
from pcalm.inference import al_energy_shifted, constraint_residuals, free_init, zero_duals_like
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def _flat_dot(a, b):
    return sum(jnp.sum(x * y) for x, y in zip(a, b))


def _flat_norm(a):
    return jnp.sqrt(_flat_dot(a, a))


def lambda_max_at_init(*, depth, width, seed, dataset, data_dir, activation="relu",
                       rho=1.0, batch_size=64, iters=200, tol=1e-10):
    phi = activation_fn(activation)
    scales = model_scales(width, depth, 784)
    skips = skip_mask(depth)
    x_train, y_train, _, _ = load_dataset(
        dataset, train_subset=batch_size, test_subset=batch_size, seed=seed,
        data_dir=data_dir, input_dim=784, output_dim=10,
    )
    x = jnp.asarray(x_train[:batch_size])
    y = jnp.asarray(y_train[:batch_size])
    params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width,
                         input_dim=784, output_dim=10, dtype=jnp.float32)

    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))

    def energy(free_):
        return al_energy_shifted(params, scales, skips, x, y, free_, duals0, rho, phi)

    grad_fn = jax.grad(energy)

    @jax.jit
    def hvp(v):
        return jax.jvp(grad_fn, (free0,), (v,))[1]

    key = jax.random.PRNGKey(seed + 99991)
    keys = jax.random.split(key, len(free0))
    v = [jax.random.normal(k, z.shape) for k, z in zip(keys, free0)]
    n = _flat_norm(v)
    v = [t / n for t in v]

    lam_prev = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = float(_flat_dot(v, w))
        nw = float(_flat_norm(w))
        if nw == 0.0:
            break
        v = [t / nw for t in w]
        if abs(lam - lam_prev) < tol * max(abs(lam), 1.0):
            lam_prev = lam
            break
        lam_prev = lam

    # al_energy_shifted is a batch-MEAN energy; the paper's eta_h = 1/lambda_max is
    # per-sample. _solve_inner rescales the step by batch_size for exactly this
    # reason, so the matching curvature is batch_size * lambda(H_mean).
    return lam_prev * batch_size


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depths", default="4,8,16,32,64,128,256")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print("depth,lambda_max_median,eta_1_over_lambda,per_seed")
    for depth in depths:
        vals = [lambda_max_at_init(depth=depth, width=args.width, seed=s,
                                   dataset=args.dataset, data_dir=args.data_dir)
                for s in seeds]
        med = float(np.median(vals))
        print(f"{depth},{med:.6f},{1.0/med:.6f},\"{';'.join(f'{v:.6f}' for v in vals)}\"", flush=True)
