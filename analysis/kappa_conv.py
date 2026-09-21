"""Condition number of the PC activity Hessian for the CONV architecture.

Uses analysis/generic_pc.py (validated to 0.0000 pp against the reference cell)
so the energy is the same object as in the dense case. No training required.
Purpose: predict the conv T_target exponent from spectra alone, BEFORE the conv
training runs finish -- an out-of-sample test of the conditioning->iterations chain.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from pcalm.data import load_dataset
from analysis.generic_pc import (al_energy_shifted, constraint_residuals,
                                 free_init, zero_duals_like)
from analysis.run_conv import build, C, SPATIAL
from analysis.kappa import power


def _flat(a): return jnp.concatenate([x.ravel() for x in a])
def _unflat(v, like):
    out, i = [], 0
    for x in like:
        n = x.size; out.append(v[i:i+n].reshape(x.shape)); i += n
    return out


def conv_spectrum(L, seed, data_dir="data"):
    phi = jax.nn.relu
    params, scales, skips, blocks = build(L, seed)
    xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=seed,
                                data_dir=data_dir, input_dim=784, output_dim=10)
    v = jnp.asarray(xt[:1]).reshape(-1, 28, 28, 1)
    p = 28 // SPATIAL
    x = jax.lax.reduce_window(v, 0.0, jax.lax.add, (1, p, p, 1), (1, p, p, 1), "VALID") / (p*p)
    y = jnp.asarray(yt[:1])
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    gradf = jax.grad(lambda f: al_energy_shifted(params, scales, skips, blocks, x, y,
                                                 f, d0, 1.0, phi))
    hvp = jax.jit(lambda vv: _flat(jax.jvp(gradf, (f0,), (_unflat(vv, f0),))[1]))
    n = int(sum(z.size for z in f0))
    lmax = power(hvp, n, iters=3000)
    mv = lambda z: np.asarray(hvp(jnp.asarray(np.asarray(z).ravel(), jnp.float32)), np.float64)
    op = LinearOperator((n, n), matvec=lambda z: lmax*np.asarray(z).ravel() - mv(z), dtype=np.float64)
    try:
        mu = float(eigsh(op, k=1, ncv=min(n-1, 200), which="LA",
                         return_eigenvectors=False, tol=1e-9, maxiter=50000)[0])
        lmin = lmax - mu
    except Exception:
        return lmax, float("nan"), float("nan"), n
    return lmax, lmin, lmax/lmin, n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="8,16,32")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    print("depth,seed,lambda_max,lambda_min,kappa,n_activities")
    for L in [int(d) for d in a.depths.split(",")]:
        for s in [int(x) for x in a.seeds.split(",")]:
            lmax, lmin, k, n = conv_spectrum(L, s)
            print(f"{L},{s},{lmax:.6f},{lmin:.8f},{k:.3f},{n}", flush=True)
