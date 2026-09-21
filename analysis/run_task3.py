"""TASK 3: dual-growth characterisation.

Task 2's mechanism (lambda += alpha*r fires once per iteration regardless of
eta_h) predicts LINEAR-to-POLYNOMIAL dual growth alongside a persistent nonzero
residual -- integral windup. EXPONENTIAL growth would instead mean the
primal-dual iteration is unstable. These are different findings and are fitted
separately here, never conflated.

Logged per inference iteration T: aggregate and per-layer dual L2, max and mean
relative constraint residual, and cosine between PC-ALM's weight-update
direction and the TRUE BP gradient at the same parameters (not to PC-ALM's own
large-T state, which is not a limit).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import (Schedule, bp_loss, constraint_residuals, free_init,
                             method_grad, zero_duals_like, _solve_inner)
from pcalm.metrics import tree_cos, tree_l2
from pcalm.model import activation_fn, model_scales, skip_mask
from analysis.run_job import train
from analysis.eta_table import eta_for_depths

BS, W = 64, 32


def traj_fn(scales, skips, phi, state_lr, rho, alpha, budget):
    @jax.jit
    def run(params, x, y):
        free = free_init(params, scales, skips, x, phi)
        duals = zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi))

        def step(carry, _):
            f, d = carry
            f = _solve_inner(params, scales, skips, x, y, f, d, state_lr, rho, 1, phi)
            rs = constraint_residuals(params, scales, skips, x, f, phi)
            d = [lam + alpha * r for lam, r in zip(d, rs)]
            dn = jnp.sqrt(sum(jnp.sum(l * l) for l in d))
            dmax = jnp.max(jnp.stack([jnp.sqrt(jnp.sum(l * l)) for l in d]))
            rel = jnp.stack([jnp.sqrt(jnp.sum(r * r)) / jnp.maximum(jnp.sqrt(jnp.sum(z * z)), 1e-30)
                             for r, z in zip(rs, f)])
            return (f, d), (dn, dmax, jnp.max(rel), jnp.mean(rel))
        _, ys = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return ys
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    L = a.depth
    slr = eta_for_depths([L])[0][L]
    t0 = time.time()
    r = train("mnist", L, W, "relu", a.seed, "pcalm", 2 * L, slr, a.data_dir)
    scales, skips, phi = r["scales"], r["skips"], r["phi"]
    x, y = r["x"], r["y"]

    out = {}
    for tag, p in (("init", r["init_params"]), ("trained", r["params"])):
        dn, dmax, rmax, rmean = traj_fn(scales, skips, phi, slr, 1.0, 1.0, a.budget)(p, x, y)
        g_bp = jax.grad(lambda q: bp_loss(q, scales, skips, x, y, phi))(p)
        cos = {}
        T = 1
        while T <= a.budget:
            sch = Schedule(family="pcalm", budget=T, alpha=1.0, inner_steps=1,
                           weight_credit_timing="pre_dual_energy")
            g = method_grad(p, scales, skips, x, y, sch, state_lr=slr, rho=1.0, phi=phi)
            cos[T] = float(tree_cos(g, g_bp))
            T = max(T + 1, int(round(T * 1.6)))
        out[tag] = dict(dual_l2=np.asarray(dn, np.float64).tolist(),
                        dual_max_layer=np.asarray(dmax, np.float64).tolist(),
                        resid_max=np.asarray(rmax, np.float64).tolist(),
                        resid_mean=np.asarray(rmean, np.float64).tolist(),
                        cos_to_true_bp={str(k): v for k, v in cos.items()})
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"L{L}_s{a.seed}.npz",
                        **{f"{t}_{k}": np.array(v) for t in out for k, v in out[t].items()
                           if k != "cos_to_true_bp"})
    (od / f"L{L}_s{a.seed}.json").write_text(json.dumps(dict(
        depth=L, seed=a.seed, budget=a.budget, state_lr=slr,
        test_acc=r["test_acc"], wall_sec=round(time.time() - t0, 2),
        cos_init=out["init"]["cos_to_true_bp"],
        cos_trained=out["trained"]["cos_to_true_bp"])) + "\n")
    print(f"L={L} s={a.seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
