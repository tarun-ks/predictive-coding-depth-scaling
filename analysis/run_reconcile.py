"""Reconcile Task 2 (decay worsens as eta_h shrinks) with Task 3 (duals bounded).

Task 2's stated mechanism -- "smaller eta_h accumulates more dual per unit of
primal progress" -- predicts UNBOUNDED growth and is falsified by Task 3. The
finding survives; only the explanation dies. Candidate reconciliation: smaller
eta_h raises the dual PLATEAU or the tail FLUCTUATION AMPLITUDE.

Inference-only isolation: one model trained at the canonical setting, then the
inference loop re-run at scaled eta_h. This separates the inference dynamics
from any change in the trained weights. T_max scales as 1/lr_scale so every
condition is observed at the same multiple of its own knee.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import constraint_residuals, free_init, zero_duals_like, _solve_inner
from analysis.run_job import train
from analysis.eta_table import eta_for_depths


def traj(scales, skips, phi, state_lr, budget):
    @jax.jit
    def run(params, x, y):
        free = free_init(params, scales, skips, x, phi)
        duals = zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi))
        def step(carry, _):
            f, d = carry
            f = _solve_inner(params, scales, skips, x, y, f, d, state_lr, 1.0, 1, phi)
            rs = constraint_residuals(params, scales, skips, x, f, phi)
            d = [lam + 1.0 * r for lam, r in zip(d, rs)]
            dn = jnp.sqrt(sum(jnp.sum(l * l) for l in d))
            rel = jnp.max(jnp.stack([jnp.sqrt(jnp.sum(r * r)) /
                                     jnp.maximum(jnp.sqrt(jnp.sum(z * z)), 1e-30)
                                     for r, z in zip(rs, f)]))
            return (f, d), (dn, rel)
        _, ys = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return ys
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = a.depth
    base = eta_for_depths([L])[0][L]
    t0 = time.time()
    r = train("mnist", L, 32, "relu", a.seed, "pcalm", 2 * L, base, a.data_dir)
    rows = []
    for sc in (1.0, 0.5, 0.25, 0.1):
        budget = int(round(4096 / sc))
        dn, rel = traj(r["scales"], r["skips"], r["phi"], base * sc, budget)(
            r["params"], r["x"], r["y"])
        dn = np.asarray(dn, np.float64); rel = np.asarray(rel, np.float64)
        pk = int(dn.argmax())
        tail = dn[min(2 * pk, len(dn) - 2):]
        rows.append(dict(depth=L, seed=a.seed, lr_scale=sc, state_lr=base * sc,
                         budget=budget, peak=float(dn.max()), peak_T=pk + 1,
                         peak_over_L=(pk + 1) / L,
                         plateau=float(tail.mean()),
                         fluct_rel=float(tail.std() / max(tail.mean(), 1e-30)),
                         resid_end=float(rel[-1]),
                         test_acc=r["test_acc"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print(f"L={L} s={a.seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
