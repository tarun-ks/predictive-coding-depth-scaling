"""ITEM 1: where does PC-ALM's gradient misalignment live?

Exact multiplicative split of the total cosine to the true BP gradient:
    cos(g,b) = MAG * DIR
    MAG = sum_l ||g_l|| ||b_l|| / (||g|| ||b||)   -- norm-profile mismatch only
    DIR = sum_l w_l cos(g_l,b_l)                  -- per-layer angle only
with w_l = ||g_l|| ||b_l|| / sum_k ||g_k|| ||b_k||. Magnitude and direction
effects are therefore measured, not inferred from the total.
Also records per-layer cos(g_l,b_l) vs T and layer index for localization.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import Schedule, bp_loss, method_grad
from analysis.run_job import train
from analysis.eta_table import eta_for_depths


def decompose(g, b):
    gn = jnp.stack([jnp.sqrt(jnp.sum(x * x)) for x in g])
    bn = jnp.stack([jnp.sqrt(jnp.sum(x * x)) for x in b])
    cl = jnp.stack([jnp.sum(x * y) / jnp.maximum(jnp.sqrt(jnp.sum(x * x)) *
                    jnp.sqrt(jnp.sum(y * y)), 1e-30) for x, y in zip(g, b)])
    prod = gn * bn
    tot_g = jnp.sqrt(jnp.sum(gn ** 2)); tot_b = jnp.sqrt(jnp.sum(bn ** 2))
    mag = jnp.sum(prod) / jnp.maximum(tot_g * tot_b, 1e-30)
    w = prod / jnp.maximum(jnp.sum(prod), 1e-30)
    dirn = jnp.sum(w * cl)
    return (np.asarray(gn, np.float64), np.asarray(bn, np.float64),
            np.asarray(cl, np.float64), float(mag), float(dirn))


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
    r = train("mnist", L, 32, "relu", a.seed, "pcalm", 2 * L, slr, a.data_dir)
    sc, sk, phi, x, y, p = r["scales"], r["skips"], r["phi"], r["x"], r["y"], r["params"]
    b = jax.grad(lambda q: bp_loss(q, sc, sk, x, y, phi))(p)

    Ts, recs = [], []
    T = 1
    while T <= a.budget:
        Ts.append(T); T = max(T + 1, int(round(T * 1.5)))
    gn_s, bn_s, cl_s, mag_s, dir_s, tot_s = [], None, [], [], [], []
    for T in Ts:
        sch = Schedule(family="pcalm", budget=T, alpha=1.0, inner_steps=1,
                       weight_credit_timing="pre_dual_energy")
        g = method_grad(p, sc, sk, x, y, sch, state_lr=slr, rho=1.0, phi=phi)
        gn, bn, cl, mag, dirn = decompose(g, b)
        gn_s.append(gn); cl_s.append(cl); mag_s.append(mag); dir_s.append(dirn)
        tot_s.append(mag * dirn); bn_s = bn
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"L{L}_s{a.seed}.npz", Ts=np.array(Ts),
                        grad_norms=np.array(gn_s), bp_norms=bn_s,
                        per_layer_cos=np.array(cl_s), mag=np.array(mag_s),
                        dir=np.array(dir_s), total=np.array(tot_s))
    (od / f"L{L}_s{a.seed}.json").write_text(json.dumps(dict(
        depth=L, seed=a.seed, budget=a.budget, state_lr=slr,
        test_acc=r["test_acc"], n_layers=len(bn_s),
        wall_sec=round(time.time() - t0, 2))) + "\n")
    print(f"L={L} s={a.seed} layers={len(bn_s)} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
