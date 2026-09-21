"""T_conv with a SELF-CONSISTENT reference budget.

run_conv.py gated the reference solve on per-chunk movement, which is too loose
when convergence is slow: with time constant tau, a chunk of c iterations moves
only ~D*c/tau of the remaining distance D, so a small chunk movement can still
leave D far above the measurement tolerance. That biased T_conv downward at
large depth.

Here the reference budget B is doubled until B >= 3 * T_conv(tightest tol), so
the reference is always well past the convergence time it is used to measure.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from analysis.instrument import make_advance_fn, make_traj_fn, initial_state, first_below
from analysis.eta_table import eta_for_depths

TOLS = [0.1, 0.03, 0.01]
MARGIN = 3.0
BS, W = 64, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", default="pc")
    ap.add_argument("--max-ref", type=int, default=8_000_000)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = a.depth
    slr = eta_for_depths([L])[0][L]
    phi = activation_fn("relu")
    scales, skips = model_scales(W, L, 784), skip_mask(L)
    x_tr, y_tr, _, _ = load_dataset("mnist", train_subset=BS, test_subset=BS,
                                    seed=a.seed, data_dir=a.data_dir,
                                    input_dim=784, output_dim=10)
    x, y = jnp.asarray(x_tr[:BS]), jnp.asarray(y_tr[:BS])
    params = init_params(jax.random.PRNGKey(a.seed), depth=L, width=W,
                         input_dim=784, output_dim=10, dtype=jnp.float32)

    t0 = time.time()
    B = max(8 * L, 1024)
    free, duals = initial_state(params, scales, skips, x, phi)
    ref_free, ref_duals, built = free, duals, 0
    tconv, dist0, ok = {t: None for t in TOLS}, None, False

    while B <= a.max_ref:
        step = B - built
        if step > 0:
            adv = make_advance_fn(scales, skips, phi, family=a.method, state_lr=slr,
                                  rho=1.0, alpha=1.0, inner_steps=1, budget=step)
            ref_free, ref_duals = adv(params, x, y, ref_free, ref_duals)
            built = B
        traj = make_traj_fn(scales, skips, phi, family=a.method, state_lr=slr, rho=1.0,
                            alpha=1.0, inner_steps=1, budget=B)
        _resid, dist = traj(params, x, y, ref_free)
        dist = np.asarray(dist)
        dist0 = float(dist[0])
        tconv = {t: first_below(dist, t) for t in TOLS}
        tight = tconv[min(TOLS)]
        if tight is not None and B >= MARGIN * tight:
            ok = True
            break
        B *= 2

    row = dict(depth=L, seed=a.seed, method=a.method, B_final=built,
               self_consistent=int(ok), dist_at_T1=dist0,
               margin_achieved=(built / tconv[min(TOLS)]) if tconv[min(TOLS)] else None,
               wall_sec=round(time.time() - t0, 2))
    for t in TOLS:
        v = tconv[t]
        row[f"T_conv_tol{t}"] = v if ok else None
        row[f"floor_limited_tol{t}"] = int(v == 1) if v else 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
