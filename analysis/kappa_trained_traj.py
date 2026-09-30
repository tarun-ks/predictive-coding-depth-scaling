"""Activity-Hessian conditioning ALONG training, not only at its end.

analysis/kappa_trained.py measures kappa once, on the weights after the epoch. Inference
cost is paid at every update, and Innocenti et al. (2025) report that ill-conditioning
spikes during training, so the endpoint can understate what the inner solver faced. This
replicates analysis/run_job.train exactly (same data, init, batch order, update function,
step-size table, optimiser) and measures kappa at fixed fractions of the epoch, on the
same input sample the initialisation spectrum used. The fraction-1.0 value must equal
kappa_trained.py's, which is checked by the caller rather than assumed.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp
from pcalm.data import load_dataset
from pcalm.inference import Schedule
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_init
from pcalm.training import adam_learning_rate, batch_order, make_update_fn
from analysis.eta_table import eta_for_depths
from analysis.kappa import spectrum

T_COARSE = {4: 3, 8: 8, 16: 16, 32: 48, 64: 96, 128: 256, 256: 512}   # PC-ALM, 90% of BP
FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="4,8,16,32,64,128")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--method", default="pcalm")
a = ap.parse_args()
print("depth,seed,method,budget,frac,step,lambda_max,lambda_min,kappa", flush=True)
for L in [int(d) for d in a.depths.split(",")]:
    eta, _ = eta_for_depths([L], dataset="mnist", activation="relu", width=32)
    for s in [int(z) for z in a.seeds.split(",")]:
        # identical to analysis/run_job.train
        lr = adam_learning_rate(32, L, 1e-3, 1.0, None)
        phi = activation_fn("relu")
        scales, skips = model_scales(32, L, 784), skip_mask(L)
        x_tr, y_tr, _, _ = load_dataset("mnist", train_subset=60000, test_subset=10000, seed=s,
                                        data_dir="data", input_dim=784, output_dim=10)
        params = init_params(jax.random.PRNGKey(s), depth=L, width=32, input_dim=784,
                             output_dim=10, dtype=jnp.float32)
        opt = adam_init(params)
        sch = Schedule(family=a.method, budget=T_COARSE[L], alpha=1.0, inner_steps=1,
                       weight_credit_timing="pre_dual_energy")
        update = make_update_fn(sch, scales, skips, phi, eta[L], 1.0, lr)
        order = list(batch_order(x_tr.shape[0], 64, s, True))
        marks = {int(round(f * len(order))): f for f in FRACS}
        # the same probe sample as kappa.py and kappa_trained.py
        xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=s,
                                    data_dir="data", input_dim=784, output_dim=10)
        x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
        for step in range(len(order) + 1):
            if step in marks:
                lmax, lmin, k, _ = spectrum(params, scales, skips, x, y, phi)
                print(f"{L},{s},{a.method},{T_COARSE[L]},{marks[step]},{step},"
                      f"{lmax:.6f},{lmin:.8f},{k:.3f}", flush=True)
            if step < len(order):
                idx = order[step]
                params, opt = update(params, opt, jnp.asarray(x_tr[idx]), jnp.asarray(y_tr[idx]))
