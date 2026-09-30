"""Is the accelerated solver's excess over L^1 explained by conditioning that grows in training?

The inner solve at INITIALISATION reaches any fixed weight-gradient accuracy in Theta(sqrt kappa)
= Theta(L) iterations (analysis/gradient_tax.py), yet the training-side PC-ALM budget scales as
L^1.21. Innocenti et al. (2025) report that inference ill-conditioning spikes during training.
If kappa at TRAINED weights grows faster with depth than at initialisation, sqrt(kappa_trained)
could account for the whole excess with no free parameter.

Each network is trained by the reference pipeline (analysis/run_job.train, identical to the
main sweep: same init, batch order, step-size table, optimiser), with PC-ALM at the
coarse-ladder budget that first reaches 90% of depth-matched backpropagation. The spectrum is
then measured on the trained weights by the validated routine of analysis/kappa.py, on the
same input sample the initialisation spectrum used.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp
from pcalm.data import load_dataset
from analysis.eta_table import eta_for_depths
from analysis.kappa import spectrum
from analysis.run_job import train

T_COARSE = {4: 3, 8: 8, 16: 16, 32: 48, 64: 96, 128: 256, 256: 512}   # PC-ALM, 90% of BP

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="4,8,16,32,64,128")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--method", default="pcalm")
a = ap.parse_args()
print("depth,seed,method,budget,test_acc,lambda_max,lambda_min,kappa,wall_sec", flush=True)
for L in [int(d) for d in a.depths.split(",")]:
    eta, _ = eta_for_depths([L], dataset="mnist", activation="relu", width=32)
    for s in [int(z) for z in a.seeds.split(",")]:
        t0 = time.time()
        r = train("mnist", L, 32, "relu", s, a.method, T_COARSE[L], eta[L], "data")
        xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=s,
                                    data_dir="data", input_dim=784, output_dim=10)
        x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
        lmax, lmin, k, _ = spectrum(r["params"], r["scales"], r["skips"], x, y, r["phi"])
        acc = r.get("test_acc", r.get("te_acc", float("nan")))
        print(f"{L},{s},{a.method},{T_COARSE[L]},{acc},{lmax:.6f},{lmin:.8f},{k:.3f},"
              f"{time.time()-t0:.1f}", flush=True)
