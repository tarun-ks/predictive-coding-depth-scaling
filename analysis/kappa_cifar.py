"""Activity-Hessian spectrum at initialisation for the CIFAR-10 networks.

Same routine as analysis/kappa.py (validated against a dense eigendecomposition), same
reference architecture and parameterisation; only the input dimension (3072) and the
input sample change. Used to test whether a spectrum-based budget model calibrated on
MNIST predicts CIFAR-10 budgets it never saw.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from analysis.kappa import spectrum
from analysis.cifar_data import load_cifar10

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="4,8,16,32,64,128")
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
phi = activation_fn("relu")
print("depth,seed,lambda_max,lambda_min,kappa,n_activities")
for L in [int(d) for d in a.depths.split(",")]:
    for s in [int(z) for z in a.seeds.split(",")]:
        xt, yt, _, _ = load_cifar10(train_subset=64, test_subset=64, seed=s)
        x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
        params = init_params(jax.random.PRNGKey(s), depth=L, width=32, input_dim=3072,
                             output_dim=10, dtype=jnp.float32)
        lmax, lmin, k, n = spectrum(params, model_scales(32, L, 3072), skip_mask(L), x, y, phi)
        print(f"{L},{s},{lmax:.6f},{lmin:.8f},{k:.3f},{n}", flush=True)
