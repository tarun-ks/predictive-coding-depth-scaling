"""TASK 1 decisive control: the study proves PC-ALM converges to EXACT BP
gradients in LINEAR PC networks. If the implementation is faithful, linear
activation must show cos(grad, BP) -> 1 and settled duals. Any non-convergence
found with relu is then a nonlinear phenomenon, not a code defect."""
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.inference import Schedule, bp_loss, infer_for_schedule, method_grad, constraint_residuals
from pcalm.metrics import tree_cos, tree_l2
from pcalm.model import activation_fn, init_params, model_scales, skip_mask

W, L, BS, SEED = 32, 32, 64, 0


def eta_for(act, width, depth, dataset="mnist"):
    for r in csv.DictReader(open("configs/eta_best_by_cell.csv")):
        if (r["dataset"] == dataset and r["activation"] == act
                and int(r["N"]) == width and int(r["L"]) == depth):
            return float(r["eta_best_1_over_lambda_median"])
    raise KeyError


x_tr, y_tr, _, _ = load_dataset("mnist", train_subset=BS, test_subset=BS, seed=SEED,
                                data_dir="data", input_dim=784, output_dim=10)
x, y = jnp.asarray(x_tr[:BS]), jnp.asarray(y_tr[:BS])

print("activation,method,T,cos_to_TRUE_BP,dual_L2,max_rel_resid")
for act in ("linear", "relu"):
    phi = activation_fn(act)
    scales, skips = model_scales(W, L, 784), skip_mask(L)
    slr = eta_for(act, W, L)
    params = init_params(jax.random.PRNGKey(SEED), depth=L, width=W,
                         input_dim=784, output_dim=10, dtype=jnp.float32)
    g_bp = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
    T = 1
    while T <= 16384:
        for fam in ("pcalm",):
            sch = Schedule(family=fam, budget=T, alpha=1.0, inner_steps=1,
                           weight_credit_timing="pre_dual_energy")
            g = method_grad(params, scales, skips, x, y, sch, state_lr=slr, rho=1.0, phi=phi)
            free, duals = infer_for_schedule(params, scales, skips, x, y, sch,
                                             state_lr=slr, rho=1.0, phi=phi)
            rs = constraint_residuals(params, scales, skips, x, free, phi)
            mr = float(jnp.max(jnp.stack([
                jnp.sqrt(jnp.sum(r*r))/jnp.maximum(jnp.sqrt(jnp.sum(z*z)), 1e-30)
                for r, z in zip(rs, free)])))
            print(f"{act},{fam},{T},{float(tree_cos(g, g_bp)):.6f},"
                  f"{float(tree_l2(duals)):.4e},{mr:.4e}", flush=True)
        T *= 4
