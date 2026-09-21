"""One (dataset, depth, seed, method, T) training run.

Mirrors `pcalm.training.train_one` exactly -- same imported update fn, same
init, same batch order, same eval -- but additionally returns the trained
parameters so the residual instrumentation can be applied at the trained point
rather than only at init. No update rule is modified.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.inference import Schedule
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_init
from pcalm.training import (adam_learning_rate, batch_order, evaluate,
                            make_eval_fn, make_update_fn)
from analysis.eta_table import eta_for_depths
from analysis.instrument import make_resid_traj_fn, first_below

TOLS = [0.03, 0.01, 0.003]   # frozen before the sweep; see REPORT.md


def train(dataset, depth, width, activation, seed, method, budget, state_lr,
          data_dir, epochs=1, batch_size=64, eta0=1e-3, gamma0=1.0,
          train_subset=60000, test_subset=10000, alpha=1.0):
    lr = adam_learning_rate(width, depth, eta0, gamma0, None)
    phi = activation_fn(activation)
    scales, skips = model_scales(width, depth, 784), skip_mask(depth)
    x_tr, y_tr, x_te, y_te = load_dataset(dataset, train_subset=train_subset,
                                          test_subset=test_subset, seed=seed,
                                          data_dir=data_dir, input_dim=784, output_dim=10)
    params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width,
                         input_dim=784, output_dim=10, dtype=jnp.float32)
    init_p = params
    opt_state = adam_init(params)
    sch = Schedule(family=method, budget=budget if method != "bp" else 0, alpha=alpha,
                   inner_steps=1, weight_credit_timing="pre_dual_energy")
    update = make_update_fn(sch, scales, skips, phi, state_lr, 1.0, lr)
    eval_batch = make_eval_fn(scales, skips, phi)
    steps = 0
    for epoch in range(epochs):
        for idx in batch_order(x_tr.shape[0], batch_size, seed + epoch, True):
            params, opt_state = update(params, opt_state, jnp.asarray(x_tr[idx]),
                                       jnp.asarray(y_tr[idx]))
            steps += 1
    tr_mse, tr_ce, tr_acc = evaluate(params, x_tr, y_tr, batch_size, eval_batch)
    te_mse, te_ce, te_acc = evaluate(params, x_te, y_te, batch_size, eval_batch)
    return dict(params=params, init_params=init_p, scales=scales, skips=skips, phi=phi,
                x=jnp.asarray(x_tr[:batch_size]), y=jnp.asarray(y_tr[:batch_size]),
                steps=steps, learning_rate=lr,
                train_acc=tr_acc, test_acc=te_acc, train_mse=tr_mse, test_mse=te_mse,
                train_ce=tr_ce, test_ce=te_ce)


def residual_probe(r, depth, method, state_lr, probe_budget):
    """Literal T_res probe plus the residual minimum, at init and at trained params."""
    out = {"resid_probe_budget": probe_budget}
    fn = make_resid_traj_fn(r["scales"], r["skips"], r["phi"], family=method,
                            state_lr=state_lr, rho=1.0, alpha=1.0, inner_steps=1,
                            budget=probe_budget)
    for tag, p in (("init", r["init_params"]), ("trained", r["params"])):
        resid = np.asarray(fn(p, r["x"], r["y"]))
        out[f"resid_{tag}_T1"] = float(resid[0])
        out[f"resid_{tag}_min"] = float(resid.min())
        out[f"resid_{tag}_argmin"] = int(resid.argmin()) + 1
        out[f"resid_{tag}_final"] = float(resid[-1])
        for tol in TOLS:
            out[f"T_res_{tag}_tol{tol}"] = first_below(resid, tol)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-lr", type=float, default=None)
    ap.add_argument("--train-subset", type=int, default=60000)
    ap.add_argument("--test-subset", type=int, default=10000)
    ap.add_argument("--probe-residual", action="store_true")
    a = ap.parse_args()

    eta, prov = eta_for_depths([a.depth]) if a.state_lr is None else ({a.depth: a.state_lr}, {a.depth: "cli"})
    slr = eta[a.depth]
    t0 = time.time()
    r = train(a.dataset, a.depth, a.width, a.activation, a.seed, a.method, a.budget,
              slr, a.data_dir, train_subset=a.train_subset, test_subset=a.test_subset)
    row = dict(dataset=a.dataset, depth=a.depth, width=a.width, activation=a.activation,
               seed=a.seed, method=a.method, budget=a.budget, state_lr=slr,
               state_lr_provenance=prov[a.depth], learning_rate=r["learning_rate"],
               steps=r["steps"], train_acc=r["train_acc"], test_acc=r["test_acc"],
               train_mse=r["train_mse"], test_mse=r["test_mse"], test_ce=r["test_ce"],
               finite=bool(np.isfinite(r["test_acc"]) and np.isfinite(r["test_mse"])),
               wall_sec=round(time.time() - t0, 2))
    if a.probe_residual and a.method != "bp":
        row.update(residual_probe(r, a.depth, a.method, slr, max(8 * a.depth, 512)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
