"""Full constraint-residual trajectory at init and at trained params (T = 2L).

Saves the whole curve so that T_res can be evaluated at any tolerance after the
fact, with one tolerance set applied uniformly to every depth, method and seed.
Training uses the paper's canonical budget rule T = 2L.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from analysis.run_job import train
from analysis.eta_table import eta_for_depths
from analysis.instrument import make_resid_traj_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--probe-mult", type=int, default=64)
    a = ap.parse_args()

    L = a.depth
    eta, prov = eta_for_depths([L])
    slr = eta[L]
    T_train = 2 * L
    t0 = time.time()
    r = train("mnist", L, 32, "relu", a.seed, a.method, T_train, slr, a.data_dir)
    probe = max(a.probe_mult * L, 2048)
    fn = make_resid_traj_fn(r["scales"], r["skips"], r["phi"], family=a.method,
                            state_lr=slr, rho=1.0, alpha=1.0, inner_steps=1,
                            budget=probe)
    curves = {}
    for tag, p in (("init", r["init_params"]), ("trained", r["params"])):
        curves[tag] = np.asarray(fn(p, r["x"], r["y"]), dtype=np.float64)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"L{L}_s{a.seed}_{a.method}"
    np.savez_compressed(outdir / f"{stem}.npz", init=curves["init"],
                        trained=curves["trained"])
    row = dict(depth=L, seed=a.seed, method=a.method, budget_train=T_train,
               state_lr=slr, state_lr_provenance=prov[L], probe_budget=probe,
               test_acc=r["test_acc"], train_acc=r["train_acc"],
               test_mse=r["test_mse"],
               finite=bool(np.isfinite(r["test_acc"]) and np.isfinite(r["test_mse"])),
               wall_sec=round(time.time() - t0, 2))
    for tag in ("init", "trained"):
        c = curves[tag]
        row[f"resid_{tag}_T1"] = float(c[0])
        row[f"resid_{tag}_min"] = float(c.min())
        row[f"resid_{tag}_argmin"] = int(c.argmin()) + 1
        row[f"resid_{tag}_final"] = float(c[-1])
        row[f"resid_{tag}_max"] = float(c.max())
    (outdir / f"{stem}.json").write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
