"""TASK 2: is the post-knee accuracy decay a step-size artefact or windup?

Re-runs the budget ladder around each condition's OWN knee. The knee moves as
~1/lr_scale (smaller activity steps need proportionally more iterations), so the
ladder is rescaled per condition; comparing at a fixed T would confound the two
effects. PC-ALM only -- PC shows no decay.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.run_job import train
from analysis.eta_table import eta_for_depths

KNEE = {64: 96, 128: 256}          # measured T_target(90% of BP) at lr_scale=1
MULTS = (0.5, 1.0, 2.0, 4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr-scale", type=float, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = a.depth
    base_eta, prov = eta_for_depths([L])
    slr = base_eta[L] * a.lr_scale
    knee = KNEE[L] / a.lr_scale
    budgets = sorted({max(1, int(round(knee * m))) for m in MULTS})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["budget"])

    with out.open("a") as f:
        for T in budgets:
            if T in done:
                continue
            t0 = time.time()
            r = train("mnist", L, 32, "relu", a.seed, "pcalm", T, slr, a.data_dir)
            row = dict(depth=L, seed=a.seed, method="pcalm", lr_scale=a.lr_scale,
                       state_lr=slr, base_state_lr=base_eta[L],
                       knee_for_scale=knee, budget=T,
                       mult_of_knee=round(T / knee, 3),
                       test_acc=r["test_acc"], train_acc=r["train_acc"],
                       test_mse=r["test_mse"],
                       finite=bool(np.isfinite(r["test_acc"]) and np.isfinite(r["test_mse"])),
                       wall_sec=round(time.time() - t0, 2))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[L={L} s={a.seed} scale={a.lr_scale}] T={T} "
                  f"({row['mult_of_knee']}x knee) acc={r['test_acc']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
