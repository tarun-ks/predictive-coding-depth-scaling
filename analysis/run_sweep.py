"""Sweep the inference budget T for one (depth, seed, method) cell.

Emits one JSON line per T. The T=2L run (the paper's canonical budget rule)
additionally carries the constraint-residual probe at init and at trained params.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from analysis.run_job import train, residual_probe, TOLS
from analysis.eta_table import eta_for_depths

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
          1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576]

# Half-bin offset grid: every point shifted by ~sqrt(1.4)=1.19 so that none of the
# original bin edges are reused. If T_target is an artefact of where the geometric
# bins happen to fall, this grid must move it; if the exponent is unchanged, the
# invariance is real and not a binning coincidence.
OFFSET_LADDER = sorted({max(1, round(t * 1.19)) for t in LADDER})

# Two further offsets so the grid systematic is estimated from more than one
# displaced grid. 1.09 and 1.30 sit either side of 1.19 and, with
# it, sample the whole width of one ladder bin (ratio 1.4).
OFFSET_LADDERS = {
    "offset": 1.19,
    "offset09": 1.09,
    "offset30": 1.30,
}


def budgets_for(depth, ref_mult=4, grid="geometric"):
    hi = ref_mult * depth
    if grid == "geometric":
        src = LADDER
    else:
        m = OFFSET_LADDERS[grid]
        src = sorted({max(1, round(t * m)) for t in LADDER})
    out = [t for t in src if t <= hi]
    if hi not in out:
        out.append(hi)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-mult", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grid", choices=["geometric", "offset", "offset09", "offset30"],
                    default="geometric")
    ap.add_argument("--budgets", default="",
                    help="explicit comma-separated budgets, bypassing the ladder. Used to "
                         "refine T_target inside a single ladder bracket, where the coarse "
                         "rung spacing rather than seed noise sets the resolution.")
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()

    eta, prov = eta_for_depths([a.depth], dataset=a.dataset,
                               activation=a.activation, width=a.width)
    slr = eta[a.depth]
    if a.budgets:
        budgets = sorted({int(t) for t in a.budgets.split(",") if t.strip()})
    else:
        budgets = [0] if a.method == "bp" else budgets_for(a.depth, a.ref_mult, a.grid)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["budget"])

    with out_path.open("a") as f:
        for T in budgets:
            if T in done:
                continue
            t0 = time.time()
            r = train(a.dataset, a.depth, a.width, a.activation, a.seed, a.method,
                      max(T, 1), slr, a.data_dir, epochs=a.epochs, alpha=a.alpha)
            row = dict(dataset=a.dataset, depth=a.depth, width=a.width,
                       activation=a.activation, seed=a.seed, method=a.method,
                       budget=T, state_lr=slr, state_lr_provenance=prov[a.depth],
                       learning_rate=r["learning_rate"], steps=r["steps"],
                       epochs=a.epochs, grid=a.grid, alpha=a.alpha,
                       train_acc=r["train_acc"], test_acc=r["test_acc"],
                       train_mse=r["train_mse"], test_mse=r["test_mse"],
                       test_ce=r["test_ce"],
                       finite=bool(np.isfinite(r["test_acc"]) and np.isfinite(r["test_mse"])),
                       wall_sec=round(time.time() - t0, 2))
            if a.method != "bp" and T == 2 * a.depth:
                row.update(residual_probe(r, a.depth, a.method, slr, max(8 * a.depth, 512)))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{a.method} L={a.depth} s={a.seed}] T={T} acc={r['test_acc']*100:.2f}% "
                  f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
