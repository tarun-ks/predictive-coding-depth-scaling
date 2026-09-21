"""Activity step size eta_h per depth for mnist / relu / width=32.

Depths 8..128 are the paper's FROZEN values, read straight from
configs/eta_best_by_cell.csv (column eta_best_1_over_lambda_median).
Depths 4 and 256 are not in that table; they are filled in using the paper's
own analytic rule eta_h = 1/lambda_max, with lambda_max from
analysis/lambda_max.py corrected by the constant factor 1.0354 by which that
estimator sits above the paper's table across all five covered depths.
Nothing here is tuned against any outcome metric.
"""
from __future__ import annotations

import csv
from pathlib import Path

CALIBRATION = 1.03542  # mean(mine / paper table) over L = 8,16,32,64,128

# lambda_max medians from analysis/lambda_max.py (seeds 0,1,2, mnist, relu, N=32)
_RAW_LAMBDA_UNCOVERED = {4: 5.205300, 256: 4.082377}

# For L=256 the corrected value lands just below the structural asymptote
# lambda_max -> 4 (the constraint operator is tridiagonal-Laplacian-like, so 4 is
# a floor approached from above). We clamp to that asymptote rather than cross it.
LAMBDA_ASYMPTOTE = 4.0


def paper_eta(path: str | Path = "configs/eta_best_by_cell.csv",
              dataset: str = "mnist", activation: str = "relu", width: int = 32):
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["activation"] == activation and int(row["N"]) == width:
                out[int(row["L"])] = float(row["eta_best_1_over_lambda_median"])
    return out


def eta_for_depths(depths, dataset="mnist", activation="relu", width=32, **kw):
    """Frozen-table eta for the given cell. The depth-4/256 fallbacks are only
    valid for the mnist/relu/width-32 cell they were computed for; any other
    cell must use depths the paper's table actually covers, so that robustness
    checks never rest on an extrapolated step size."""
    table = paper_eta(dataset=dataset, activation=activation, width=width, **kw)
    native = (dataset == "mnist" and activation == "relu" and width == 32)
    out, provenance = {}, {}
    for L in depths:
        if L in table:
            out[L] = table[L]
            provenance[L] = "paper_frozen_table"
        elif native and L in _RAW_LAMBDA_UNCOVERED:
            lam = max(_RAW_LAMBDA_UNCOVERED[L] / CALIBRATION, LAMBDA_ASYMPTOTE)
            out[L] = 1.0 / lam
            provenance[L] = ("calibrated_power_iteration_clamped"
                             if lam == LAMBDA_ASYMPTOTE else "calibrated_power_iteration")
        else:
            raise KeyError(
                f"no frozen eta for depth {L} at dataset={dataset} "
                f"activation={activation} width={width}; refusing to extrapolate")
    return out, provenance


if __name__ == "__main__":
    eta, prov = eta_for_depths([4, 8, 16, 32, 64, 128, 256])
    print("depth,state_lr,provenance")
    for L in sorted(eta):
        print(f"{L},{eta[L]:.6f},{prov[L]}")
