"""Figure: inner-solve cost versus depth for three solvers, work and rounds.

Left panel is work, the count a serial implementation pays. Right panel is rounds, the
count the nearest-neighbour floor bounds, which charges a level-k multigrid operation 2^k
nearest-neighbour hops and an exact line search one global reduction, 2(n-1) hops on a
chain of n layers. Nesterov uses a fixed step and needs no reduction, so for it the two
counters coincide; steepest descent and multigrid pay a reduction per line search.

THE REFERENCE MINIMUM MUST BE VERIFIED CONVERGED. The energy threshold is set relative
to an estimate E* of the achievable minimum, and that estimate has to be converged or
the threshold is too easy and every solver looks faster than it is. E* is the lowest
energy any iterate of any reference run reached. We run each depth at two reference
budgets a factor of four apart and use the larger only when the solver costs they imply
agree; with E* defined this way 20k and 80k agree at every depth. The final iterate is
not a usable E*: on this piecewise-quadratic energy steepest descent reaches its lowest
point early and then wanders above it (analysis/estar_wander.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from analysis.report import loglog_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
CSD, CNAG, CMG, CFL = "#0072B2", "#D55E00", "#009E73", "#666666"

ROWS = [json.loads(l) for f in sorted(Path("results/strengthen/multigrid").glob("*.jsonl"))
        for l in open(f) if l.strip()]
# depth -> converged reference budget (the larger of the two checked)
REF = {L: 80000 for L in (8, 16, 32, 64, 128, 256)}


def series(solver, metric):
    """Per-depth mean over seeds. A depth where any seed failed to reach the target within
    its work budget is censored and dropped, as in the training sweeps: keeping only the
    seeds that converged would be survivorship selection."""
    d = {}
    for L, rw in REF.items():
        g = [r for r in ROWS
             if r["solver"] == solver and r["depth"] == L and r["ref_work"] == rw]
        if g and all(r["converged"] for r in g):
            d[L] = [r[metric] for r in g]
        elif g:
            print(f"  {solver} L={L}: {sum(not r['converged'] for r in g)} of {len(g)} seeds "
                  f"did not converge; depth censored")
    return d


def expo(d):
    """Cluster-corrected log-log fit on per-depth means of log cost (analysis/report.py)."""
    f = loglog_fit([L for L in d for _ in d[L]], [v for L in d for v in d[L]])
    return f["slope"], f["ci_lo"], f["ci_hi"]


def points(d):
    Ls = sorted(d)
    return np.array(Ls), np.array([np.exp(np.mean(np.log(d[L]))) for L in Ls])


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, (metric, title) in zip(axes, [("work", "(a) work: fine-level evaluations"),
                                      ("rounds", "(b) rounds: nearest-neighbour hops")]):
    for solver, col, lab in (("sd", CSD, "steepest descent"),
                             ("nag", CNAG, "Nesterov"),
                             ("mg", CMG, "multigrid")):
        d = series(solver, metric)
        if len(d) < 3:
            continue
        p, lo, hi = expo(d)
        L, y = points(d)
        ax.loglog(L, y, "o-", color=col, ms=4, lw=1.6,
                  label=f"{lab}  $L^{{{p:.2f}}}$")
    ref = np.array([8, 256])
    ax.loglog(ref, ref / ref[0] * 32, "k:", lw=1.1, zorder=0)
    ax.annotate("$L^{1}$ floor", (110, 32 * 110 / 8 * .38), fontsize=7.5)
    ax.loglog(ref, (ref / ref[0]) ** 2 * 32, color=CFL, ls="--", lw=1.1, zorder=0)
    ax.annotate("$L^{2}$", (150, (150 / 8) ** 2 * 32 * .42), fontsize=7.5, color=CFL)
    ax.set_xlabel("depth $L$")
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left", framealpha=.95)
axes[0].set_ylabel("cost to $99\\%$ of the\nachievable energy reduction")
fig.tight_layout()
fig.savefig(OUT / "fig_solvers.pdf", dpi=600)
print("wrote", OUT / "fig_solvers.pdf")
for solver in ("sd", "nag", "mg"):
    for metric in ("work", "rounds"):
        if solver == "nag" and metric == "rounds":
            continue
        d = series(solver, metric)
        p, lo, hi = expo(d)
        print(f"  {solver:>4}/{metric:<7} L^{p:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  depths={sorted(d)}")
