"""Figure: inner-solve cost versus depth for three solvers, work and rounds.

Left panel is work, the count a serial implementation pays. Right panel is rounds, the
count the nearest-neighbour floor bounds, which charges a level-k multigrid operation 2^k
nearest-neighbour hops. Steepest descent and Nesterov are level-0 throughout, so for
them the two counters coincide and only multigrid separates them.

THE REFERENCE MINIMUM MUST BE VERIFIED CONVERGED. The energy threshold is set relative
to an estimate E* of the achievable minimum, and that estimate has to be converged or
the threshold is too easy and every solver looks faster than it is. We therefore run
each depth at two reference budgets a factor of four apart and use the larger only
when the two agree. Shallow depths (8-64) are converged by 20k/80k; depths 128 and 256
are not, and need 200k/800k. Using the small reference at L=256 returns L^0.80 for
Nesterov, BELOW the Omega(L) floor and therefore impossible; that is the check earning
its keep.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("../paper_nn/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
CSD, CNAG, CMG, CFL = "#0072B2", "#D55E00", "#009E73", "#666666"

A = [json.loads(l) for l in open("results/strengthen/multigrid.jsonl") if l.strip()]
B = [json.loads(l) for l in open("results/strengthen/multigrid_deepref.jsonl") if l.strip()]
A = [r for r in A if r["converged"]]
B = [r for r in B if r["converged"]]
# depth -> (rowset, converged reference budget)
REF = {8: (A, 80000), 16: (A, 80000), 32: (A, 80000), 64: (A, 80000),
       128: (B, 800000), 256: (B, 800000)}


def series(solver, metric):
    d = {}
    for L, (rows, rw) in REF.items():
        g = [r[metric] for r in rows
             if r["solver"] == solver and r["depth"] == L and r["ref_work"] == rw]
        if g:
            d[L] = np.mean(g)
    Ls = sorted(d)
    return np.array(Ls), np.array([d[L] for L in Ls])


def expo(Ls, ys):
    f = stats.linregress(np.log(Ls), np.log(ys))
    tc = stats.t.ppf(.975, len(Ls) - 2)
    return f.slope, f.slope - tc * f.stderr, f.slope + tc * f.stderr


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, (metric, title) in zip(axes, [("work", "(a) work: fine-level evaluations"),
                                      ("rounds", "(b) rounds: nearest-neighbour hops")]):
    for solver, col, lab in (("sd", CSD, "steepest descent"),
                             ("nag", CNAG, "Nesterov"),
                             ("mg", CMG, "multigrid")):
        L, y = series(solver, metric)
        if len(L) < 2:
            continue
        p, lo, hi = expo(L, y)
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
axes[0].set_ylabel("iterations to $99\\%$ of the\nachievable energy reduction")
fig.tight_layout()
fig.savefig(OUT / "fig_solvers.pdf", dpi=600)
print("wrote", OUT / "fig_solvers.pdf")
for solver in ("sd", "nag", "mg"):
    for metric in ("work", "rounds"):
        if solver != "mg" and metric == "rounds":
            continue
        L, y = series(solver, metric)
        p, lo, hi = expo(L, y)
        print(f"  {solver:>4}/{metric:<7} L^{p:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  depths={list(L)}")
