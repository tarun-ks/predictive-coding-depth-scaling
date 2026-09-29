"""Figure: the rate ladder.

Every depth exponent measured or derived in this repository, on one axis. The point of the
figure is that the measured values are bracketed by two classical rates and a
communication floor, so a reader can see the whole argument at once.

The exponents plotted here are not recomputed: each is the value reported by
the analysis script named beside it, and this figure only arranges them.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("../paper_nn/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
CPC, CALM, CTH, CFL = "#0072B2", "#D55E00", "#666666", "#009E73"

# (label, exponent, lo, hi, kind, colour)   kind: 'theory' | 'measured'
ROWS = [
    ("$\\kappa$, conditioning\n(kappa.py, $L\\!=\\!4$–$256$)", 2.024, 1.953, 2.095, "measured", CTH),
    ("$\\Theta(\\kappa)$: unaccelerated rate\nimplied by $\\kappa=\\Theta(L^2)$", 2.02, None, None, "theory", CTH),
    ("PC, measured budget\n(run_sweep.py)", 1.916, 1.782, 2.050, "measured", CPC),
    ("PC-ALM, measured budget\n(refine_ttarget.py)", 1.208, 1.175, 1.241, "measured", CALM),
    ("Nesterov on PC, measured\n(run_momentum.py)", 1.213, 1.156, 1.271, "measured", CALM),
    ("$\\Theta(\\sqrt{\\kappa})$: ideally accelerated\nrate on the same $\\kappa$", 1.01, None, None, "theory", CTH),
    ("$\\Omega(L)$ floor: any\nnearest-neighbour scheme", 1.00, None, None, "floor", CFL),
]

fig, ax = plt.subplots(figsize=(5.6, 3.5))
ys = list(range(len(ROWS)))[::-1]
for y, (lab, e, lo, hi, kind, col) in zip(ys, ROWS):
    if kind == "measured":
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot([lo, lo], [y - .13, y + .13], color=col, lw=1.4, zorder=3)
            ax.plot([hi, hi], [y - .13, y + .13], color=col, lw=1.4, zorder=3)
        ax.plot([e], [y], "o", color=col, ms=7, zorder=4,
                markeredgecolor="white", markeredgewidth=.8)
    else:
        ax.plot([e], [y], "D" if kind == "theory" else "^", color=col, ms=7,
                zorder=4, markerfacecolor="white", markeredgewidth=1.6)

ax.axvspan(1.00, 1.01, color=CFL, alpha=.10, zorder=0)
ax.axvline(1.00, color=CFL, lw=1.1, ls="-", zorder=1)
ax.axvline(2.02, color=CTH, lw=1.0, ls=":", zorder=1)

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlabel("depth exponent $p$ in $\;\\propto L^{\\,p}$")
ax.set_xlim(0.92, 2.18)
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.grid(axis="y", visible=False)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=CPC, marker="o", lw=2.2, label="measured, with 95% CI"),
    Line2D([], [], color=CTH, marker="D", lw=0, markerfacecolor="white",
           markeredgewidth=1.6, label="classical rate for this $\\kappa$"),
    Line2D([], [], color=CFL, marker="^", lw=0, markerfacecolor="white",
           markeredgewidth=1.6, label="$\\Omega(L)$ communication floor"),
], loc="lower right", fontsize=7.5, framealpha=.95)

fig.savefig(OUT / "fig_ladder.pdf", dpi=600)
print("wrote", OUT / "fig_ladder.pdf")
