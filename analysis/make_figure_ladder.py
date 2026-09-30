"""Figure: the rate ladder.

Every depth exponent measured or derived in this repository, on one axis. The point of the
figure is that the measured values are bracketed by two classical rates and a
communication floor, so a reader can see the whole argument at once.

The measured exponents and their intervals are recomputed from results/ by
analysis/summary_numbers.py (the script named beside each row produced the data). The
two classical rates are derived from the measured kappa exponent: Theta(kappa) has the
kappa exponent itself, Theta(sqrt(kappa)) half of it.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.summary_numbers import headline_fits, kappa_spectrum, momentum

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
CPC, CALM, CTH, CFL = "#0072B2", "#D55E00", "#666666", "#009E73"

K = kappa_spectrum(quiet=True)
H = headline_fits(quiet=True)
M = momentum(quiet=True)
P_KAPPA = K["kappa"]            # Theta(kappa): unaccelerated rate
P_ACCEL = K["kappa"] / 2        # Theta(sqrt(kappa)): ideally accelerated rate


def measured(res, key):
    return res[key], res[key + "_lo"], res[key + "_hi"]


# (label, exponent, lo, hi, kind, colour)   kind: 'theory' | 'measured'
ROWS = [
    ("$\\kappa$, conditioning\n(kappa.py, $L\\!=\\!4$–$256$)", *measured(K, "kappa"), "measured", CTH),
    ("$\\Theta(\\kappa)$: unaccelerated rate\nimplied by $\\kappa=\\Theta(L^2)$", P_KAPPA, None, None, "theory", CTH),
    ("PC, measured budget\n(run_sweep.py)", *measured(H, "pc7"), "measured", CPC),
    ("PC-ALM, measured budget\n(refine_ttarget.py)", *measured(H, "ref7"), "measured", CALM),
    ("Nesterov on PC, measured\n(run_momentum.py)", *measured(M, "nag"), "measured", CALM),
    ("$\\Theta(\\sqrt{\\kappa})$: ideally accelerated\nrate on the same $\\kappa$", P_ACCEL, None, None, "theory", CTH),
    ("$\\Omega(L)$ floor: any\nnearest-neighbour scheme", 1.00, None, None, "floor", CFL),
]
for lab, e, lo, hi, kind, _ in ROWS:
    print(f"{lab.splitlines()[0]:42s} {e:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""))

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

ax.axvspan(1.00, P_ACCEL, color=CFL, alpha=.10, zorder=0)
ax.axvline(1.00, color=CFL, lw=1.1, ls="-", zorder=1)
ax.axvline(P_KAPPA, color=CTH, lw=1.0, ls=":", zorder=1)

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
