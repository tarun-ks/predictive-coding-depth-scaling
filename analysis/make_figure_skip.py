"""Figure: the conditioning-stability trade-off in skip strength.

Three panels over the same sweep (analysis/skip_strength.py), one line per skip
strength c in z_i = s_i*block_i(z_{i-1}) + c*z_{i-1}:

  (a) condition number of the activity Hessian at initialisation,
  (b) RMS activation of the last hidden layer at initialisation,
  (c) backpropagation test accuracy after one epoch.

Only c = 1 shows quadratic conditioning, and only c = 1 keeps a forward pass or
a trainable network. Everything is measured on the same initialised networks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

R = [json.loads(l) for l in open("results/strengthen/skip_strength.jsonl") if l.strip()]
by = {}
for r in R:
    by.setdefault((r["skip_strength"], r["depth"]), []).append(r)
cs = sorted({c for c, _ in by})
cmap = plt.cm.viridis(np.linspace(0, .88, len(cs)))

def series(c, key):
    Ls = sorted({L for cc, L in by if cc == c})
    return (np.array(Ls),
            np.array([np.mean([x[key] for x in by[(c, L)]]) for L in Ls]))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
FLOOR = 1e-18   # float32 underflow: exact zeros are drawn on the axis floor

for c, col in zip(cs, cmap):
    lab = f"$c={c:g}$" + (" (reference)" if c == 1.0 else "")
    L, k = series(c, "kappa")
    axes[0].loglog(L, k, "o-", color=col, ms=4, lw=1.5, label=lab)
    L, m = series(c, "act_rms_last")
    axes[1].semilogy(L, np.maximum(m, FLOOR), "o-", color=col, ms=4, lw=1.5)
    L, a = series(c, "bp_test_acc")
    axes[2].semilogx(L, 100 * a, "o-", color=col, ms=4, lw=1.5)

ref = np.array([4, 128])
axes[0].loglog(ref, 3.4 * ref ** 2, "k:", lw=1.2, zorder=0)
axes[0].annotate("$L^{2}$", (90, 3.4 * 90 ** 2 * 1.5), fontsize=8)
axes[0].set_xlabel("depth $L$"); axes[0].set_ylabel("$\\kappa$")
axes[0].set_title("(a) conditioning", fontsize=9, loc="left")

axes[1].axhline(FLOOR, color="k", lw=.8, ls="-")
axes[1].set_xscale("log")
axes[1].set_xlabel("depth $L$")
axes[1].set_ylabel("last hidden RMS")
axes[1].set_title("(b) forward pass", fontsize=9, loc="left")

axes[2].axhline(10, color="k", lw=.8, ls="--")
axes[2].annotate("chance", (5, 11.5), fontsize=7.5)
axes[2].set_xlabel("depth $L$"); axes[2].set_ylabel("BP test accuracy (\\%)")
axes[2].set_ylim(0, 100)
axes[2].set_title("(c) trainability", fontsize=9, loc="left")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(cs), fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_skip.pdf", dpi=600)
print("wrote", OUT / "fig_skip.pdf", "from", len(R), "runs")
