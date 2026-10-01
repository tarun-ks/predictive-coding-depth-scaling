"""Width-128 depth sweep: iterations and wall-clock time to reach the accuracy bar.

Reads results/wide128/ (written by run_wide.py and run_wide_extend.sh) and, if present,
results/wide128_timing.csv (written by timing_wide.py). For each arm, depth and seed,
T_target is the smallest budget at which test accuracy reaches a fraction of
backpropagation's at the same depth and seed and stays there up to 2T, as in the main sweep.
A cell that only qualifies at the largest budget run is right-censored, and a depth with any
censored seed is dropped from that arm's fit.

THE BAR MUST EXERCISE DEPTH. A budget below the L-1 floor means the bar was met before
credit could reach the input-side layers. At width 32 the 90% bar clears the floor
everywhere; at width 128 it does not, because the wider network clears 90% of
backpropagation with its input layers barely trained. The rule, fixed before looking at
exponents: use the loosest bar on the grid 90/95/97/98/99% at which every measured budget
of every arm clears the floor, and report the neighbouring bars as robustness checks.
"""
from __future__ import annotations
import csv, glob, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from analysis.report import loglog_fit, fmt
from analysis.refine_ttarget import t_target

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "wide128"
ARMS = ("pc", "pcalm", "nag")
HOLD, STEPS = 2.0, 937                      # one epoch of MNIST at batch 64
BARS = (0.90, 0.95, 0.97, 0.98, 0.99)


def curves(method):
    cur = {}
    for f in glob.glob(str(RES / f"L*_s*_{method}.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("finite", True):
                    cur.setdefault((r["depth"], r["seed"]), {})[r["budget"]] = r["test_acc"]
    return cur


def targets(cur, bp, frac):
    det, cens = {}, {}
    for (L, s), c in sorted(cur.items()):
        if (L, s) not in bp:
            continue
        t = t_target(c, frac * bp[(L, s)], HOLD)
        if t is None or t == max(c):
            cens[L] = cens.get(L, 0) + 1
        else:
            det.setdefault(L, []).append(t)
    return det, cens


def fit(det, cens, depths=None):
    Ls = sorted(L for L in det if not cens.get(L) and (depths is None or L in depths))
    return loglog_fit([L for L in Ls for _ in det[L]], [t for L in Ls for t in det[L]]), Ls


def main():
    bp = {k: v[0] for k, v in curves("bp").items()}
    print("backpropagation accuracy (%), mean over seeds:",
          {L: round(100 * np.mean([a for (d, s), a in bp.items() if d == L]), 2)
           for L in sorted({d for d, _ in bp})})
    cur = {m: curves(m) for m in ARMS}
    below = {}
    for frac in BARS:
        r = {m: targets(cur[m], bp, frac) for m in ARMS}
        below[frac] = sum(t < L - 1 for m in ARMS for L, ts in r[m][0].items() for t in ts)
    print("cells below the L-1 floor, by bar:", {f"{f:.0%}": n for f, n in below.items()})
    FRAC = next(f for f in BARS if below[f] == 0)
    print(f"bar used: {FRAC:.0%} of backpropagation (loosest with no cell below the floor)")
    for frac in BARS:
        r = {m: targets(cur[m], bp, frac) for m in ARMS}
        print(f"  at {frac:.0%}: " + "; ".join(f"{m} {fmt(fit(*r[m])[0])[:30]}" for m in ARMS))
    res = {m: targets(cur[m], bp, FRAC) for m in ARMS}
    depths = sorted({L for m in ARMS for L in res[m][0]} | {L for m in ARMS for L in res[m][1]})
    print(f"\nT_target at {FRAC:.0%} of backpropagation (per seed); c = censored seeds")
    print("depth  " + "  ".join(f"{m:>20s}" for m in ARMS))
    for L in depths:
        cells = []
        for m in ARMS:
            det, cens = res[m]
            cells.append(f"{str(sorted(det.get(L, []))):>16s}" + (f" c{cens[L]}" if cens.get(L) else "   "))
        print(f"{L:5d}  " + "  ".join(f"{c:>20s}" for c in cells))
    print("\ndepth exponents")
    fits = {}
    for m in ARMS:
        f, Ls = fit(*res[m])
        fits[m] = (f, Ls)
        print(f"  {m:6s} L={Ls}: {fmt(f)}")
    common = sorted(set(fits["nag"][1]) & set(fits["pcalm"][1]))
    for m in ("nag", "pcalm"):
        f, _ = fit(*res[m], depths=common)
        print(f"  {m:6s} over the common depths {common}: {fmt(f)}")

    tfile = ROOT / "results" / "wide128_timing.csv"
    if not tfile.exists():
        print("\nno timing file yet; run analysis/timing_wide.py on an idle machine")
        return
    cost = {(int(r["depth"]), r["method"]): (float(r["fixed_ms"]), float(r["per_iter_ms"]))
            for r in csv.DictReader(open(tfile))}
    print("\nwall-clock for one epoch at the budget each arm needs (minutes), and speedup")
    print("depth  " + "  ".join(f"{m:>8s}" for m in ARMS) + "   PC/Nesterov  PC-ALM/Nesterov")
    for L in depths:
        mins = {}
        for m in ARMS:
            det, cens = res[m]
            if L in det and not cens.get(L) and (L, m) in cost:
                fx, per = cost[(L, m)]
                mins[m] = STEPS * (fx + np.mean(det[L]) * per) / 6e4
        row = "  ".join(f"{mins[m]:8.2f}" if m in mins else f"{'n/a':>8s}" for m in ARMS)
        sp = lambda a, b: f"{mins[a] / mins[b]:11.1f}x" if a in mins and b in mins else f"{'n/a':>12s}"
        print(f"{L:5d}  {row}   {sp('pc', 'nag')}  {sp('pcalm', 'nag')}")


if __name__ == "__main__":
    main()
