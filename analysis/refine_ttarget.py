"""T_target with a hold window, and with the ladder refined inside its bracket.

Two separate problems with the coarse reading:

1. RESOLUTION. The geometric ladder quantizes T_target to rungs a factor 1.4 apart,
   so the fitted slope inherits a systematic that no number of seeds removes.
   results/bisect/ refines the bracket that contains each crossing.

2. NON-MONOTONICITY. PC-ALM's accuracy is not monotone in T -- it peaks and then
   decays -- so "smallest T above threshold" can latch onto a single point that
   happens to clear the bar and is not sustained. We therefore define

       T_target = min{ T : acc(T') >= thr for every measured T' in [T, W*T] }

   with hold factor W. W = 1 recovers the naive first-crossing reading.

Both readings are reported. If they differ the naive one was latching onto noise.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats


def load_curves(depths, method):
    """budget -> test_acc per (depth, seed), merged across every source directory."""
    cur = {}
    for d in ("results/sweep", "results/bisect", "results/d256"):
        for f in glob.glob(f"{d}/L*_s*_{method}.jsonl"):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["depth"] not in depths or not r.get("finite", True):
                    continue
                cur.setdefault((r["depth"], r["seed"]), {})[r["budget"]] = r["test_acc"]
    return cur


def bp_accuracy():
    """Backpropagation accuracy per (depth, seed). The threshold is depth- AND
    seed-matched: collapsing it to one value per depth silently compares each
    method run against a different seed's baseline."""
    bp = {}
    for r in csv.DictReader(open("results/t_target_bp_relative.csv")):
        if r["frac"] == "0.9":
            bp[(int(r["depth"]), int(r["seed"]))] = float(r["bp_acc"])
    for f in glob.glob("results/d256/L256_s*_bp.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                bp[(r["depth"], r["seed"])] = r["test_acc"]
    return bp


def t_target(curve, thr, hold):
    """Smallest measured T whose threshold crossing is sustained out to hold*T."""
    Ts = sorted(curve)
    for i, T in enumerate(Ts):
        if curve[T] < thr:
            continue
        window = [t for t in Ts[i:] if t <= hold * T]
        if all(curve[t] >= thr for t in window):
            return T
    return None


def fit(d2t, label):
    Ls = np.array(sorted(d2t))
    if len(Ls) < 3:
        print(f"  {label:38s} too few depths")
        return
    ym = np.array([np.mean(np.log(d2t[L])) for L in Ls])
    r = stats.linregress(np.log(Ls), ym)
    half = stats.t.ppf(0.975, len(Ls) - 2) * r.stderr
    print(f"  {label:38s} slope={r.slope:+.3f}  95%CI=[{r.slope-half:+.3f},{r.slope+half:+.3f}]"
          f"  width={2*half:.3f}  R2={r.rvalue**2:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="pcalm")
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--holds", default="1.0,1.5,2.0")
    ap.add_argument("--depths", default="4,8,16,32,64,128,256")
    a = ap.parse_args()

    depths = {int(d) for d in a.depths.split(",")}
    bp, cur = bp_accuracy(), load_curves(depths, a.method)
    print(f"  {a.method}, threshold = {a.frac:g} x depth-matched BP\n")
    for hold in [float(h) for h in a.holds.split(",")]:
        detail, cens = {}, {}
        for (L, s), c in sorted(cur.items()):
            if (L, s) not in bp:
                continue
            t = t_target(c, a.frac * bp[(L, s)], hold)
            if t is None:
                cens[L] = cens.get(L, 0) + 1
            else:
                detail.setdefault(L, []).append(t)
        tag = "naive first crossing" if hold == 1.0 else f"hold x{hold:g}"
        print(f"  --- {tag} ---")
        for L in sorted(set(detail) | set(cens)):
            n, nc = len(detail.get(L, [])), cens.get(L, 0)
            flag = "  <-- CENSORED, excluded from fit" if nc else ""
            print(f"    L={L:4d} n={n} censored={nc} T={sorted(detail.get(L, []))}{flag}")
        # A depth with any censored seed is survivorship-biased downward: only the
        # seeds that happened to converge inside the budget are visible. Exclude it.
        clean = {L: v for L, v in detail.items() if not cens.get(L)}
        fit(clean, tag + " (uncensored depths)")
        print()


if __name__ == "__main__":
    main()
