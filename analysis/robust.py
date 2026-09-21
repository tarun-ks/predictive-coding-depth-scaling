"""Go/no-go table: is the depth exponent stable across dataset, width, epochs?

Every comparison is made against an MNIST/width-32/1-epoch baseline RESTRICTED TO
THE SAME DEPTHS as the condition it is compared with, otherwise a change in depth
range would be read as a change in exponent. All fits use the cluster-corrected
estimator (per-depth means, df = n_depths - 2).
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.report import loglog_fit, BP_FRACS


def load(pat):
    return [json.loads(l) for f in glob.glob(pat) for l in open(f) if l.strip()]


def key(r):
    return (r.get("dataset", "mnist"), r.get("width", 32), r.get("epochs", 1))


LADDER=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096]


def cell_complete(rs, L, method):
    """A cell is complete only when its whole ladder has run. Reading T_target
    off a partial cell biases low: a cell that has not yet reached its crossing
    budget yields None and is dropped, while one that has already crossed yields
    a value, so the survivors are systematically the fast ones."""
    need = 1 if method == "bp" else len([t for t in LADDER if t <= 4 * L])
    return len(rs) >= need


def condition_complete(rows, depths):
    cells = {}
    for r in rows:
        cells.setdefault((r["depth"], r["seed"], r["method"]), []).append(r)
    for L in depths:
        for s in range(5):
            for m in ("bp", "pc", "pcalm"):
                if not cell_complete(cells.get((L, s, m), []), L, m):
                    return False
    return True


def t_target(rows, frac, depths=None):
    """(depth, T) pairs per method, using per-seed depth-matched BP."""
    bp = {(r["depth"], r["seed"]): r["test_acc"] for r in rows if r["method"] == "bp"}
    out = {"pc": [], "pcalm": []}
    cells = {}
    for r in rows:
        if r["method"] == "bp":
            continue
        cells.setdefault((r["depth"], r["seed"], r["method"]), []).append(r)
    for (L, s, m), rs in cells.items():
        if depths and L not in depths:
            continue
        if (L, s) not in bp:
            continue
        rs = sorted(rs, key=lambda r: r["budget"])
        thr = frac * bp[(L, s)]
        h = next((r["budget"] for r in rs if r["test_acc"] >= thr), None)
        if h is not None:
            out[m].append((L, h))
    return out


def fit_of(pairs):
    return loglog_fit([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None


def fmt(f):
    if f is None:
        return "n/a"
    if f.get("exact_fit"):
        return f"{f['slope']:.3f} (exact)"
    return f"{f['slope']:.3f} [{f['ci_lo']:.3f},{f['ci_hi']:.3f}]"


def main():
    base_all = load("results/sweep/*.jsonl")
    rob = load("results/robust/*.jsonl")
    conds = {}
    for r in rob:
        conds.setdefault(key(r), []).append(r)

    print("=" * 96)
    print("GO / NO-GO: depth-exponent stability".center(96))
    print("=" * 96)
    for frac in BP_FRACS:
        print(f"\n### threshold = {int(frac*100)}% of depth-matched BP\n")
        print(f"  {'condition':34s} {'depths':14s} {'PC exponent':26s} {'PC-ALM exponent':26s}")
        print("  NOTE: robustness conditions span 4 depths -> df=2, so their intervals are")
        print("  wide by construction and cannot independently exclude 1.0. They test")
        print("  STABILITY OF THE POINT ESTIMATE; the ballistic exclusion rests on the")
        print("  6-depth main sweep.")
        print("  " + "-" * 92)
        for k in sorted(conds, key=lambda k: (k[2], k[1], k[0])):
            rows = conds[k]
            ds, w, ep = k
            depths = sorted({r["depth"] for r in rows})
            complete = condition_complete(rows, depths)
            tt = t_target(rows, frac, set(depths))
            bt = t_target(base_all, frac, set(depths))
            # Compare over the INTERSECTION of depths uncensored on BOTH sides.
            # Otherwise a depth censored only in the condition (its ladder is
            # shorter) silently changes the depth range of one fit but not the
            # other, and a range difference reads as an exponent difference.
            used = {}
            for m in ("pc", "pcalm"):
                dc = {L for L, _ in tt[m]}
                db = {L for L, _ in bt[m]}
                keep = dc & db
                used[m] = sorted(keep)
                tt[m] = [(L, t) for L, t in tt[m] if L in keep]
                bt[m] = [(L, t) for L, t in bt[m] if L in keep]
            fp, fa = fit_of(tt["pc"]), fit_of(tt["pcalm"])
            bp_, ba = fit_of(bt["pc"]), fit_of(bt["pcalm"])
            n_cells = len({(r['depth'], r['seed'], r['method']) for r in rows})
            label = f"{ds}/w{w}/{ep}ep" + ("" if complete else "  [PARTIAL]")
            if not complete:
                print(f"  {label:34s} {str(depths):14s} "
                      f"{'-- withheld, cells incomplete --':>26s}")
                print()
                continue
            print(f"  {label:34s} {str(depths):14s} {fmt(fp):26s} {fmt(fa):26s}")
            print(f"  {'  baseline mnist/w32/1ep':34s} "
                  f"{'PC' + str(used['pc']):14s} {fmt(bp_):26s} {fmt(ba):26s}")
            if used["pc"] != list(depths) or used["pcalm"] != list(depths):
                print(f"  {'    matched depths:':34s} PC={used['pc']}  "
                      f"PC-ALM={used['pcalm']}  (censored depths dropped from BOTH sides)")
            for nm, a, b in (("PC", fp, bp_), ("PC-ALM", fa, ba)):
                if a and b:
                    d = a["slope"] - b["slope"]
                    # Two exact fits have zero-width intervals; comparing them with
                    # float inequalities produces spurious "moved" verdicts, so
                    # compare the point estimates directly.
                    if a.get("exact_fit") and b.get("exact_fit"):
                        olap = abs(d) < 0.01
                        how = "point estimates (both exact fits)"
                    else:
                        tol = 1e-9
                        olap = not (a["ci_hi"] < b["ci_lo"] - tol
                                    or b["ci_hi"] < a["ci_lo"] - tol)
                        how = f"CI overlap, df={a['n_depths']-2}"
                    print(f"  {'    -> ' + nm + ' delta':34s} {d:+.3f}   "
                          f"{'STABLE' if olap else 'MOVED'}  ({how})")
            print()


if __name__ == "__main__":
    main()
