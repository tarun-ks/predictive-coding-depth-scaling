"""Out-of-sample test: does a spectrum-based budget model calibrated on MNIST predict CIFAR-10?

Calibration uses MNIST only: PC as T = c * kappa, PC-ALM as T = c * kappa^q, fitted on the
coarse-ladder budgets at L = 8 to 128. The model is then evaluated on the CIFAR-10 spectra
(analysis/kappa_cifar.py) and compared, ladder rung by ladder rung, with the CIFAR-10 budgets
measured by analysis/run_cifar.py.
"""
import csv, json, glob, numpy as np

def kappa_by_depth(files):
    d = {}
    for fn in files:
        for r in csv.DictReader(open(fn)):
            try: d.setdefault(int(r["depth"]), []).append(float(r["kappa"]))
            except (ValueError, KeyError): pass
    return {L: float(np.exp(np.mean(np.log(v)))) for L, v in d.items()}

def budgets(pattern, method, depths, frac=0.9):
    R = [json.loads(l) for f in glob.glob(pattern) for l in open(f) if l.strip()]
    bp = {(r["depth"], r["seed"]): r["test_acc"] for r in R if r["method"] == "bp"}
    out = {}
    for L in depths:
        v = []
        for s in range(5):
            rr = sorted([r for r in R if r["method"] == method and r["depth"] == L and r["seed"] == s],
                        key=lambda r: r["budget"])
            if not rr or (L, s) not in bp: continue
            tau = frac * bp[(L, s)]
            for r in rr:
                if r["test_acc"] >= tau:
                    hold = [q for q in rr if r["budget"] <= q["budget"] <= 2 * r["budget"]]
                    if all(q["test_acc"] >= tau for q in hold): v.append(r["budget"]); break
        if v: out[L] = float(np.exp(np.mean(np.log(v))))
    return out

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
rung = lambda t: min(range(len(LADDER)), key=lambda i: abs(np.log(LADDER[i]) - np.log(t)))

kM = kappa_by_depth(["results/kappa/standard_w32.csv", "results/kappa/mup_256.csv"])
kC = kappa_by_depth(["results/strengthen/kappa_cifar.csv"])
cal = [8, 16, 32, 64, 128]
TMpc = budgets("results/sweep/*.jsonl", "pc", cal)
TMalm = budgets("results/sweep/*.jsonl", "pcalm", cal)
TCpc = budgets("results/strengthen/cifar/*.jsonl", "pc", [4, 8, 16, 32, 64])
TCalm = budgets("results/strengthen/cifar/*.jsonl", "pcalm", [4, 8, 16, 32, 64])
c_pc = float(np.exp(np.mean([np.log(TMpc[L] / kM[L]) for L in cal])))
q, lc = np.polyfit(np.log([kM[L] for L in cal]), np.log([TMalm[L] for L in cal]), 1)
dev = max(abs(kC[L] / kM[L] - 1) for L in kC)
print(f"kappa MNIST vs CIFAR-10: max relative difference {100*dev:.1f}% over L={sorted(kC)}")
print(f"MNIST calibration: PC T = {c_pc:.3f} kappa;  PC-ALM T = {np.exp(lc):.3f} kappa^{q:.3f}")
print(f"{'L':>4} {'PC pred':>8} {'meas':>6} {'rungs':>6} {'ALM pred':>9} {'meas':>6} {'rungs':>6}")
for L in sorted(TCpc):
    p1, p2 = c_pc * kC[L], np.exp(lc) * kC[L] ** q
    print(f"{L:>4} {p1:>8.1f} {TCpc[L]:>6.0f} {rung(TCpc[L])-rung(p1):>+6d} "
          f"{p2:>9.1f} {TCalm[L]:>6.0f} {rung(TCalm[L])-rung(p2):>+6d}")
