"""Is PC-ALM's refined budget a power law, or linear with a logarithmic correction?

Compares, by AICc on log T over the refined seven-depth ladder, a free power law
T = a L^p (2 parameters) with T = a L (1 + b ln L) (2 parameters), pure L ln L (1) and
pure L (1). Budgets are the hold-window T_target at 90% of depth-matched backpropagation.
"""
import json, glob, math, numpy as np
from scipy import stats, optimize

R = [json.loads(l) for f in glob.glob("results/bisect/*.jsonl") + glob.glob("results/sweep/*.jsonl")
     + glob.glob("results/d256/*.jsonl") for l in open(f) if l.strip()]
bp = {(r["depth"], r["seed"]): r["test_acc"] for r in R if r.get("method") == "bp"}

def tt(L, s):
    rr = sorted({r["budget"]: r for r in R if r.get("method") == "pcalm" and r["depth"] == L
                 and r["seed"] == s}.values(), key=lambda r: r["budget"])
    if (L, s) not in bp or not rr: return None
    tau = 0.9 * bp[(L, s)]
    for r in rr:
        if r["test_acc"] >= tau:
            hold = [q for q in rr if r["budget"] <= q["budget"] <= 2 * r["budget"]]
            if all(q["test_acc"] >= tau for q in hold): return r["budget"]
    return None

d = {}
for L in (4, 8, 16, 32, 64, 128, 256):
    v = [x for x in (tt(L, s) for s in range(5)) if x]
    if v: d[L] = float(np.exp(np.mean(np.log(v))))
Ls = np.array(sorted(d)); y = np.log([d[L] for L in Ls]); x = np.log(Ls); n = len(Ls)
aicc = lambda rss, k: n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
f = stats.linregress(x, y); r_pow = float(np.sum((y - f.intercept - f.slope * x) ** 2))
res = lambda p: np.log(np.maximum(p[0] * Ls * (1 + p[1] * np.log(Ls)), 1e-12)) - y
p = optimize.least_squares(res, [1.0, 0.5], bounds=([1e-6, 0], [1e3, 1e3])).x
r_log = float(np.sum(res(p) ** 2))
g = np.log(Ls * np.log(Ls)); r_lnl = float(np.sum((y - np.mean(y - g) - g) ** 2))
r_lin = float(np.sum((y - np.mean(y - x) - x) ** 2))
print("refined PC-ALM budgets:", {int(L): round(d[L], 1) for L in Ls})
print(f"power law  p = {f.slope:.3f}                AICc {aicc(r_pow,2):7.2f}")
print(f"a L (1 + b ln L)  a = {p[0]:.3f}, b = {p[1]:.3f}  AICc {aicc(r_log,2):7.2f}")
print(f"L ln L                            AICc {aicc(r_lnl,1):7.2f}")
print(f"L                                 AICc {aicc(r_lin,1):7.2f}")
print(f"Delta AICc (power law minus log-corrected linear) = {aicc(r_pow,2)-aicc(r_log,2):.2f}")
