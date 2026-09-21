"""Fit dual-norm growth: linear vs power-law vs exponential.
Windup predicts linear-to-polynomial with a persistent nonzero residual.
Exponential would mean the primal-dual iteration is unstable -- a different
finding, fitted separately and never merged with the other two."""
from __future__ import annotations
import glob, json
import numpy as np
from scipy import stats

def fits(T, y):
    """R^2 of three growth models, each in its own natural coordinates,
    all converted back to R^2 on the raw curve for comparability."""
    out = {}
    # linear: y = a + b T
    r = stats.linregress(T, y); pred = r.intercept + r.slope * T
    out["linear"] = (1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), r.slope)
    # power law: y = a T^p
    m = (T > 0) & (y > 0)
    r2 = stats.linregress(np.log(T[m]), np.log(y[m])); pred = np.exp(r2.intercept) * T ** r2.slope
    out["power"] = (1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), r2.slope)
    # exponential: y = a exp(bT)
    r3 = stats.linregress(T[m], np.log(y[m])); pred = np.exp(r3.intercept) * np.exp(r3.slope * T)
    out["exponential"] = (1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), r3.slope)
    return out

print("Dual-norm growth, fitted over T in [knee, 4096] (post-knee regime).")
print("R^2 on the raw curve for all three models; 'p' is the power-law exponent.\n")
for tag in ("trained", "init"):
    print(f"===== {tag} parameters =====")
    print(f"  {'L':>4s} {'seeds':>6s} {'lin R2':>8s} {'pow R2':>8s} {'p':>6s} {'exp R2':>8s}"
          f" {'resid@end':>10s} {'verdict':>14s}")
    for L in (16, 64, 128):
        fs = sorted(glob.glob(f"results/task3/L{L}_s*.npz"))
        if not fs: continue
        agg = {k: [] for k in ("linear", "power", "exponential")}
        ps, rend = [], []
        for f in fs:
            z = np.load(f)
            y = z[f"{tag}_dual_l2"]; rr = z[f"{tag}_resid_max"]
            T = np.arange(1, len(y) + 1)
            lo = 2 * L                      # post-knee only
            res = fits(T[lo:], y[lo:])
            for k in agg: agg[k].append(res[k][0])
            ps.append(res["power"][1]); rend.append(rr[-1])
        me = {k: float(np.mean(v)) for k, v in agg.items()}
        best = max(me, key=me.get)
        verdict = ("WINDUP" if best in ("linear", "power") and me[best] > 0.9
                   else "UNSTABLE" if best == "exponential" and me[best] > 0.9 else "unclear")
        print(f"  {L:4d} {len(fs):6d} {me['linear']:8.4f} {me['power']:8.4f} "
              f"{np.mean(ps):6.3f} {me['exponential']:8.4f} {np.mean(rend):10.2e} {verdict:>14s}")
    print()
# does the residual stay nonzero? (windup requires a persistent driving error)
print("Persistent residual check (windup needs residual NOT going to zero):")
for L in (16, 64, 128):
    fs = sorted(glob.glob(f"results/task3/L{L}_s*.npz"))
    if not fs: continue
    r0, r1 = [], []
    for f in fs:
        z = np.load(f); rr = z["trained_resid_max"]
        r0.append(rr[2 * L]); r1.append(rr[-1])
    print(f"  L={L:4d}  resid@2L={np.mean(r0):.3e}  resid@4096={np.mean(r1):.3e}  "
          f"ratio={np.mean(r1)/np.mean(r0):.2f}")
