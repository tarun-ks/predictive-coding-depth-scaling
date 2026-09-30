"""A certificate that smooth-activation PC inference is non-convex at depth.

With a smooth activation the output-loss block B of H = A^T A + B can be indefinite.
Let -mu < 0 be its smallest eigenvalue, w the eigenvector, and choose u so that
Q_{n-1} u is parallel to w. The quarter-wave test vector v_k = sin(pi k / 2(n-1)) Q_k u
vanishes at the clamped input, and A acts on it as a first difference, so

    v^T H v  =  sum_k (phi_k - phi_{k-1})^2 |Q_k u|^2  -  mu |Q_{n-1} u|^2,

whose positive part shrinks as 1/n. If it is negative, H is PROVABLY indefinite. The
analytic crossing depth is n* ~ pi^2 D^2 / (8 mu), D = max_k |Q_k u| / |Q_{n-1} u|.
We evaluate v^T H v exactly with a Hessian-vector product, so the certificate is exact.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np
from pcalm.data import load_dataset
from pcalm.inference import al_energy_shifted, constraint_residuals, free_init, zero_duals_like
from pcalm.model import activation_fn
from analysis.isometry_kappa import build, layer_jacobians, output_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="orth:tanh:1,orth:tanh:1.2")
    ap.add_argument("--depths", default="4,8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        for spec in a.arms.split(","):
            arm, act, gain = spec.split(":"); gain = float(gain); phi = activation_fn(act)
            for L in [int(d) for d in a.depths.split(",")]:
                for s in [int(z) for z in a.seeds.split(",")]:
                    xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=s,
                                                data_dir="data", input_dim=784, output_dim=10)
                    x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
                    p, sc, sk = build(L, s, arm, act, gain)
                    free, Js = layer_jacobians(p, sc, sk, x, act)
                    B = output_block(p, sc, sk, x, y, free, act)
                    ev, V = np.linalg.eigh(0.5 * (B + B.T)); mu, w = -ev[0], V[:, 0]
                    n = len(free)
                    Q = [np.eye(B.shape[0])]
                    for J in Js: Q.append(J @ Q[-1])
                    try:
                        u = np.linalg.solve(Q[-1], w); u /= np.linalg.norm(u)
                    except np.linalg.LinAlgError:
                        continue
                    aks = np.array([np.linalg.norm(Qk @ u) for Qk in Q])
                    phik = np.sin(math.pi * np.arange(n) / (2 * (n - 1)))
                    v = [jnp.asarray((phik[k] * (Q[k] @ u))[None, :], jnp.float32) for k in range(n)]
                    d0 = zero_duals_like(constraint_residuals(p, sc, sk, x, free, phi))
                    g = jax.grad(lambda f: al_energy_shifted(p, sc, sk, x, y, f, d0, 1.0, phi))
                    Hv = jax.jvp(g, (free,), (v,))[1]
                    vHv = float(sum(jnp.sum(a_ * b_) for a_, b_ in zip(v, Hv)))
                    vv = float(sum(jnp.sum(a_ * a_) for a_ in v))
                    D = float(aks.max() / aks[-1])
                    nstar = (math.pi ** 2 * D ** 2 / (8 * mu)) if mu > 0 else float("inf")
                    row = dict(arm=arm, act=act, gain=gain, depth=L, seed=s, n=n, mu=float(mu),
                               D=D, rayleigh=vHv / vv, certified_indefinite=bool(vHv < 0),
                               n_star=nstar)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{spec} L={L} s={s}] mu={mu:.2e} D={D:.2f} RQ={vHv/vv:+.2e} "
                          f"certified_indef={vHv<0} n*={nstar:.1f}", flush=True)


if __name__ == "__main__":
    main()
