"""Is quadratic conditioning a property of the skip connection, or of signal propagation?

Claim under test. At the forward-pass point the PC residuals vanish, so the activity
Hessian is exactly H = A^T A + B, with A block lower-bidiagonal (identity on the
diagonal, -J_k below it, J_k the layer Jacobian) and B the supervised term on the last
layer. For ReLU this is exact, not a linearisation. Two bounds follow, for any unit
direction u with Q_k = J_k ... J_1 and D_u = max_k |Q_k u| / min_k |Q_k u|:

  lower:  kappa >= n^2 / (pi^2 D_u^2 (1 + 2 beta / n))          (test vector sin(pi k/n) Q_k u)
  upper:  kappa <= ((1 + J_max)^2 + beta) * || [ |T_{k<-j}| ] ||^2   (A^-1 has blocks T_{k<-j})

so a network whose forward map preserves some direction with bounded distortion has
kappa = Theta(n^2) whatever its architecture, and bounded kappa needs the signal to vary
by a factor that grows with depth in every direction.

Arms (hidden layers only; the input and read-out layers are always the reference ones):
  reference      muP Gaussian weights + identity skip          (the reference network)
  orth           orthogonal weights, gain g x isometric gain, NO skip
  gauss          Gaussian He-scaled weights, gain g, NO skip   (non-isometric)
The spectrum is computed by analysis/kappa.py's validated routine on pcalm's own energy.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.inference import free_init, supervised_loss
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from analysis.kappa import spectrum

ISO_GAIN = {"relu": math.sqrt(2.0), "tanh": 1.0, "linear": 1.0}


def orthogonal(key, n):
    q, r = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    return q * jnp.sign(jnp.diag(r))[None, :]


def build(L, seed, arm, act, gain, width=32):
    params = list(init_params(jax.random.PRNGKey(seed), depth=L, width=width,
                              input_dim=784, output_dim=10, dtype=jnp.float32))
    scales = list(model_scales(width, L, 784))
    skips = list(skip_mask(L))
    if arm == "reference":
        return params, scales, tuple(skips)
    keys = jax.random.split(jax.random.PRNGKey(10_000 + seed), L)
    for k in range(1, L - 1):
        skips[k] = False
        if arm == "orth":
            params[k] = orthogonal(keys[k], width)
            scales[k] = gain * ISO_GAIN[act]
        elif arm == "gauss":
            params[k] = jax.random.normal(keys[k], (width, width))
            scales[k] = gain * ISO_GAIN[act] / math.sqrt(width)
        else:
            raise ValueError(arm)
    return params, scales, tuple(skips)


def layer_jacobians(params, scales, skips, x, act):
    """J_k = d pred_k / d z_{k-1} at the forward-pass point, k = 1..n-1 (n = L-1 free layers)."""
    phi = activation_fn(act)
    dphi = jax.vmap(jax.grad(lambda t: phi(t)))
    free = free_init(params, scales, skips, x, phi)
    Js = []
    for k in range(1, len(free)):
        z = free[k - 1][0]
        J = scales[k] * np.asarray(params[k]) * np.asarray(dphi(z))[None, :]
        if skips[k]:
            J = J + np.eye(J.shape[0])
        Js.append(np.asarray(J, np.float64))
    return free, Js


def output_block(params, scales, skips, x, y, free, act):
    phi = activation_fn(act)
    last = lambda zl: supervised_loss(params, scales, skips, x, y, free[:-1] + [zl[None, :]], phi)
    return np.asarray(jax.hessian(last)(free[-1][0]), np.float64)


def bounds(Js, B):
    n = len(Js) + 1
    W = B.shape[0]
    beta = max(float(np.linalg.eigvalsh(0.5 * (B + B.T))[-1]), 0.0)
    Jmax = max(float(np.linalg.norm(J, 2)) for J in Js) if Js else 0.0
    # transfer norms |T_{k<-j}|, T_{k<-k} = I
    Tn = np.zeros((n, n))
    for j in range(n):
        P = np.eye(W); Tn[j, j] = 1.0
        for k in range(j + 1, n):
            P = Js[k - 1] @ P
            Tn[k, j] = np.linalg.norm(P, 2)
    upper = ((1.0 + Jmax) ** 2 + beta) * np.linalg.norm(Tn, 2) ** 2
    # best direction for the lower bound among the right singular vectors of Q_{n-1}
    Q = [np.eye(W)]
    for J in Js:
        Q.append(J @ Q[-1])
    _, _, Vt = np.linalg.svd(Q[-1])
    best = None
    for u in list(Vt) + [np.ones(W) / math.sqrt(W)]:
        norms = np.array([np.linalg.norm(Qk @ u) for Qk in Q])
        if norms.min() <= 0:
            continue
        D = norms.max() / norms.min()
        if best is None or D < best[0]:
            best = (D, u)
    D = best[0] if best else float("inf")
    lower = n ** 2 / (math.pi ** 2 * D ** 2 * (1.0 + 2.0 * beta / n))
    return dict(n=n, beta=beta, J_max=Jmax, transfer_max=float(Tn.max()),
                distortion=float(D), kappa_lower=float(lower), kappa_upper=float(upper))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="reference:relu:1,orth:relu:1,orth:tanh:1")
    ap.add_argument("--depths", default="4,8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["arm"], r["act"], r["gain"], r["depth"], r["seed"]))
    with out.open("a") as fh:
        for spec in a.arms.split(","):
            arm, act, gain = spec.split(":"); gain = float(gain)
            phi = activation_fn(act)
            for L in [int(d) for d in a.depths.split(",")]:
                for s in [int(z) for z in a.seeds.split(",")]:
                    if (arm, act, gain, L, s) in done:
                        continue
                    t0 = time.time()
                    xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=s,
                                                data_dir="data", input_dim=784, output_dim=10)
                    x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
                    params, scales, skips = build(L, s, arm, act, gain)
                    lmax, lmin, kap, nact = spectrum(params, scales, skips, x, y, phi)
                    free, Js = layer_jacobians(params, scales, skips, x, act)
                    B = output_block(params, scales, skips, x, y, free, act)
                    bd = bounds(Js, B)
                    rms = float(jnp.sqrt(jnp.mean(free[-1] ** 2)))
                    ok = (bd["kappa_lower"] <= kap * (1 + 1e-6) <= bd["kappa_upper"] * (1 + 1e-6)
                          if np.isfinite(kap) else None)
                    row = dict(arm=arm, act=act, gain=gain, depth=L, seed=s, lambda_max=lmax,
                               lambda_min=lmin, kappa=kap, last_hidden_rms=rms,
                               bounds_hold=ok, wall_sec=round(time.time() - t0, 1), **bd)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{arm}:{act}:{gain} L={L} s={s}] kappa={kap:.1f} "
                          f"lower={bd['kappa_lower']:.1f} upper={bd['kappa_upper']:.3g} "
                          f"D={bd['distortion']:.2f} rms={rms:.2e} hold={ok}", flush=True)


if __name__ == "__main__":
    main()
