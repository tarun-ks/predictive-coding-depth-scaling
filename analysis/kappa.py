"""Condition number of the PC activity Hessian versus depth.

Route 1 of our derivation asserts lambda_min = Theta(L^-2) from the
first-difference/Laplacian structure, giving kappa = Theta(L^2). That is an
argument; this measures it. Requires no training -- the spectrum is evaluated at
initialisation, one sample at a time (the Hessian is block-diagonal over the
batch, so a batch operator would report max-over-samples / min-over-samples and
inflate kappa).

lambda_max: power iteration on H.
lambda_min: power iteration on the shifted operator (lambda_max I - H), whose
            largest eigenvalue is lambda_max - lambda_min.
Both validated against a dense eigendecomposition on a small network.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
if os.environ.get("KAPPA_X64") == "1":
    # The activity Hessian is a mathematical object; float32 is only our
    # arithmetic. At large depth lambda_min falls to float32's resolution
    # relative to lambda_max and ARPACK stops converging. Evaluating the same
    # operator in float64 estimates the same spectrum more accurately -- it does
    # not change the object. Validated by agreement with float32 where both work.
    import jax
    jax.config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import al_energy_shifted, constraint_residuals, free_init, zero_duals_like
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def _flat(a): return jnp.concatenate([x.ravel() for x in a])
def _unflat(v, like):
    out, i = [], 0
    for x in like:
        n = x.size
        out.append(v[i:i + n].reshape(x.shape)); i += n
    return out


def hessian_ops(params, scales, skips, x, y, phi, rho=1.0):
    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    energy = lambda f: al_energy_shifted(params, scales, skips, x, y, f, duals0, rho, phi)
    gradf = jax.grad(energy)
    hvp = jax.jit(lambda v: _flat(jax.jvp(gradf, (free0,), (_unflat(v, free0),))[1]))
    n = int(sum(z.size for z in free0))
    return hvp, n, free0


def power(hvp, n, seed=0, iters=4000, tol=1e-12):
    v = jax.random.normal(jax.random.PRNGKey(seed), (n,))
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        new = float(jnp.dot(v, w))
        nw = float(jnp.linalg.norm(w))
        if nw == 0: break
        v = w / nw
        if abs(new - lam) < tol * max(abs(new), 1.0): lam = new; break
        lam = new
    return lam


def spectrum(params, scales, skips, x, y, phi, iters=4000):
    """lambda_max by power iteration (accurate to ~1e-6), lambda_min by Lanczos
    on the shifted operator. Shifted POWER iteration was tried first and gave up
    to 9.7% error on lambda_min because the shifted spectrum is dense near its
    maximum; Lanczos does not have that failure mode."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    hvp, n, _ = hessian_ops(params, scales, skips, x, y, phi)
    lmax = power(hvp, n, iters=iters)
    _dt = jnp.float64 if os.environ.get("KAPPA_X64") == "1" else jnp.float32
    mv = lambda v: np.asarray(hvp(jnp.asarray(v, _dt)), np.float64)
    op = LinearOperator((n, n), matvec=lambda v: lmax * np.asarray(v).ravel() - mv(v),
                        dtype=np.float64)
    # ncv (Krylov basis size) is the parameter that matters here, NOT precision:
    # the shifted operator's spectrum is dense near its maximum, so the default
    # basis is too small to separate the target. float64 made no difference;
    # ncv=200 converges, and agrees with LOBPCG to 0.1% at L=256.
    k = min(3, max(1, n - 2))
    ncv = min(n - 1, 200)
    try:
        mu = float(eigsh(op, k=k, ncv=ncv, which="LA", return_eigenvectors=False,
                         tol=1e-9, maxiter=50000).max())
    except Exception:
        # ARPACK cannot resolve lambda_min once kappa is large enough that
        # lambda_min sits at the edge of float precision relative to lambda_max.
        # Report as non-convergent rather than emitting a fabricated number.
        return lmax, float("nan"), float("nan"), n
    lmin = lmax - mu
    return lmax, lmin, (lmax / lmin if lmin > 0 else float("inf")), n


def sp_model(width, depth, input_dim, seed, output_dim=10):
    """Standard parameterization (muPC Table 1, 'Standard PC' column):
    unit forward multipliers a=1, init variance b = 1/fan_in.
    Architecture (skips) unchanged -- only scaling and init differ."""
    scales = [1.0] * depth
    keys = jax.random.split(jax.random.PRNGKey(seed), depth)
    params = []
    for i in range(depth):
        fin = input_dim if i == 0 else width
        fout = output_dim if i == depth - 1 else width
        params.append(jax.random.normal(keys[i], (fout, fin)) / np.sqrt(fin))
    return params, scales


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--depths", default="4,8,16,32,64,128,256")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--samples", type=int, default=1,
                    help="input samples per seed. The headline run used 1; >1 separates "
                         "input variation from initialization variation.")
    ap.add_argument("--param", choices=["mup", "sp"], default="mup")
    a = ap.parse_args()
    phi = activation_fn("relu")

    if a.validate:
        print("VALIDATION: iterative vs dense eigendecomposition (small net)")
        for (L, W_, D) in [(4, 6, 5), (6, 5, 4), (8, 4, 3)]:
            params = init_params(jax.random.PRNGKey(0), depth=L, width=W_,
                                 input_dim=D, output_dim=3, dtype=jnp.float32)
            sc, sk = model_scales(W_, L, D), skip_mask(L)
            x = jax.random.normal(jax.random.PRNGKey(1), (1, D))
            y = jax.nn.one_hot(jnp.array([1]), 3)
            hvp, n, free0 = hessian_ops(params, sc, sk, x, y, phi)
            H = np.stack([np.asarray(hvp(jnp.eye(n)[i])) for i in range(n)])
            H = 0.5 * (H + H.T)
            ev = np.linalg.eigvalsh(H)
            lmax, lmin, k, _ = spectrum(params, sc, sk, x, y, phi, iters=8000)
            print(f"  L={L} W={W_} n={n}: dense lmax={ev[-1]:.6f} lmin={ev[0]:.6f} "
                  f"kappa={ev[-1]/ev[0]:.2f}")
            print(f"           iterative lmax={lmax:.6f} lmin={lmin:.6f} kappa={k:.2f}  "
                  f"-> rel err lmax {abs(lmax-ev[-1])/ev[-1]:.2e}, lmin {abs(lmin-ev[0])/ev[0]:.2e}")
        sys.exit()

    from pcalm.data import load_dataset
    depths = [int(d) for d in a.depths.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    print("depth,seed,sample,lambda_max,lambda_min,kappa,n_activities")
    for L in depths:
        sk = skip_mask(L)
        for s in seeds:
            xt, yt, _, _ = load_dataset("mnist", train_subset=max(8, a.samples),
                                        test_subset=8, seed=s,
                                        data_dir="data", input_dim=784, output_dim=10)
            # The network depends on the seed only; the input varies within a seed,
            # so seed-to-seed spread mixes both while sample-to-sample isolates input.
            if a.param == "mup":
                # Always DRAW in float32 so the network is identical regardless of
                # arithmetic precision (jax.random.normal returns different values
                # for the same key at different dtypes), then cast.
                sc = model_scales(a.width, L, 784)
                params = init_params(jax.random.PRNGKey(s), depth=L, width=a.width,
                                     input_dim=784, output_dim=10, dtype=jnp.float32)
                if os.environ.get("KAPPA_X64") == "1":
                    params = [jnp.asarray(w, jnp.float64) for w in params]
            else:
                params, sc = sp_model(a.width, L, 784, s)
            for smp in range(a.samples):
                x, y = jnp.asarray(xt[smp:smp + 1]), jnp.asarray(yt[smp:smp + 1])
                if a.param == "mup" and os.environ.get("KAPPA_X64") == "1":
                    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
                lmax, lmin, k, n = spectrum(params, sc, sk, x, y, phi)
                print(f"{L},{s},{smp},{lmax:.6f},{lmin:.8f},{k:.3f},{n}", flush=True)
