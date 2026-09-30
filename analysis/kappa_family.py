"""Activity-Hessian conditioning across three layer-local energies.

The predictive-coding energy has kappa ~ L^2 (analysis/kappa.py). That energy is
a sum of squared layerwise residuals, so its Hessian carries the A^T A structure
of a discrete derivative. Two questions follow, and neither is answered by the
PC measurement alone:

  1. Is the residual (skip) connection load-bearing? The first-difference
     structure of the constraint operator comes FROM the skip. Without skips the constraint
     operator is block-bidiagonal rather than a difference operator.
  2. Is quadratic conditioning a property of the PC energy specifically, or of
     chain-coupled layer-local inference generally? The contrastive-Hebbian /
     equilibrium-propagation family uses a Hopfield energy, whose Hessian is
     I - (C + C^T) rather than A^T A. Same chain, same architecture, same
     initialisation, different functional.

Everything here is evaluated at initialisation on the feedforward state, so no
training is involved and the comparison is on one fixed network per seed.
pcalm/ is never modified: this module is an external caller, and the 'pc'
family reproduces analysis/kappa.py as a self-check.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get("KAPPA_X64") == "1":
    import jax
    jax.config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import (al_energy_shifted, constraint_residuals, free_init,
                             supervised_loss, zero_duals_like)
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from analysis.kappa import power


def _flat(a): return jnp.concatenate([x.ravel() for x in a])
def _unflat(v, like):
    out, i = [], 0
    for x in like:
        n = x.size
        out.append(v[i:i + n].reshape(x.shape)); i += n
    return out


def hopfield_energy(params, scales, skips, x, y, free, phi, beta=1.0):
    """Hopfield / contrastive-Hebbian energy on the same layer chain.

        E(z) = sum_i 0.5||z_i||^2 - sum_i <z_i, block_i(z_{i-1})>
               + beta * supervised loss at the output

    `free` holds the hidden activities only (length L-1), exactly as in
    pcalm.inference.free_init, and the final block maps free[-1] to the output,
    exactly as in pcalm.inference.supervised_loss. The block predictor, its
    scale and the skip structure are therefore identical to the PC network; the
    only thing that differs is the functional. beta = 1 matches the weight PC's
    energy gives the supervised term. beta enters only the last hidden block's
    diagonal and cannot change the depth scaling of the chain coupling.
    """
    from pcalm.model import block_pred
    batch = x.shape[0]
    total = 0.5 * sum(jnp.sum(z * z) for z in free)
    for i, z_l in enumerate(free):
        z_prev = x if i == 0 else free[i - 1]
        pred = block_pred(params[i], scales[i], skips[i], z_prev, phi,
                          is_first=(i == 0))
        total = total - jnp.sum(z_l * pred)
    total = total / batch
    if beta:
        total = total + beta * supervised_loss(params, scales, skips, x, y, free, phi)
    return total


def pc_energy(params, scales, skips, x, y, free, phi):
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi))
    return al_energy_shifted(params, scales, skips, x, y, free, duals0, 1.0, phi)


def hvp_for(family, params, scales, skips, x, y, phi, beta):
    free0 = free_init(params, scales, skips, x, phi)
    if family == "hopfield":
        energy = lambda f: hopfield_energy(params, scales, skips, x, y, f, phi, beta)
    else:
        energy = lambda f: pc_energy(params, scales, skips, x, y, f, phi)
    gradf = jax.grad(energy)
    hvp = jax.jit(lambda v: _flat(jax.jvp(gradf, (free0,), (_unflat(v, free0),))[1]))
    n = int(sum(z.size for z in free0))
    return hvp, n


def spectrum_of(hvp, n, iters=4000):
    """lambda_max by power iteration; lambda_min by Lanczos on (lmax I - H).

    Identical procedure to analysis/kappa.py. The Hopfield Hessian is NOT
    guaranteed positive definite -- I - (C + C^T) is indefinite once the
    coupling is strong enough -- so we also report the algebraically smallest
    eigenvalue directly and flag sign changes rather than reporting a condition
    number for an indefinite operator.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh
    _dt = jnp.float64 if os.environ.get("KAPPA_X64") == "1" else jnp.float32
    mv = lambda v: np.asarray(hvp(jnp.asarray(np.asarray(v).ravel(), _dt)), np.float64)
    lmax_abs = power(hvp, n, iters=iters)
    op = LinearOperator((n, n), matvec=lambda v: lmax_abs * np.asarray(v).ravel() - mv(v),
                        dtype=np.float64)
    k, ncv = min(3, max(1, n - 2)), min(n - 1, 200)
    try:
        mu = float(eigsh(op, k=k, ncv=ncv, which="LA", return_eigenvectors=False,
                         tol=1e-9, maxiter=50000).max())
        lmin = lmax_abs - mu
    except Exception:
        return lmax_abs, float("nan"), float("nan"), n
    kappa = (lmax_abs / lmin) if lmin > 0 else float("nan")
    return lmax_abs, lmin, kappa, n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pc,pc_noskip,hopfield")
    ap.add_argument("--depths", default="4,8,16,32,64,128")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    phi = activation_fn("relu")
    from pcalm.data import load_dataset

    print("family,depth,seed,lambda_max,lambda_min,kappa,n_activities")
    for fam in a.families.split(","):
        for L in [int(d) for d in a.depths.split(",")]:
            # pc_noskip removes the residual connection; everything else is held.
            # "hopfield" carries the residual connection INTO the Hopfield energy as a
            # unit-strength coupling between adjacent layers, which on its own gives a
            # spectrum 1 - 2 cos(theta), i.e. indefinite. "hopfield_noskip" is the form
            # equilibrium propagation uses, with inter-layer coupling through the weights only.
            # On the reference weights that coupling shrinks as L^-1/2 and the forward signal
            # dies, so "*_orth_linear" / "*_orth_tanh" repeat both energies on the no-skip
            # orthogonal networks of isometry_kappa.py, which carry a signal and train.
            sk = tuple([False] * L) if fam in ("pc_noskip", "hopfield_noskip") else skip_mask(L)
            base = "hopfield" if fam.startswith("hopfield") else "pc"
            orth_act = fam.rsplit("_", 1)[1] if "_orth_" in fam else None
            for s in [int(z) for z in a.seeds.split(",")]:
                xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8,
                                            seed=s, data_dir="data",
                                            input_dim=784, output_dim=10)
                sc = model_scales(a.width, L, 784)
                params = init_params(jax.random.PRNGKey(s), depth=L, width=a.width,
                                     input_dim=784, output_dim=10, dtype=jnp.float32)
                if os.environ.get("KAPPA_X64") == "1":
                    params = [jnp.asarray(w, jnp.float64) for w in params]
                x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
                if os.environ.get("KAPPA_X64") == "1":
                    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
                phi_f = phi
                if orth_act is not None:
                    from analysis.isometry_kappa import build as iso_build
                    params, sc, sk = iso_build(L, s, "orth", orth_act, 1.0, width=a.width)
                    phi_f = activation_fn(orth_act)
                hvp, n = hvp_for(base, params, sc, sk, x, y, phi_f, a.beta)
                lmax, lmin, k, n = spectrum_of(hvp, n)
                print(f"{fam},{L},{s},{lmax:.6f},{lmin:.10f},{k:.3f},{n}", flush=True)
