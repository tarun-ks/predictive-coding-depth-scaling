"""Why geometric multigrid does not transfer: is the slow mode smooth along depth?

Geometric multigrid works when the error the smoother cannot remove is SMOOTH on the
grid, so that linear interpolation from a coarser grid can represent it. On this
problem the grid is the layer index, and the question is whether the eigenvector of
the activity Hessian belonging to lambda_min is smooth in that index.

It has two parts that can behave differently:
  * the ENVELOPE, the per-layer norm ||v_i||, and
  * the DIRECTION within each layer, v_i / ||v_i||.
A diffusion operator on a line has both smooth. Here each layer applies its own random
weight matrix, so the envelope can be smooth while consecutive directions are nearly
uncorrelated -- and linear interpolation, which assumes v_{2j+1} ~ (v_{2j}+v_{2j+2})/2,
cannot represent that. We measure both.

  envelope_roughness  = || D ||v_i|| || / || ||v_i|| ||   (D = first difference)
  direction_alignment = mean_i  <v_i, v_{i+1}> / (||v_i|| ||v_{i+1}||)

A smooth mode has small envelope roughness AND alignment near 1. Alignment near 0
means consecutive layers point in unrelated directions, which is the obstruction.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.inference import (al_energy_shifted, constraint_residuals, free_init,
                             zero_duals_like)


def hvp_ops(params, sc, sk, x, y, phi):
    f0 = free_init(params, sc, sk, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, sc, sk, x, f0, phi))
    E = lambda f: al_energy_shifted(params, sc, sk, x, y, f, d0, 1.0, phi)
    g = jax.grad(E)
    flat = lambda a: jnp.concatenate([z.ravel() for z in a])
    def unflat(v):
        out, i = [], 0
        for z in f0:
            n = z.size; out.append(v[i:i+n].reshape(z.shape)); i += n
        return out
    hvp = jax.jit(lambda v: flat(jax.jvp(g, (f0,), (unflat(v),))[1]))
    return hvp, int(sum(z.size for z in f0)), len(f0), f0[0].shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from scipy.sparse.linalg import LinearOperator, eigsh
    phi = activation_fn("relu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for L in [int(d) for d in a.depths.split(",")]:
            for s in [int(z) for z in a.seeds.split(",")]:
                xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8,
                                            seed=s, data_dir="data", input_dim=784,
                                            output_dim=10)
                x, y = jnp.asarray(xt[:1]), jnp.asarray(yt[:1])
                params = init_params(jax.random.PRNGKey(s), depth=L, width=a.width,
                                     input_dim=784, output_dim=10, dtype=jnp.float32)
                sc, sk = model_scales(a.width, L, 784), skip_mask(L)
                hvp, n, nlayers, shp = hvp_ops(params, sc, sk, x, y, phi)
                mv = lambda v: np.asarray(hvp(jnp.asarray(np.asarray(v).ravel(),
                                                          jnp.float32)), np.float64)
                op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
                # smallest algebraic eigenpair via shift: lmax I - H, largest
                lmax = float(eigsh(op, k=1, which="LA", ncv=min(n-1, 200),
                                   return_eigenvectors=False, tol=1e-8)[0])
                op2 = LinearOperator((n, n),
                                     matvec=lambda v: lmax*np.asarray(v).ravel()-mv(v),
                                     dtype=np.float64)
                w, V = eigsh(op2, k=1, which="LA", ncv=min(n-1, 200), tol=1e-9)
                v = V[:, 0].reshape(nlayers, -1)                    # (layers, width)
                env = np.linalg.norm(v, axis=1)
                env_rough = (np.linalg.norm(np.diff(env)) /
                             max(np.linalg.norm(env), 1e-30))
                vn = v / np.maximum(env[:, None], 1e-30)
                align = float(np.mean(np.sum(vn[:-1] * vn[1:], axis=1)))
                # what linear interpolation from a 2x coarse grid would recover
                approx = v.copy()
                approx[1::2] = 0.5 * (v[0:-1:2] + v[2::2]) if nlayers > 2 else v[1::2]
                interp_err = (np.linalg.norm(approx - v) /
                              max(np.linalg.norm(v), 1e-30))
                row = dict(depth=L, seed=s, width=a.width, lambda_min=float(lmax-w[0]),
                           envelope_roughness=float(env_rough),
                           direction_alignment=align,
                           linear_interp_rel_error=float(interp_err))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[L={L} s={s}] env_rough={env_rough:.4f} align={align:+.4f} "
                      f"interp_err={interp_err:.4f}", flush=True)


if __name__ == "__main__":
    main()
