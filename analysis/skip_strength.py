"""The conditioning-stability trade-off, traced continuously in skip strength.

analysis/kappa_family.py shows the PC activity Hessian is quadratically
conditioned with the residual connection and becomes BETTER conditioned with
depth without it; analysis/run_noskip.py shows the no-skip network is at chance
past L=8 because its forward pass vanishes. Those are the two endpoints. This
script interpolates between them.

The block predictor becomes

    z_i = s_i * block_i(z_{i-1}) + c * z_{i-1}

with c = 1 the reference residual network and c = 0 the plain chain. For each c
we measure, on the SAME initialised network:

  * kappa of the activity Hessian at initialisation (no training), and
  * the RMS activation of the last hidden layer (forward-pass survival), and
  * backpropagation test accuracy after one epoch (is the network usable).

Everything else is held to the reference dense configuration. The spectra and
the training both run through analysis/generic_pc.py, whose dense path
reproduces the reference cell to 0.0000 pp (analysis/validate_generic.py).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get("KAPPA_X64") == "1":
    import jax
    jax.config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.model import init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import batch_order
from analysis.generic_pc import (al_energy_shifted, constraint_residuals, forward,
                                 free_init, method_grad, zero_duals_like)
from analysis.kappa import power


def scaled_skip_block(c):
    """Dense block with a residual connection of strength c (c=1 is the reference)."""
    def blk(W, scale, skip, z_prev, phi, is_first):
        inp = z_prev if is_first else phi(z_prev)
        pred = scale * (inp @ W.T)
        return pred + c * z_prev if skip else pred
    return blk


def build(depth, seed, width, c):
    blocks = [scaled_skip_block(c)] * depth
    skips = skip_mask(depth)          # WHERE a skip may occur: unchanged from reference
    scales = model_scales(width, depth, 784)
    params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width,
                         input_dim=784, output_dim=10, dtype=jnp.float32)
    return list(params), scales, skips, blocks


def _flat(a): return jnp.concatenate([x.ravel() for x in a])
def _unflat(v, like):
    out, i = [], 0
    for x in like:
        n = x.size
        out.append(v[i:i + n].reshape(x.shape)); i += n
    return out


def spectrum(params, scales, skips, blocks, x, y, phi):
    from scipy.sparse.linalg import LinearOperator, eigsh
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    gf = jax.grad(lambda f: al_energy_shifted(params, scales, skips, blocks, x, y, f,
                                              d0, 1.0, phi))
    hvp = jax.jit(lambda v: _flat(jax.jvp(gf, (f0,), (_unflat(v, f0),))[1]))
    n = int(sum(z.size for z in f0))
    lmax = power(hvp, n, iters=4000)
    _dt = jnp.float64 if os.environ.get("KAPPA_X64") == "1" else jnp.float32
    mv = lambda v: np.asarray(hvp(jnp.asarray(np.asarray(v).ravel(), _dt)), np.float64)
    op = LinearOperator((n, n), matvec=lambda v: lmax * np.asarray(v).ravel() - mv(v),
                        dtype=np.float64)
    try:
        mu = float(eigsh(op, k=min(3, max(1, n - 2)), ncv=min(n - 1, 200), which="LA",
                         return_eigenvectors=False, tol=1e-9, maxiter=50000).max())
        lmin = lmax - mu
    except Exception:
        return lmax, float("nan"), float("nan"), n
    return lmax, lmin, (lmax / lmin if lmin > 0 else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--train-subset", type=int, default=60000)
    ap.add_argument("--no-train", action="store_true",
                    help="spectra and forward RMS only; skip the BP epoch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BS, phi = 64, jax.nn.relu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["depth"], r["seed"], r["skip_strength"]))

    with out.open("a") as fh:
        for c in [float(z) for z in a.strengths.split(",")]:
            for L in [int(d) for d in a.depths.split(",")]:
                for s in [int(z) for z in a.seeds.split(",")]:
                    if (L, s, c) in done: continue
                    t0 = time.time()
                    x_tr, y_tr, x_te, y_te = load_dataset(
                        "mnist", train_subset=a.train_subset, test_subset=10000,
                        seed=s, data_dir="data", input_dim=784, output_dim=10)
                    x_tr, x_te = jnp.asarray(x_tr), jnp.asarray(x_te)
                    params0, scales, skips, blocks = build(L, s, a.width, c)
                    xb0, yb0 = x_tr[:BS], jnp.asarray(y_tr[:BS])
                    lmax, lmin, kap, n = spectrum(params0, scales, skips, blocks,
                                                  xb0[:1], yb0[:1], phi)
                    acts = forward(params0, scales, skips, blocks, xb0, phi)
                    rms = [float(jnp.sqrt(jnp.mean(z ** 2))) for z in acts]
                    acc = float("nan")
                    if not a.no_train:
                        lr = 1e-3 * math.sqrt(a.width / L)
                        params = [p for p in params0]; opt = adam_init(params)

                        @jax.jit
                        def update(p, o, xb, yb):
                            g = method_grad(p, scales, skips, blocks, xb, yb, "bp",
                                            state_lr=1.0, rho=1.0, alpha=1.0, budget=1,
                                            inner_steps=1, phi=phi)
                            return adam_apply(p, g, o, lr)

                        for idx in batch_order(x_tr.shape[0], BS, s, True):
                            params, opt = update(params, opt, x_tr[idx],
                                                 jnp.asarray(y_tr[idx]))
                        tot, m = 0.0, 0
                        for i in range(0, x_te.shape[0], BS):
                            e = min(i + BS, x_te.shape[0])
                            lg = forward(params, scales, skips, blocks, x_te[i:e], phi)[-1]
                            _, _, ac = mse_ce_accuracy(lg, jnp.asarray(y_te[i:e]))
                            tot += float(ac) * (e - i); m += e - i
                        acc = tot / m
                    row = dict(depth=L, seed=s, skip_strength=c, width=a.width,
                               lambda_max=lmax, lambda_min=lmin, kappa=kap,
                               n_activities=n, act_rms_last=rms[-2] if len(rms) > 1 else rms[-1],
                               act_rms_first=rms[0], bp_test_acc=acc,
                               wall_sec=round(time.time() - t0, 2))
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[c={c} L={L} s={s}] kappa={kap:.1f} rms={row['act_rms_last']:.3e} "
                          f"bp={100*acc:.2f}% ({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
