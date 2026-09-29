"""Is the residual connection what creates the quadratic conditioning?

analysis/kappa_family.py measures that the PC activity Hessian is quadratically
conditioned WITH the residual connection and becomes better conditioned with
depth WITHOUT it. That is a statement about the inner problem only. This script
asks the question that decides what it means: can the no-skip network be
trained at all at depth?

Everything is held to the reference dense configuration -- same scales, same
initialisation, same activation, same optimiser, same data -- and the ONLY
change is skips = (False,)*L instead of pcalm.model.skip_mask(L). The training
path is analysis/generic_pc.py, which reproduces the reference dense cell to
0.0000 pp (analysis/validate_generic.py).

eta_h uses the analytic rule eta_h = 1/lambda_max with lambda_max from power
iteration, the same estimator analysis/run_conv.py uses, applied identically at
every depth so the depth comparison is internally consistent.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.model import init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import batch_order
from analysis.generic_pc import (al_energy_shifted, constraint_residuals, dense_block,
                                 forward, free_init, method_grad, zero_duals_like)

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
          768, 1024, 1536, 2048]


def build(depth, seed, width, skips_on):
    blocks = [dense_block] * depth
    skips = skip_mask(depth) if skips_on else tuple([False] * depth)
    scales = model_scales(width, depth, 784)
    params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width,
                         input_dim=784, output_dim=10, dtype=jnp.float32)
    return list(params), scales, skips, blocks


def lambda_max(params, scales, skips, blocks, x, y, phi, iters=200):
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    gf = jax.grad(lambda f: al_energy_shifted(params, scales, skips, blocks, x, y, f,
                                              d0, 1.0, phi))
    hvp = jax.jit(lambda v: jax.jvp(gf, (f0,), (v,))[1])
    ks = jax.random.split(jax.random.PRNGKey(12345), len(f0))
    v = [jax.random.normal(k, z.shape) for k, z in zip(ks, f0)]
    nrm = lambda a: float(jnp.sqrt(sum(jnp.sum(t * t) for t in a)))
    n = nrm(v); v = [t / n for t in v]
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = float(sum(jnp.sum(a * b) for a, b in zip(v, w)))
        nw = nrm(w)
        if nw == 0: break
        v = [t / nw for t in w]
    return lam * x.shape[0]


def forward_scale(params, scales, skips, blocks, x, phi):
    """RMS activation of the last hidden layer: the vanishing-forward-pass check."""
    acts = forward(params, scales, skips, blocks, x, phi)
    return [float(jnp.sqrt(jnp.mean(a ** 2))) for a in acts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--method", default="bp")
    ap.add_argument("--skips", choices=["on", "off"], default="off")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--train-subset", type=int, default=60000)
    ap.add_argument("--max-budget-mult", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BS, phi = 64, jax.nn.relu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["depth"], r["seed"], r["method"], r["budget"]))

    with out.open("a") as fh:
        for L in [int(d) for d in a.depths.split(",")]:
            for s in [int(z) for z in a.seeds.split(",")]:
                x_tr, y_tr, x_te, y_te = load_dataset(
                    "mnist", train_subset=a.train_subset, test_subset=10000, seed=s,
                    data_dir="data", input_dim=784, output_dim=10)
                x_tr, x_te = jnp.asarray(x_tr), jnp.asarray(x_te)
                params0, scales, skips, blocks = build(L, s, a.width, a.skips == "on")
                xb0, yb0 = x_tr[:BS], jnp.asarray(y_tr[:BS])
                lam = lambda_max(params0, scales, skips, blocks, xb0, yb0, phi)
                slr = 1.0 / lam
                rms = forward_scale(params0, scales, skips, blocks, xb0, phi)
                lr = 1e-3 * math.sqrt(a.width / L)
                budgets = [0] if a.method == "bp" else [
                    t for t in LADDER if t <= a.max_budget_mult * L]
                for T in budgets:
                    key = (L, s, a.method, T)
                    if key in done: continue
                    t0 = time.time()
                    params = [p for p in params0]; opt = adam_init(params)

                    @jax.jit
                    def update(p, o, xb, yb):
                        g = method_grad(p, scales, skips, blocks, xb, yb, a.method,
                                        state_lr=slr, rho=1.0, alpha=1.0,
                                        budget=max(T, 1), inner_steps=1, phi=phi)
                        return adam_apply(p, g, o, lr)

                    for idx in batch_order(x_tr.shape[0], BS, s, True):
                        params, opt = update(params, opt, x_tr[idx], jnp.asarray(y_tr[idx]))
                    tot, n = 0.0, 0
                    for i in range(0, x_te.shape[0], BS):
                        e = min(i + BS, x_te.shape[0])
                        lg = forward(params, scales, skips, blocks, x_te[i:e], phi)[-1]
                        _, _, acc = mse_ce_accuracy(lg, jnp.asarray(y_te[i:e]))
                        tot += float(acc) * (e - i); n += e - i
                    row = dict(arch="dense", skips=a.skips, depth=L, seed=s,
                               method=a.method, budget=T, width=a.width,
                               state_lr=slr, lambda_max=lam,
                               act_rms_last=rms[-2] if len(rms) > 1 else rms[-1],
                               act_rms_first=rms[0], learning_rate=lr,
                               test_acc=tot / n, finite=bool(np.isfinite(tot / n)),
                               wall_sec=round(time.time() - t0, 2))
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{a.skips} {a.method} L={L} s={s}] T={T} "
                          f"acc={100*tot/n:.2f}% rms_last={row['act_rms_last']:.3e} "
                          f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
