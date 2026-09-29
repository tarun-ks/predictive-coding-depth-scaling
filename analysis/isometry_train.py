"""Does a network with NO skip connection but isometric weights train at depth?

analysis/isometry_kappa.py asks whether such a network is quadratically conditioned.
This asks whether it is a network anyone would use: backpropagation test accuracy after
one epoch, with the same optimiser, learning-rate rule and data as the reference sweep.
If an orthogonal no-skip network both trains and has kappa ~ L^2, the quadratic cannot
be a property of residual connections; it is a property of signal propagation.

Training runs through analysis/generic_pc.py, whose dense path reproduces the reference
cell to 0.0000 pp (analysis/validate_generic.py).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.model import activation_fn
from pcalm.optim import adam_apply, adam_init
from pcalm.training import batch_order
from analysis.generic_pc import dense_block, forward, method_grad
from analysis.isometry_kappa import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="reference:relu:1,orth:tanh:1")
    ap.add_argument("--depths", default="8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BS, W = 64, 32
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
                    x_tr, y_tr, x_te, y_te = load_dataset("mnist", train_subset=60000,
                                                          test_subset=10000, seed=s,
                                                          data_dir="data", input_dim=784,
                                                          output_dim=10)
                    x_tr, x_te = jnp.asarray(x_tr), jnp.asarray(x_te)
                    params, scales, skips = build(L, s, arm, act, gain)
                    blocks = [dense_block] * L
                    lr = 1e-3 * math.sqrt(W / L)
                    opt = adam_init(params)

                    @jax.jit
                    def update(p, o, xb, yb):
                        g = method_grad(p, scales, skips, blocks, xb, yb, "bp", state_lr=1.0,
                                        rho=1.0, alpha=1.0, budget=1, inner_steps=1, phi=phi)
                        return adam_apply(p, g, o, lr)

                    for idx in batch_order(x_tr.shape[0], BS, s, True):
                        params, opt = update(params, opt, x_tr[idx], jnp.asarray(y_tr[idx]))
                    tot, n = 0.0, 0
                    for i in range(0, x_te.shape[0], BS):
                        e = min(i + BS, x_te.shape[0])
                        lg = forward(params, scales, skips, blocks, x_te[i:e], phi)[-1]
                        _, _, acc = mse_ce_accuracy(lg, jnp.asarray(y_te[i:e]))
                        tot += float(acc) * (e - i); n += e - i
                    row = dict(arm=arm, act=act, gain=gain, depth=L, seed=s,
                               bp_test_acc=tot / n, wall_sec=round(time.time() - t0, 1))
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{arm}:{act}:{gain} L={L} s={s}] BP={100*tot/n:.2f}% "
                          f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
