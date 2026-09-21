"""ITEM 2: is the localized per-layer rotation CAUSAL?

Freeze the duals in one contiguous half of the layers and measure the post-knee
accuracy decay. The output-half arm is the control: without it, "freezing any
duals helps" would be an unexcluded explanation.

Built on analysis/generic_pc.py, which reproduces the reference cell to 0.0000 pp
with freeze=None (analysis/validate_generic.py), so the only difference between
arms is which duals update.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import adam_learning_rate, batch_order
from analysis.generic_pc import dense_block, forward, method_grad
from analysis.eta_table import eta_for_depths

KNEE = {64: 96, 128: 256}
MULTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# 8x and 16x added after the first pass: freezing duals slows the method, moving
# its knee outward, so every intervention arm peaked at the 4x window edge
# (argmax = 4.00) and reported drop = 0.00 purely because the peak lay outside
# the window. The drop is only interpretable once argmax is INTERIOR.


def freeze_mask(L, arm):
    n = L - 1                      # hidden constraints
    half = n // 2
    if arm == "none":   return None
    if arm == "input":  return [i < half for i in range(n)]
    if arm == "output": return [i >= n - half for i in range(n)]
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["none", "input", "output"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L, W, BS = a.depth, 32, 64
    slr = eta_for_depths([L])[0][L]
    phi = activation_fn("relu")
    sc, sk = model_scales(W, L, 784), skip_mask(L)
    blocks = [dense_block] * L
    lr = adam_learning_rate(W, L, 1e-3, 1.0, None)
    fz = freeze_mask(L, a.arm)
    xtr, ytr, xte, yte = load_dataset("mnist", train_subset=60000, test_subset=10000,
                                      seed=a.seed, data_dir=a.data_dir,
                                      input_dim=784, output_dim=10)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["budget"])
    with out.open("a") as fh:
        for m in MULTS:
            T = max(1, int(round(KNEE[L] * m)))
            if T in done: continue
            t0 = time.time()
            params = init_params(jax.random.PRNGKey(a.seed), depth=L, width=W,
                                 input_dim=784, output_dim=10, dtype=jnp.float32)
            opt = adam_init(params)

            @jax.jit
            def upd(p, o, xb, yb):
                g = method_grad(p, sc, sk, blocks, xb, yb, "pcalm", state_lr=slr,
                                rho=1.0, alpha=1.0, budget=T, inner_steps=1,
                                phi=phi, freeze=fz)
                return adam_apply(p, g, o, lr)

            for idx in batch_order(xtr.shape[0], BS, a.seed, True):
                params, opt = upd(params, opt, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]))
            tot, n = 0.0, 0
            for s0 in range(0, xte.shape[0], BS):
                e = min(s0 + BS, xte.shape[0])
                lg = forward(params, sc, sk, blocks, jnp.asarray(xte[s0:e]), phi)[-1]
                _, _, acc = mse_ce_accuracy(lg, jnp.asarray(yte[s0:e]))
                tot += float(acc) * (e - s0); n += e - s0
            row = dict(depth=L, seed=a.seed, arm=a.arm, budget=T, mult_of_knee=m,
                       n_frozen=(0 if fz is None else int(sum(fz))), n_hidden=L - 1,
                       state_lr=slr, test_acc=tot / n,
                       finite=bool(np.isfinite(tot / n)),
                       wall_sec=round(time.time() - t0, 2))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[{a.arm} L={L} s={a.seed}] T={T} ({m}x knee) acc={100*tot/n:.2f}%", flush=True)


if __name__ == "__main__":
    main()
