"""VALIDATION CONTRACT for analysis/generic_pc.py.

Trains with the generic module instantiated with DENSE blocks, using the
reference package's own init, scales, skip mask, batch order, optimiser and
eval. If generic_pc is a faithful re-expression, this must reproduce the
published cell to the same precision as the reference: 78.66 / 68.13 / 77.75.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.metrics import mse_ce_accuracy
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import adam_learning_rate, batch_order
from analysis.generic_pc import dense_block, forward, method_grad

L, W_, SEED, BS, T = 32, 32, 0, 64, 64
SLR = 0.23588549900873967

phi = activation_fn("relu")
scales = model_scales(W_, L, 784)
skips = skip_mask(L)
blocks = [dense_block] * L
lr = adam_learning_rate(W_, L, 1e-3, 1.0, None)
x_tr, y_tr, x_te, y_te = load_dataset("fashion_mnist", train_subset=60000,
                                      test_subset=10000, seed=SEED, data_dir="data",
                                      input_dim=784, output_dim=10)

print("generic_pc.py with DENSE blocks vs published reference cell")
print("  method   generic   published   delta")
for fam, ref in (("bp", 78.66), ("pc", 68.13), ("pcalm", 77.75)):
    params = init_params(jax.random.PRNGKey(SEED), depth=L, width=W_,
                         input_dim=784, output_dim=10, dtype=jnp.float32)
    opt = adam_init(params)

    @jax.jit
    def update(p, o, xb, yb):
        g = method_grad(p, scales, skips, blocks, xb, yb, fam, state_lr=SLR, rho=1.0,
                        alpha=1.0, budget=T, inner_steps=1, phi=phi)
        return adam_apply(p, g, o, lr)

    for idx in batch_order(x_tr.shape[0], BS, SEED, True):
        params, opt = update(params, opt, jnp.asarray(x_tr[idx]), jnp.asarray(y_tr[idx]))

    tot, n = 0.0, 0
    for s in range(0, x_te.shape[0], BS):
        e = min(s + BS, x_te.shape[0])
        lg = forward(params, scales, skips, blocks, jnp.asarray(x_te[s:e]), phi)[-1]
        _, _, acc = mse_ce_accuracy(lg, jnp.asarray(y_te[s:e]))
        tot += float(acc) * (e - s); n += e - s
    got = 100 * tot / n
    flag = "OK" if abs(got - ref) < 0.005 else "*** MISMATCH ***"
    print(f"  {fam:6s}  {got:7.2f}%   {ref:7.2f}%   {got-ref:+.4f}  {flag}")
