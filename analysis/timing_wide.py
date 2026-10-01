"""Wall-clock cost of one training step at width 128, for each inner solver.

Each arm is timed with the same code path it trained with: plain PC and PC-ALM through the
reference implementation's update function, Nesterov through analysis/momentum_pc.py. A
step costs a fixed part (the forward pass, the weight gradient, Adam) plus T inference
iterations, so each depth is timed at two budgets and the difference gives the cost per
inference iteration. Run on an otherwise idle machine; parallel load distorts it.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.inference import Schedule
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import adam_learning_rate, make_update_fn
from analysis.eta_table import eta_for_depths
from analysis.generic_pc import dense_block
from analysis.momentum_pc import beta_nag, method_grad_mom


def step_fn(method, L, W, slr, lr, scales, skips, phi, T):
    if method in ("pc", "pcalm"):
        sch = Schedule(family=method, budget=T, alpha=1.0, inner_steps=1,
                       weight_credit_timing="pre_dual_energy")
        return make_update_fn(sch, scales, skips, phi, slr, 1.0, lr)
    n = L - 1
    beta = beta_nag(math.cos(math.pi / (2 * n + 1)) ** 2 / math.sin(math.pi / (4 * n + 2)) ** 2)
    blocks = [dense_block] * L

    @jax.jit
    def update(p, o, xb, yb):
        g = method_grad_mom(p, scales, skips, blocks, xb, yb, "pc", state_lr=slr, rho=1.0,
                            budget=T, phi=phi, variant="nag", beta=beta)
        return adam_apply(p, g, o, lr)
    return update


def seconds_per_step(update, params, opt, xb, yb, reps):
    p, o = update(params, opt, xb, yb)
    jax.block_until_ready(p)
    t0 = time.time()
    for _ in range(reps):
        p, o = update(p, o, xb, yb)
    jax.block_until_ready(p)
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depths", default="8,16,32,64,128")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    W, phi = a.width, activation_fn("relu")
    rng = np.random.default_rng(0)
    xb = jnp.asarray(rng.random((64, 784), dtype=np.float32))
    yb = jnp.asarray(np.eye(10, dtype=np.float32)[rng.integers(0, 10, 64)])
    print("width,depth,method,fixed_ms,per_iter_ms", flush=True)
    for L in [int(d) for d in a.depths.split(",")]:
        scales, skips = model_scales(W, L, 784), skip_mask(L)
        params = init_params(jax.random.PRNGKey(0), depth=L, width=W, input_dim=784,
                             output_dim=10, dtype=jnp.float32)
        opt = adam_init(params)
        lr = adam_learning_rate(W, L, 1e-3, 1.0, None)
        slr = eta_for_depths([L], width=W)[0][L]
        for method in ("pc", "pcalm", "nag"):
            t1, t2 = L, 4 * L
            s1 = seconds_per_step(step_fn(method, L, W, slr, lr, scales, skips, phi, t1),
                                  params, opt, xb, yb, a.reps)
            s2 = seconds_per_step(step_fn(method, L, W, slr, lr, scales, skips, phi, t2),
                                  params, opt, xb, yb, a.reps)
            per = (s2 - s1) / (t2 - t1)
            print(f"{W},{L},{method},{1e3 * (s1 - t1 * per):.3f},{1e3 * per:.5f}", flush=True)


if __name__ == "__main__":
    main()
