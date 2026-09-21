"""Architecture-generic PC / PC-ALM, mirroring pcalm/inference.py line for line
with a PLUGGABLE per-layer block predictor.

Why this exists: pcalm/inference.py hard-codes a dense block (`inp @ W.T`), so a
conv variant cannot be expressed without editing the reference package -- which
the protocol forbids. This module re-expresses the same energy, the same activity
step, the same dual step and the same weight-credit timing, with the block
predictor supplied per layer.

VALIDATION CONTRACT: instantiated with dense blocks it must reproduce the
reference implementation's published cell exactly (78.66 / 68.13 / 77.75).
analysis/validate_generic.py checks this. Any conv result is only as trustworthy
as that check.
"""
from __future__ import annotations
import jax, jax.numpy as jnp


def dense_block(W, scale, skip, z_prev, phi, is_first):
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * (inp @ W.T)
    return pred + z_prev if skip else pred


def conv_block(W, scale, skip, z_prev, phi, is_first):
    """3x3 same-padding conv on NHWC. Skip is identity, matching the dense case."""
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * jax.lax.conv_general_dilated(
        inp, W, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return pred + z_prev if skip else pred


def flatten_dense_block(W, scale, skip, z_prev, phi, is_first):
    """Final read-out: flatten the feature map, then dense. Never carries a skip.

    Flatten rather than global-average-pool: the dense reference's read-out sees
    its entire width-dim hidden state, so the faithful conv analogue reads the
    whole H*W*C feature map. Global pooling would compress 7x7x8 to 8 dimensions
    for a 10-class problem and bottleneck the architecture rather than test it.
    """
    inp = z_prev if is_first else phi(z_prev)
    # Average-pool 7x7 -> 2x2 before flattening, giving 2*2*C = 32 features, the
    # same read-out capacity as the dense reference (width 32). Flattening the
    # full 7*7*C = 392 map let a linear read-out solve the task on its own, so
    # the conv stack contributed nothing and depth was never exercised.
    pooled = jax.lax.reduce_window(inp, 0.0, jax.lax.add, (1, 3, 3, 1), (1, 3, 3, 1),
                                   "VALID") / 9.0
    flat = pooled.reshape(pooled.shape[0], -1)
    return scale * (flat @ W.T)


def forward(params, scales, skips, blocks, x, phi):
    acts, z = [], x
    for i, W in enumerate(params):
        z = blocks[i](W, scales[i], skips[i], z, phi, is_first=(i == 0))
        acts.append(z)
    return acts


def free_init(params, scales, skips, blocks, x, phi):
    return forward(params, scales, skips, blocks, x, phi)[:-1]


def supervised_loss(params, scales, skips, blocks, x, y, free, phi):
    y_pred = blocks[-1](params[-1], scales[-1], skips[-1], free[-1], phi, is_first=False)
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def constraint_residuals(params, scales, skips, blocks, x, free, phi):
    out = []
    for i, z in enumerate(free):
        prev = x if i == 0 else free[i - 1]
        out.append(z - blocks[i](params[i], scales[i], skips[i], prev, phi, is_first=(i == 0)))
    return out


def zero_duals_like(rs):
    return [jnp.zeros_like(r) for r in rs]


def al_energy_shifted(params, scales, skips, blocks, x, y, free, duals, rho, phi):
    rs = constraint_residuals(params, scales, skips, blocks, x, free, phi)
    tot = supervised_loss(params, scales, skips, blocks, x, y, free, phi)
    bs = x.shape[0]
    for r, l in zip(rs, duals):
        sh = r + l / rho
        tot = tot + 0.5 * rho * jnp.sum(sh * sh) / bs
    return tot


def solve_inner(params, scales, skips, blocks, x, y, free, duals, state_lr, rho, steps, phi):
    grad_free = jax.grad(lambda f: al_energy_shifted(
        params, scales, skips, blocks, x, y, f, duals, rho, phi))
    eff = state_lr * free[0].shape[0]      # batch-mean energy -> per-sample step

    def step(f, _):
        g = grad_free(f)
        return [z - eff * gg for z, gg in zip(f, g)], None
    if steps <= 0:
        return free
    free, _ = jax.lax.scan(step, free, xs=None, length=steps)
    return free


def run_pc(params, scales, skips, blocks, x, y, *, state_lr, rho, steps, phi):
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    return solve_inner(params, scales, skips, blocks, x, y, f0, d0,
                       state_lr, rho, steps, phi), d0


def run_pcalm(params, scales, skips, blocks, x, y, *, state_lr, rho, alpha, budget,
              inner_steps, phi, freeze=None):
    """freeze: optional boolean list over the L-1 hidden constraints. Where True,
    that layer's dual is held at zero (no ascent). freeze=None reproduces the
    reference exactly and is the validated default."""
    free = free_init(params, scales, skips, blocks, x, phi)
    duals = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, free, phi))

    def dual_step(d, rs):
        upd = [l + alpha * r for l, r in zip(d, rs)]
        if freeze is None:
            return upd
        return [l if fz else u for l, u, fz in zip(d, upd, freeze)]

    def outer(carry, _):
        f, d = carry
        f = solve_inner(params, scales, skips, blocks, x, y, f, d, state_lr, rho,
                        inner_steps, phi)
        rs = constraint_residuals(params, scales, skips, blocks, x, f, phi)
        return (f, dual_step(d, rs)), None

    if budget > 1:
        (free, duals_before), _ = jax.lax.scan(outer, (free, duals), xs=None,
                                               length=budget - 1)
    else:
        duals_before = duals
    free = solve_inner(params, scales, skips, blocks, x, y, free, duals_before,
                       state_lr, rho, inner_steps, phi)
    return free, duals_before          # pre_dual_energy, the paper's Algorithm 1


def method_grad(params, scales, skips, blocks, x, y, family, *, state_lr, rho, alpha,
                budget, inner_steps, phi, freeze=None):
    if family == "bp":
        return jax.grad(lambda p: 0.5 * jnp.mean(jnp.sum(
            (forward(p, scales, skips, blocks, x, phi)[-1] - y) ** 2, axis=-1)))(params)
    if family == "pc":
        free, duals = run_pc(params, scales, skips, blocks, x, y,
                             state_lr=state_lr, rho=rho, steps=budget, phi=phi)
    else:
        free, duals = run_pcalm(params, scales, skips, blocks, x, y, state_lr=state_lr,
                                rho=rho, alpha=alpha, budget=budget,
                                inner_steps=inner_steps, phi=phi, freeze=freeze)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    return jax.grad(lambda p: al_energy_shifted(
        p, scales, skips, blocks, x, y, free, duals, rho, phi))(params)
