"""Momentum on the ACTIVITY updates: does the L^2 inner-loop cost fall to L^1?

The central question this addresses: the study argues T_target ~ kappa
because plain gradient descent on the activities is a DIFFUSIVE solver -- it needs
Theta(kappa) steps. Acceleration theory says a momentum method needs Theta(sqrt(kappa))
on the same quadratic, which would be BALLISTIC: kappa ~ L^2 => T ~ L^1.

If momentum moves the exponent from 2 to 1, the L^2 is a property of the SOLVER and
is fixable. If it does not, the L^2 is a property of the problem as posed, and the
diameter-floor argument (Omega(L), Section 4.2) is what binds.

This module adds ONLY an inner-solver variant. It does not touch pcalm/ and does not
touch analysis/generic_pc.py (whose bitwise-identical validation must stay intact).
The `gd` variant here must reproduce the existing PC numbers exactly; run_momentum.py
checks that as its control arm.

Step sizes are NOT tuned. They are set from acceleration theory using the
independently measured spectrum (results/kappa/), so this stays a measurement:
  nag: beta = (sqrt(k)-1)/(sqrt(k)+1),  step unchanged from the frozen eta table.
  hb : beta = ((sqrt(k)-1)/(sqrt(k)+1))^2, step scaled by the theoretical optimum
       ratio 4*lmax/(sqrt(lmax)+sqrt(lmin))^2 (-> 4 for large kappa).
NAG is the cleaner test: it holds the step size fixed, so only momentum changes.
HB changes step size AND momentum together; that confound is reported with it.
"""
from __future__ import annotations
import jax, jax.numpy as jnp

from analysis.generic_pc import (al_energy_shifted, constraint_residuals, free_init,
                                 forward, zero_duals_like)


def beta_nag(kappa):
    s = jnp.sqrt(kappa)
    return float((s - 1.0) / (s + 1.0))


def beta_hb(kappa):
    return float(beta_nag(kappa) ** 2)


def hb_step_ratio(lmax, lmin):
    """eta_hb / eta_gd under the classical optimal heavy-ball pair, with eta_gd=1/lmax."""
    import math
    return 4.0 * lmax / (math.sqrt(lmax) + math.sqrt(lmin)) ** 2


def solve_inner_mom(params, scales, skips, blocks, x, y, free, duals, state_lr, rho,
                    steps, phi, *, variant="gd", beta=0.0):
    """Identical to generic_pc.solve_inner when variant='gd'."""
    grad_free = jax.grad(lambda f: al_energy_shifted(
        params, scales, skips, blocks, x, y, f, duals, rho, phi))
    eff = state_lr * free[0].shape[0]      # batch-mean energy -> per-sample step

    if steps <= 0:
        return free

    if variant == "gd":
        def step(f, _):
            g = grad_free(f)
            return [z - eff * gg for z, gg in zip(f, g)], None
        free, _ = jax.lax.scan(step, free, xs=None, length=steps)
        return free

    if variant == "nag":
        def step(carry, _):
            f, fp = carry
            yv = [z + beta * (z - zp) for z, zp in zip(f, fp)]
            g = grad_free(yv)
            fn = [z - eff * gg for z, gg in zip(yv, g)]
            return (fn, f), None
        (free, _), _ = jax.lax.scan(step, (free, free), xs=None, length=steps)
        return free

    if variant == "hb":
        def step(carry, _):
            f, fp = carry
            g = grad_free(f)
            fn = [z - eff * gg + beta * (z - zp) for z, gg, zp in zip(f, g, fp)]
            return (fn, f), None
        (free, _), _ = jax.lax.scan(step, (free, free), xs=None, length=steps)
        return free

    raise ValueError(f"unknown variant {variant!r}")


def run_pc_mom(params, scales, skips, blocks, x, y, *, state_lr, rho, steps, phi,
               variant="gd", beta=0.0):
    f0 = free_init(params, scales, skips, blocks, x, phi)
    d0 = zero_duals_like(constraint_residuals(params, scales, skips, blocks, x, f0, phi))
    return solve_inner_mom(params, scales, skips, blocks, x, y, f0, d0, state_lr, rho,
                           steps, phi, variant=variant, beta=beta), d0


def method_grad_mom(params, scales, skips, blocks, x, y, family, *, state_lr, rho,
                    budget, phi, variant="gd", beta=0.0):
    if family == "bp":
        return jax.grad(lambda p: 0.5 * jnp.mean(jnp.sum(
            (forward(p, scales, skips, blocks, x, phi)[-1] - y) ** 2, axis=-1)))(params)
    if family != "pc":
        raise ValueError("momentum arm is defined for PC only")
    free, duals = run_pc_mom(params, scales, skips, blocks, x, y, state_lr=state_lr,
                             rho=rho, steps=budget, phi=phi, variant=variant, beta=beta)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    return jax.grad(lambda p: al_energy_shifted(
        p, scales, skips, blocks, x, y, free, duals, rho, phi))(params)
