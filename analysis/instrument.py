"""Read-only instrumentation of the PC / PC-ALM inner inference loop.

The activity step, the dual step and the residual definition are all imported
verbatim from `pcalm.inference`. The activity trajectory produced here is the
same sequence `run_pc` / `run_pcalm` would produce for the same budget; the only
addition is that statistics are read out along the way. No update rule is
altered.

Two families of per-iteration statistic are recorded:

* `resid`  - max over layers of ||z_l - pred_l||_F / ||z_l||_F.
             This is the quantity in the literal T_res definition. Note it is
             exactly 0 at T=0 because `free_init` is the forward pass, and for
             PC it *grows* toward a nonzero equilibrium (the equilibrium
             residuals are PC's error signals), so thresholding it downward is
             degenerate. Recorded anyway, for the record.
* `dist`   - max over layers of ||z_l(T) - z_l*||_F / ||z_l*||_F, where z* is
             the activity state after a large reference budget. This is a
             well-posed "iterations to converge" measure for both methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from pcalm.inference import _solve_inner, constraint_residuals, free_init, zero_duals_like


def _advance(params, scales, skips, x, y, free, duals, state_lr, rho, alpha,
             inner_steps, phi, family):
    """One outer iteration, exactly as the corresponding repo routine does it."""
    free = _solve_inner(params, scales, skips, x, y, free, duals,
                        state_lr, rho, inner_steps, phi)
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    if family == "pcalm":
        duals = [lam + alpha * r for lam, r in zip(duals, residuals)]
    return free, duals, residuals


def _init_state(params, scales, skips, x, phi):
    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    return free0, duals0


def make_final_state_fn(scales, skips, phi, *, family, state_lr, rho, alpha,
                        inner_steps, budget):
    """fn(params, x, y) -> activity state after `budget` outer iterations."""
    if family not in {"pc", "pcalm"}:
        raise ValueError(f"unsupported family: {family}")

    @jax.jit
    def final_state(params, x, y):
        free, duals = _init_state(params, scales, skips, x, phi)

        def step(carry, _):
            f, d = carry
            f, d, _r = _advance(params, scales, skips, x, y, f, d, state_lr, rho,
                                alpha, inner_steps, phi, family)
            return (f, d), None

        (free, _), _ = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return free

    return final_state


def make_traj_fn(scales, skips, phi, *, family, state_lr, rho, alpha,
                 inner_steps, budget):
    """fn(params, x, y, free_star) -> (resid[T], dist[T]), both max over layers."""
    if family not in {"pc", "pcalm"}:
        raise ValueError(f"unsupported family: {family}")

    @jax.jit
    def traj(params, x, y, free_star):
        free, duals = _init_state(params, scales, skips, x, phi)

        def step(carry, _):
            f, d = carry
            f, d, residuals = _advance(params, scales, skips, x, y, f, d, state_lr,
                                       rho, alpha, inner_steps, phi, family)
            resid = jnp.max(jnp.stack([
                jnp.sqrt(jnp.sum(r * r)) / jnp.maximum(jnp.sqrt(jnp.sum(z * z)), 1e-30)
                for r, z in zip(residuals, f)
            ]))
            dist = jnp.max(jnp.stack([
                jnp.sqrt(jnp.sum((z - zs) ** 2)) / jnp.maximum(jnp.sqrt(jnp.sum(zs * zs)), 1e-30)
                for z, zs in zip(f, free_star)
            ]))
            return (f, d), (resid, dist)

        _, ys = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return ys

    return traj


def first_below(series, tol):
    """1-indexed first iteration whose value is < tol, or None if never reached."""
    for i, v in enumerate(series):
        if float(v) < tol:
            return i + 1
    return None


def make_advance_fn(scales, skips, phi, *, family, state_lr, rho, alpha,
                    inner_steps, budget):
    """fn(params, x, y, free, duals) -> (free, duals) after `budget` iterations.

    The inner dynamics are Markovian in (free, duals), so chaining calls to this
    is identical to one long run. Used to extend the reference solve until the
    activity state stops moving, without recomputing from scratch.
    """
    if family not in {"pc", "pcalm"}:
        raise ValueError(f"unsupported family: {family}")

    @jax.jit
    def advance(params, x, y, free, duals):
        def step(carry, _):
            f, d = carry
            f, d, _r = _advance(params, scales, skips, x, y, f, d, state_lr, rho,
                                alpha, inner_steps, phi, family)
            return (f, d), None

        (free_out, duals_out), _ = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return free_out, duals_out

    return advance


def initial_state(params, scales, skips, x, phi):
    return _init_state(params, scales, skips, x, phi)


def rel_gap(a, b):
    """max over layers of ||a_l - b_l|| / ||b_l||."""
    return float(jnp.max(jnp.stack([
        jnp.sqrt(jnp.sum((u - v) ** 2)) / jnp.maximum(jnp.sqrt(jnp.sum(v * v)), 1e-30)
        for u, v in zip(a, b)
    ])))


def make_resid_traj_fn(scales, skips, phi, *, family, state_lr, rho, alpha,
                       inner_steps, budget):
    """fn(params, x, y) -> resid[T]: max-over-layers relative constraint residual."""
    if family not in {"pc", "pcalm"}:
        raise ValueError(f"unsupported family: {family}")

    @jax.jit
    def traj(params, x, y):
        free, duals = _init_state(params, scales, skips, x, phi)

        def step(carry, _):
            f, d = carry
            f, d, residuals = _advance(params, scales, skips, x, y, f, d, state_lr,
                                       rho, alpha, inner_steps, phi, family)
            resid = jnp.max(jnp.stack([
                jnp.sqrt(jnp.sum(r * r)) / jnp.maximum(jnp.sqrt(jnp.sum(z * z)), 1e-30)
                for r, z in zip(residuals, f)
            ]))
            return (f, d), resid

        _, ys = jax.lax.scan(step, (free, duals), xs=None, length=budget)
        return ys

    return traj
