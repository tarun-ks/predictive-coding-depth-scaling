"""Multigrid on the activity relaxation: does a hierarchical solver reach the floor?

Three facts motivate this module. The inner problem is quadratically conditioned
(kappa = Theta(L^2), see kappa.py); a first-order solver therefore needs
Theta(kappa) = Theta(L^2) steps, and an accelerated one Theta(sqrt kappa) = Theta(L);
and no scheme whose layers talk only to their neighbours can use fewer than Omega(L)
ROUNDS. Multigrid is the classical answer to a diameter-limited elliptic solve, and
had not been tried on this problem. This is the test.

WHAT IS COARSENED. The layer index is a one-dimensional grid of size L-1 (all hidden
activities share the reference width), and the activity Hessian is block tridiagonal
along it. So we coarsen the DEPTH axis: full-weighting restriction, linear
interpolation, standard geometric multigrid on a line.

WHY SUBSPACE CORRECTION RATHER THAN GALERKIN. With ReLU the inner energy is piecewise
quadratic, not quadratic, so the residual equation A e = r that textbook multigrid
solves is not available. We use the nonlinear-safe formulation instead: at level k we
minimise the TRUE fine energy over the affine subspace f + range(P_k), i.e. we descend
in the coarse variable d with gradient P_k^T grad E(f + P_k d). Every gradient is an
exact fine-level gradient, so nothing is linearised and the smoother, the coarse
correction and the reference PC solver all descend the same function.

NOTHING IS TUNED. Every step uses exact line minimisation, alpha = g.g / g.Ag, which
costs one extra Hessian-vector product and removes the step size as a free parameter.

TWO COST COUNTERS, AND THEY DIFFER. This is the point of the experiment.
  * work   = number of fine-level gradient/Hessian evaluations. This is what a serial
             implementation pays.
  * rounds = nearest-neighbour communication rounds. A level-k operation spans 2^k
             fine layers, and on a substrate where a layer talks only to its
             neighbours that costs 2^k rounds, so coarse levels are NOT free. This is
             the quantity the nearest-neighbour floor bounds below by Omega(L).
Reporting only `work` would appear to beat the floor; it does not, and the two
counters are what separates the appearance from the fact.

This module adds an inner solver only. pcalm/ is untouched, and with `levels=0` the
solver reduces to steepest descent on the same energy as analysis/generic_pc.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import jax, jax.numpy as jnp

from analysis.generic_pc import al_energy_shifted


# ---------------------------------------------------------------- grid transfers

def restrict(v):
    """Full weighting along the layer axis (axis 0): v_c[j] = (v[2j-1] + 2v[2j] + v[2j+1])/4.

    v has shape (nL, batch, width). Odd end points are handled by clamping, which is
    the standard treatment of a Dirichlet-like boundary and keeps R and P adjoint up
    to the usual factor of two.
    """
    n = v.shape[0]
    nc = (n - 1) // 2
    if nc < 1:
        return None
    idx = 2 * jnp.arange(1, nc + 1)                      # 2, 4, ..., 2nc
    return 0.25 * (v[idx - 1] + 2.0 * v[idx] + v[idx + 1])


def prolong(vc, n):
    """Linear interpolation from the coarse grid back to n fine layers (adjoint of R x2)."""
    nc = vc.shape[0]
    out = jnp.zeros((n,) + vc.shape[1:], vc.dtype)
    idx = 2 * jnp.arange(1, nc + 1)
    out = out.at[idx].add(vc)
    out = out.at[idx - 1].add(0.5 * vc)
    out = out.at[jnp.minimum(idx + 1, n - 1)].add(0.5 * vc)
    return out


def _stack(free):   return jnp.stack(free, axis=0)
def _unstack(arr):  return [arr[i] for i in range(arr.shape[0])]


# ---------------------------------------------------------------- cost accounting

@dataclass
class Cost:
    work: int = 0                 # fine-level gradient / hvp evaluations
    rounds: int = 0               # nearest-neighbour communication rounds
    per_level: dict = field(default_factory=dict)

    def charge(self, level, n_evals=1):
        span = 2 ** level         # fine layers a level-k operation reaches across
        self.work += n_evals
        self.rounds += n_evals * span
        self.per_level[level] = self.per_level.get(level, 0) + n_evals


# ---------------------------------------------------------------- the solver

def level_sizes(nfine, levels):
    """Grid sizes down the hierarchy, stopping when a level would be smaller than 2."""
    sizes = [nfine]
    for _ in range(levels):
        nc = (sizes[-1] - 1) // 2
        if nc < 2:
            break
        sizes.append(nc)
    return sizes


def solve_inner_mg(params, scales, skips, blocks, x, y, free, duals, state_lr, rho,
                   phi, *, cycles=1, levels=3, nu=2, cost=None):
    """V-cycle subspace-correction multigrid on the activities.

    The search direction at level k is the fine gradient restricted k times and
    prolonged back, i.e. the component of the gradient lying in the k-th nested
    subspace. Level 0 is ordinary steepest descent, so `levels=0` is the control arm.
    Every step uses exact line minimisation, so no step size is tuned.

    Returns (free, cost).
    """
    cost = cost if cost is not None else Cost()
    sizes = level_sizes(len(free), levels)
    nfine = sizes[0]

    energy = lambda arr: al_energy_shifted(
        params, scales, skips, blocks, x, y, _unstack(arr), duals, rho, phi)
    grad_e = jax.jit(jax.grad(energy))
    hvp = jax.jit(lambda arr, d: jax.jvp(grad_e, (arr,), (d,))[1])

    def subspace_dir(g, k):
        """Restrict the fine gradient k times, then prolong it back to the fine grid."""
        v = g
        for _ in range(k):
            v = restrict(v)
            if v is None:
                return None
        for j in range(k, 0, -1):
            v = prolong(v, sizes[j - 1])
        return v

    def level_step(arr, k):
        g = grad_e(arr)
        d = subspace_dir(g, k) if k > 0 else g
        if d is None:
            return arr
        Ad = hvp(arr, d)
        den = jnp.sum(d * Ad)
        alpha = jnp.where(den > 0, jnp.sum(d * g) / den, 0.0)
        cost.charge(k, 2)                 # one gradient + one Hessian-vector product
        return arr - alpha * d

    def vcycle(arr):
        top = len(sizes) - 1
        for k in list(range(top + 1)) + list(range(top - 1, -1, -1)):
            for _ in range(nu):
                arr = level_step(arr, k)
        return arr

    arr = _stack(free)
    for _ in range(cycles):
        arr = vcycle(arr)
    return _unstack(arr), cost
