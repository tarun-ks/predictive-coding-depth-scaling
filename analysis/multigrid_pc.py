"""Grid transfers and cost accounting for multigrid on the activity relaxation.

The inner problem is quadratically conditioned (kappa = Theta(L^2), see kappa.py); a
first-order solver therefore needs Theta(kappa) = Theta(L^2) steps and an accelerated one
Theta(sqrt kappa) = Theta(L); and no scheme whose layers talk only to their neighbours
can use fewer than Omega(L) ROUNDS. Multigrid is the classical answer to a
diameter-limited elliptic solve. analysis/run_multigrid.py is the solver and benchmark;
this module holds the pieces it shares.

WHAT IS COARSENED. The layer index is a one-dimensional grid of n = L-1 hidden layers
(all hidden activities share the reference width), and the activity Hessian is block
tridiagonal along it. We coarsen the depth axis with standard vertex-centred full
weighting: coarse point j sits on fine index 2j+1 and averages fine indices 2j, 2j+1,
2j+2. For odd n, which is every depth in this study (n = L-1, L a power of two), the
coarse grid covers every fine layer including layer 0, and every index is in bounds.
Prolongation is linear interpolation, P = 2 R^T.

COST COUNTERS.
  * work   = fine-level gradient or Hessian-vector evaluations: what a serial
             implementation pays.
  * rounds = nearest-neighbour communication rounds. A level-k operation spans 2^k
             fine layers, so it costs 2^k rounds. An exact line search needs a GLOBAL
             reduction (g.g and g.Ag summed over every layer), which on a chain of n
             layers costs 2(n-1) rounds: n-1 hops to reduce to one end, n-1 to
             broadcast the step back. Fixed-step methods (Richardson, Nesterov) need no
             reduction. The nearest-neighbour floor bounds rounds, not work.

pcalm/ is untouched: everything here is an external caller.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import jax.numpy as jnp


def restrict(v):
    """Full weighting along axis 0: v_c[j] = (v[2j] + 2 v[2j+1] + v[2j+2]) / 4.

    v has shape (n, batch, width). Returns None when the grid is too small to coarsen.
    """
    n = v.shape[0]
    nc = (n - 1) // 2
    if nc < 1:
        return None
    idx = 2 * jnp.arange(nc) + 1                   # 1, 3, ..., 2nc-1; 2nc <= n-1
    return 0.25 * (v[idx - 1] + 2.0 * v[idx] + v[idx + 1])


def prolong(vc, n):
    """Linear interpolation back to n fine layers; the adjoint of restrict, times two."""
    nc = vc.shape[0]
    out = jnp.zeros((n,) + vc.shape[1:], vc.dtype)
    idx = 2 * jnp.arange(nc) + 1
    out = out.at[idx].add(vc)
    out = out.at[idx - 1].add(0.5 * vc)
    out = out.at[idx + 1].add(0.5 * vc)
    return out


def _stack(free):   return jnp.stack(free, axis=0)
def _unstack(arr):  return [arr[i] for i in range(arr.shape[0])]


@dataclass
class Cost:
    work: int = 0                 # fine-level gradient / hvp evaluations
    rounds: int = 0               # nearest-neighbour communication rounds
    reductions: int = 0           # global reductions (exact line searches)
    per_level: dict = field(default_factory=dict)

    def charge(self, level, n_evals=1):
        span = 2 ** level         # fine layers a level-k operation reaches across
        self.work += n_evals
        self.rounds += n_evals * span
        self.per_level[level] = self.per_level.get(level, 0) + n_evals

    def reduce(self, n_layers):
        """One global reduction plus broadcast along a chain of n_layers layers."""
        self.reductions += 1
        self.rounds += 2 * max(n_layers - 1, 0)


def level_sizes(nfine, levels):
    """Grid sizes down the hierarchy, stopping when a level would be smaller than 2."""
    sizes = [nfine]
    for _ in range(levels):
        nc = (sizes[-1] - 1) // 2
        if nc < 2:
            break
        sizes.append(nc)
    return sizes
