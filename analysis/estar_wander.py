"""Why the reference minimum E* is the LOWEST iterate, not the last.

With ReLU the inner energy is piecewise quadratic, and none of the reference solvers is
monotone near its minimum. Steepest descent with exact line search takes the step that
is exact for the quadratic piece it is on; when that step crosses a kink the energy can
rise. This runs the steepest-descent and multigrid references of run_multigrid.py at two
budgets and prints, for each, the final energy, the lowest energy, where the lowest
occurred, and how many steps raised the energy.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from analysis.generic_pc import al_energy_shifted, constraint_residuals, free_init, zero_duals_like
from analysis.multigrid_pc import Cost, _stack, _unstack, level_sizes
from analysis.run_multigrid import build, make_ops, power_lmax, run_mg, run_sd
from pcalm.data import load_dataset


def main():
    phi = jax.nn.relu
    print("depth,seed,ref_work,solver,e_final,e_min,argmin,n_iter,n_increases")
    for L, s_ in [(8, 0), (8, 2), (32, 1)]:
        x_tr, y_tr, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=s_,
                                        data_dir="data", input_dim=784, output_dim=10)
        x, y = jnp.asarray(x_tr[:1]), jnp.asarray(y_tr[:1])
        params, sc, sk, blocks = build(L, s_)
        f0 = free_init(params, sc, sk, blocks, x, phi)
        d0 = zero_duals_like(constraint_residuals(params, sc, sk, blocks, x, f0, phi))
        energy = jax.jit(lambda arr: al_energy_shifted(params, sc, sk, blocks, x, y,
                                                       _unstack(arr), d0, 1.0, phi))
        g, h = make_ops(params, sc, sk, blocks, x, y, d0, phi)
        arr0 = _stack(f0)
        sizes = level_sizes(len(f0), 12)
        lm = power_lmax(arr0, h)
        for rw in (20000, 80000):
            for name in ("sd", "mg"):
                tr = []
                if name == "sd":
                    _, e = run_sd(arr0, energy, g, h, -1e30, rw // 2, Cost(), trace=tr)
                else:
                    _, e = run_mg(arr0, energy, g, h, sizes, -1e30,
                                  rw // (2 * 2 * (2 * len(sizes) - 1)), 2, Cost(), 1.0 / lm,
                                  trace=tr)
                es = np.array([t[2] for t in tr])
                print(f"{L},{s_},{rw},{name},{e:.7f},{es.min():.7f},{int(es.argmin())},"
                      f"{len(es)},{int(np.sum(np.diff(es) > 0))}", flush=True)


if __name__ == "__main__":
    main()
