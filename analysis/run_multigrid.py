"""How many rounds, and how much work, does each inner solver need versus depth?

This measures the inner solve directly rather than through training: starting from
the reference initial activities, how many iterations does each solver need to drive the
inner ENERGY to within a fixed fraction of its achievable minimum? That is the quantity the
Theta(kappa) / Theta(sqrt kappa) / Omega(L) ladder actually predicts, and unlike the
training-side budget it needs no epochs, so it can be measured to depth 256 cheaply.

Three solvers, all descending the same energy, none with a tuned step size:
  sd  steepest descent with exact line search   -- expected Theta(kappa) = L^2
  nag Nesterov, beta set from the measured kappa -- expected Theta(sqrt kappa) = L^1
  mg  V-cycle subspace-correction multigrid      -- the question

For multigrid both counters are reported. `work` counts fine-level evaluations;
`rounds` charges a level-k operation 2^k nearest-neighbour hops, which is what it
costs on a substrate where a layer talks only to its neighbours. The nearest-neighbour floor
bounds `rounds`, not `work`, and the two are expected to scale differently.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.model import init_params, model_scales, skip_mask
from analysis.generic_pc import (al_energy_shifted, constraint_residuals, dense_block,
                                 free_init, zero_duals_like)
from analysis.multigrid_pc import (Cost, _stack, _unstack, level_sizes, prolong,
                                   restrict)


def build(L, seed, width=32):
    params = list(init_params(jax.random.PRNGKey(seed), depth=L, width=width,
                              input_dim=784, output_dim=10, dtype=jnp.float32))
    return params, model_scales(width, L, 784), skip_mask(L), [dense_block] * L


def make_ops(params, sc, sk, blocks, x, y, duals, phi):
    energy = lambda arr: al_energy_shifted(params, sc, sk, blocks, x, y,
                                           _unstack(arr), duals, 1.0, phi)
    g = jax.jit(jax.grad(energy))
    h = jax.jit(lambda arr, d: jax.jvp(g, (arr,), (d,))[1])
    return g, h


def _energy_fn(grad_e, energy):
    return energy


def run_sd(arr, energy, grad_e, hvp, target, maxit, cost, trace=None):
    for _ in range(maxit):
        e = float(energy(arr))
        if trace is not None: trace.append((cost.work, cost.rounds, e))
        if e <= target:
            return arr, e
        g = grad_e(arr)
        Ag = hvp(arr, g)
        den = float(jnp.sum(g * Ag))
        if den <= 0:
            return arr, e
        arr = arr - (float(jnp.sum(g * g)) / den) * g
        cost.charge(0, 2)
    return arr, float(energy(arr))


def run_nag(arr, energy, grad_e, eff, beta, target, maxit, cost, trace=None):
    prev = arr
    for _ in range(maxit):
        e = float(energy(arr))
        if trace is not None: trace.append((cost.work, cost.rounds, e))
        if e <= target:
            return arr, e
        yv = arr + beta * (arr - prev)
        gy = grad_e(yv)
        prev, arr = arr, yv - eff * gy
        cost.charge(0, 1)
    return arr, float(energy(arr))


def run_mg(arr, energy, grad_e, hvp, sizes, target, maxcyc, nu, cost, smooth_step,
           trace=None):
    """V-cycle: Richardson smoothing at the fine level, line-searched coarse correction.

    The smoother must damp HIGH-frequency error along the layer axis, which is what a
    fixed step near 1/lambda_max does; exact line search does the opposite, choosing
    the step that best reduces the lowest-frequency component, and used as a smoother
    it leaves the oscillatory error the coarse grid cannot see. So the fine level uses
    Richardson at 1/lambda_max, the step PC itself uses, and the coarse levels, where
    the aim is to solve rather than to smooth, use exact line minimisation.
    """
    top = len(sizes) - 1

    def direction(g, k):
        v = g
        for _ in range(k):
            v = restrict(v)
            if v is None:
                return None
        for j in range(k, 0, -1):
            v = prolong(v, sizes[j - 1])
        return v

    for _ in range(maxcyc):
        e = float(energy(arr))
        if trace is not None: trace.append((cost.work, cost.rounds, e))
        if e <= target:
            return arr, e
        for k in list(range(top + 1)) + list(range(top - 1, -1, -1)):
            for _ in range(nu):
                g = grad_e(arr)
                if k == 0:
                    arr = arr - smooth_step * g
                    cost.charge(0, 1)
                    continue
                d = direction(g, k)
                if d is None:
                    continue
                Ad = hvp(arr, d)
                den = float(jnp.sum(d * Ad))
                if den <= 0:
                    continue
                arr = arr - (float(jnp.sum(d * g)) / den) * d
                cost.charge(k, 2)
    return arr, float(energy(arr))


def power_lmax(arr, hvp, iters=150, seed=0):
    v = jax.random.normal(jax.random.PRNGKey(seed), arr.shape)
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = hvp(arr, v)
        lam = float(jnp.sum(v * w))
        n = float(jnp.linalg.norm(w))
        if n == 0: break
        v = w / n
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="8,16,32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--solvers", default="sd,nag,mg")
    ap.add_argument("--frac", type=float, default=0.99,
                    help="fraction of the achievable energy reduction to reach")
    ap.add_argument("--nu", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ref-work", type=int, default=20000,
                    help="work budget used to estimate the reference minimum E*")
    ap.add_argument("--max-work", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phi = jax.nn.relu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["depth"], r["seed"], r["solver"], r["ref_work"]))

    import csv
    kap = {}
    for fn in ("results/kappa/standard_w32.csv", "results/kappa/mup_256.csv"):
        for row in csv.DictReader(open(fn)):
            kap.setdefault(int(row["depth"]), []).append(float(row["kappa"]))

    with out.open("a") as fh:
        for L in [int(d) for d in a.depths.split(",")]:
            for s_ in [int(z) for z in a.seeds.split(",")]:
                x_tr, y_tr, _, _ = load_dataset("mnist", train_subset=max(8, a.batch),
                                                test_subset=8, seed=s_, data_dir="data",
                                                input_dim=784, output_dim=10)
                x, y = jnp.asarray(x_tr[:a.batch]), jnp.asarray(y_tr[:a.batch])
                params, sc, sk, blocks = build(L, s_)
                f0 = free_init(params, sc, sk, blocks, x, phi)
                d0 = zero_duals_like(constraint_residuals(params, sc, sk, blocks, x, f0, phi))
                energy = jax.jit(lambda arr: al_energy_shifted(
                    params, sc, sk, blocks, x, y, _unstack(arr), d0, 1.0, phi))
                grad_e, hvp = make_ops(params, sc, sk, blocks, x, y, d0, phi)
                arr0 = _stack(f0)
                sizes = level_sizes(len(f0), 12)
                e0 = float(energy(arr0))
                lmax_meas = power_lmax(arr0, hvp)
                smooth_step = 1.0 / lmax_meas if lmax_meas > 0 else 0.2
                k_here = float(np.mean(kap.get(L, [max(1.0, L ** 2)])))
                lmax, beta = 4.2, None
                import math as _m
                beta = (_m.sqrt(k_here) - 1) / (_m.sqrt(k_here) + 1)

                # Reference minimum at two budgets a factor of four apart.
                estars = {}
                for rw in (a.ref_work, 4 * a.ref_work):
                    best = e0
                    for fn, kw in (("mg", {}), ("sd", {})):
                        c = Cost()
                        if fn == "mg":
                            _, e = run_mg(arr0, energy, grad_e, hvp, sizes, -1e30,
                                          rw // (2 * a.nu * (2 * len(sizes) - 1)), a.nu, c,
                                          smooth_step)
                        else:
                            _, e = run_sd(arr0, energy, grad_e, hvp, -1e30, rw // 2, c)
                        best = min(best, e)
                    estars[rw] = best

                for rw, estar in estars.items():
                    target = estar + (1.0 - a.frac) * (e0 - estar)
                    for solver in a.solvers.split(","):
                        if (L, s_, solver, rw) in done: continue
                        t0 = time.time(); cost = Cost()
                        if solver == "sd":
                            _, e = run_sd(arr0, energy, grad_e, hvp, target,
                                          a.max_work // 2, cost)
                        elif solver == "nag":
                            _, e = run_nag(arr0, energy, grad_e, (1.0 / lmax) * a.batch,
                                           beta, target, a.max_work, cost)
                        else:
                            _, e = run_mg(arr0, energy, grad_e, hvp, sizes, target,
                                          a.max_work // (2 * a.nu * (2 * len(sizes) - 1)),
                                          a.nu, cost, smooth_step)
                        row = dict(depth=L, seed=s_, solver=solver, frac=a.frac,
                                   nu=a.nu, batch=a.batch, levels=len(sizes) - 1,
                                   kappa=k_here, e0=e0, e_star=estar, target=target,
                           lmax_measured=lmax_meas, smooth_step=smooth_step,
                                   e_reached=e, ref_work=rw,
                                   work=cost.work, rounds=cost.rounds,
                                   converged=bool(e <= target),
                                   per_level=cost.per_level,
                                   wall_sec=round(time.time() - t0, 2))
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"[{solver} L={L} s={s_} ref={rw}] work={cost.work} "
                              f"rounds={cost.rounds} conv={row['converged']} "
                              f"({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
