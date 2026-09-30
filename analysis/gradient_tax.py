"""Where does the accelerated solver's extra cost come from?

Training-side budgets: PC's T/kappa is flat (about 0.030 from L=8 to 128), so plain
gradient descent pays exactly its Theta(kappa) rate. The accelerated solvers' T/sqrt(kappa)
climbs from 0.42 to 0.88, so they pay more than Theta(sqrt kappa). This measures the inner
solve directly, at initialisation, with no outer loop: how many iterations until the WEIGHT
GRADIENT the inner solve produces is within delta of its converged value, overall and for
the input-side and output-side quarters of the network separately?

Reference: the gradient after a long run. Its convergence is checked PER GROUP, by how far
a further 50% more iterations move that group's gradient; a group's budget at delta is
only trusted when that movement is below delta/3.

The budget is a HOLD-WINDOW crossing, as in the main sweep: the first grid point t at
which the relative error stays at or below delta for every grid point in [t, 2t]. An
accelerated solver oscillates, so a first crossing can record a transient dip; the
first crossing is stored alongside for comparison, not used.

Groups: the whole network; the input layer W_0 on its own (it has 784 x 32 entries,
25x any hidden matrix, and would dominate any group it joins); the first and last
quarter of the HIDDEN matrices; and the read-out layer.
Solvers descend pcalm.inference's own energy; the weight gradient is jax.grad of that
energy in the parameters at fixed activities, which is PC's update with zero duals.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.inference import al_energy_shifted, constraint_residuals, free_init, zero_duals_like
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def kappa_table():
    k = {}
    for fn in ("results/kappa/standard_w32.csv", "results/kappa/mup_256.csv",
               "results/strengthen/kappa_512.csv"):
        for r in csv.DictReader(open(fn)):
            try: k.setdefault(int(r["depth"]), []).append(float(r["kappa"]))
            except (ValueError, KeyError): pass
    return {L: float(np.exp(np.mean(np.log(v)))) for L, v in k.items()}


def lam_max(hvp, like, iters=300):
    v = [jax.random.normal(jax.random.PRNGKey(7 + i), z.shape) for i, z in enumerate(like)]
    nrm = lambda a: float(jnp.sqrt(sum(jnp.sum(t * t) for t in a)))
    n = nrm(v); v = [t / n for t in v]; lam = 0.0
    for _ in range(iters):
        w = hvp(v); lam = float(sum(jnp.sum(a * b) for a, b in zip(v, w)))
        nw = nrm(w)
        if nw == 0: break
        v = [t / nw for t in w]
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,16,32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--solver", choices=["nag", "gd"], default="nag")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phi = activation_fn("relu")
    K = kappa_table()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["depth"], r["seed"], r["solver"]))
    with out.open("a") as fh:
        for L in [int(d) for d in a.depths.split(",")]:
            for s in [int(z) for z in a.seeds.split(",")]:
                if (L, s, a.solver) in done: continue
                t0 = time.time()
                xt, yt, _, _ = load_dataset("mnist", train_subset=max(a.batch, 8), test_subset=8,
                                            seed=s, data_dir="data", input_dim=784, output_dim=10)
                x, y = jnp.asarray(xt[:a.batch]), jnp.asarray(yt[:a.batch])
                params = list(init_params(jax.random.PRNGKey(s), depth=L, width=32,
                                          input_dim=784, output_dim=10, dtype=jnp.float32))
                sc, sk = model_scales(32, L, 784), skip_mask(L)
                f0 = free_init(params, sc, sk, x, phi)
                d0 = zero_duals_like(constraint_residuals(params, sc, sk, x, f0, phi))
                E = lambda f: al_energy_shifted(params, sc, sk, x, y, f, d0, 1.0, phi)
                gz = jax.grad(E)
                hvp = jax.jit(lambda v: jax.jvp(gz, (f0,), (v,))[1])
                lm = lam_max(hvp, f0)
                eta = 1.0 / lm                      # per-energy step; energy is a batch mean
                kap = K.get(L, float(L) ** 2)
                beta = (math.sqrt(kap) - 1) / (math.sqrt(kap) + 1) if a.solver == "nag" else 0.0
                gw = jax.jit(lambda f: jax.grad(lambda p: al_energy_shifted(
                    p, sc, sk, x, y, f, d0, 1.0, phi))(params))

                def _run(state, n):
                    def step(c, _):
                        f, fp = c
                        yv = [z + beta * (z - q) for z, q in zip(f, fp)]
                        g = gz(yv)
                        return ([z - eta * gg for z, gg in zip(yv, g)], f), None
                    return jax.lax.scan(step, state, None, length=n)[0]
                _jrun = jax.jit(_run, static_argnums=1)

                def run(state, d):
                    """Advance d steps in power-of-two chunks: at most ~17 compilations."""
                    while d > 0:
                        p = 1 << (d.bit_length() - 1)
                        state = _jrun(state, p); d -= p
                    return state

                def flat(g): return jnp.concatenate([t.ravel() for t in g])
                nL = len(params)
                hid = list(range(1, nL - 1))
                q = max(1, len(hid) // 4)
                groups = {"all": list(range(nL)), "input_layer": [0],
                          "input_hidden": hid[:q], "output_hidden": hid[-q:],
                          "readout": [nL - 1]}
                def gsub(g, idx): return jnp.concatenate([g[i].ravel() for i in idx])

                # reference: long run, verified converged
                unit = int(math.ceil(math.sqrt(kap))) if a.solver == "nag" else int(math.ceil(kap))
                Tref = max(2000, (80 if a.solver == "nag" else 40) * unit)
                st = (f0, f0); st = run(st, Tref); g1 = gw(st[0])
                st = run(st, Tref // 2); g2 = gw(st[0])
                conv = float(jnp.linalg.norm(flat(g2) - flat(g1)) / jnp.linalg.norm(flat(g2)))
                ginf = g2
                def gsub_(g, idx): return jnp.concatenate([g[i].ravel() for i in idx])
                conv_g = {k: float(jnp.linalg.norm(gsub_(g2, v) - gsub_(g1, v))
                                   / jnp.linalg.norm(gsub_(g2, v))) for k, v in groups.items()}

                # trajectory on a geometric grid
                grid = sorted({int(round(v)) for v in np.geomspace(1, Tref, 160)})
                st = (f0, f0); t_prev = 0; traj = []
                g0 = gw(f0)
                def rho(g, idx):
                    num = jnp.linalg.norm(gsub(g, idx) - gsub(ginf, idx))
                    den = jnp.linalg.norm(gsub(ginf, idx))
                    return float(num / den) if float(den) > 0 else float("nan")
                traj.append((0, {k: rho(g0, v) for k, v in groups.items()}))
                for t in grid:
                    st = run(st, t - t_prev); t_prev = t
                    traj.append((t, {k: rho(gw(st[0]), v) for k, v in groups.items()}))
                first = lambda k, dl: next((t for t, r in traj if r[k] <= dl), None)
                def held(k, dl):
                    for i, (t, r) in enumerate(traj):
                        if t == 0 or r[k] > dl:
                            continue
                        win = [rr[k] for tt, rr in traj[i:] if tt <= 2 * t]
                        if all(x <= dl for x in win):
                            return t
                    return None
                DL = (0.3, 0.1, 0.03, 0.01)
                row = dict(depth=L, seed=s, solver=a.solver, batch=a.batch, kappa=kap,
                           lambda_max=lm, beta=beta, T_ref=Tref, ref_converged_rel=conv,
                           ref_converged_group=conv_g,
                           t_delta={k: {str(dl): held(k, dl) for dl in DL} for k in groups},
                           t_first={k: {str(dl): first(k, dl) for dl in DL} for k in groups},
                           trajectory=[[t, r] for t, r in traj],
                           wall_sec=round(time.time() - t0, 1))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{a.solver} L={L} s={s}] conv={conv:.1e} "
                      f"t(0.1) held: all={row['t_delta']['all']['0.1']} "
                      f"in_hid={row['t_delta']['input_hidden']['0.1']} "
                      f"in_layer={row['t_delta']['input_layer']['0.1']} "
                      f"first all={row['t_first']['all']['0.1']} ({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
