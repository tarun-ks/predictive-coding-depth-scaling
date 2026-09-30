"""Does momentum on the activities move the depth exponent from 2 to 1?

Same protocol as the headline PC sweep (MNIST, width 32, 1 epoch, same frozen eta
table, same Adam rule, same batch order, same geometric budget ladder, T <= 4L), with
the inner activity solver swapped for a momentum variant. The `gd` arm is the control:
it must reproduce the existing PC sweep numbers, since generic_pc's dense path is
bitwise-identical to the reference.

Momentum parameters come from the independently measured spectrum, not from tuning.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.optim import adam_apply, adam_init
from pcalm.training import adam_learning_rate, batch_order, evaluate, make_eval_fn
from analysis.eta_table import eta_for_depths
from analysis.generic_pc import dense_block
from analysis.momentum_pc import beta_hb, beta_nag, hb_step_ratio, method_grad_mom

LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
          1024, 1536, 2048, 3072, 4096]
SPECTRUM_FILES = ["results/kappa/standard_w32.csv", "results/kappa/mup_256.csv"]


def spectrum_by_depth(root):
    """Geometric mean over seeds of the measured lambda_max / lambda_min / kappa."""
    acc = {}
    for rel in SPECTRUM_FILES:
        p = Path(root) / rel
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            if not r.get("kappa", "").strip():
                continue
            d = int(r["depth"])
            acc.setdefault(d, []).append((float(r["lambda_max"]), float(r["lambda_min"])))
    out = {}
    for d, v in acc.items():
        lmax = math.exp(sum(math.log(a) for a, _ in v) / len(v))
        lmin = math.exp(sum(math.log(b) for _, b in v) / len(v))
        out[d] = (lmax, lmin, lmax / lmin)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["gd", "nag", "hb"], required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ref-mult", type=int, default=4)
    ap.add_argument("--beta-gap-scale", type=float, default=1.0,
                    help="perturb momentum as beta' = 1 - c(1-beta). Since 1-beta ~ "
                         "2/sqrt(kappa), this scales the quantity theory actually sets, "
                         "and stays below 1 for any c > 0. Used to test whether the "
                         "agreement with PC-ALM is knife-edge in beta.")
    ap.add_argument("--budgets", default="",
                    help="explicit comma-separated budgets, bypassing the ladder; used to "
                         "test whether the NAG/PC-ALM agreement survives finer resolution "
                         "or is a coincidence of landing on the same rungs.")
    ap.add_argument("--max-t", type=int, default=0,
                    help="cap the ladder at this budget (default: ref_mult*depth). "
                         "Set to the headline sweep's per-depth maximum so the "
                         "momentum arms see an identical budget set.")
    ap.add_argument("--beta-from", choices=["spectrum", "depth"], default="spectrum",
                    help="'depth' sets beta from the closed-form condition number of an "
                         "isometric chain, kappa = cos^2(pi/(2n+1)) / sin^2(pi/(4n+2)), "
                         "n = L-1: no spectrum is measured. The step is unchanged.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    L, W, BS = a.depth, a.width, 64
    spec = spectrum_by_depth(root)
    if L in spec:
        lmax, lmin, kappa = spec[L]
    elif a.variant == "nag" and a.beta_from == "depth":
        lmax = lmin = kappa = float("nan")        # not needed: beta comes from depth alone
    else:
        raise SystemExit(f"no measured spectrum for depth {L}; run analysis/kappa.py first")
    kappa_used = kappa

    eta, prov = eta_for_depths([L], dataset=a.dataset, activation=a.activation, width=W)
    slr_gd = eta[L]
    if a.variant == "gd":
        beta, slr = 0.0, slr_gd
    elif a.variant == "nag":
        if a.beta_from == "depth":
            import math
            n = L - 1
            kappa_iso = math.cos(math.pi / (2 * n + 1)) ** 2 / math.sin(math.pi / (4 * n + 2)) ** 2
            beta, slr = beta_nag(kappa_iso), slr_gd
            kappa_used = kappa_iso
        else:
            beta, slr = beta_nag(kappa), slr_gd      # step held fixed: only momentum changes
        beta = 1.0 - a.beta_gap_scale * (1.0 - beta)
    else:
        beta, slr = beta_hb(kappa), slr_gd * hb_step_ratio(lmax, lmin)

    phi = activation_fn(a.activation)
    scales, skips = model_scales(W, L, 784), skip_mask(L)
    blocks = [dense_block] * L
    x_tr, y_tr, x_te, y_te = load_dataset(a.dataset, train_subset=60000, test_subset=10000,
                                          seed=a.seed, data_dir=a.data_dir,
                                          input_dim=784, output_dim=10)
    params0 = init_params(jax.random.PRNGKey(a.seed), depth=L, width=W,
                          input_dim=784, output_dim=10, dtype=jnp.float32)
    lr = adam_learning_rate(W, L, 1e-3, 1.0, None)
    eval_batch = make_eval_fn(scales, skips, phi)

    if a.budgets:
        budgets = sorted({int(t) for t in a.budgets.split(",") if t.strip()})
    else:
        hi = a.max_t if a.max_t > 0 else a.ref_mult * L
        budgets = [t for t in LADDER if t <= hi]
        if hi not in budgets:
            budgets.append(hi)
        budgets = sorted(set(budgets))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["budget"])

    with out.open("a") as fh:
        for T in budgets:
            if T in done:
                continue
            t0 = time.time()
            params = [p for p in params0]; opt = adam_init(params)

            @jax.jit
            def update(p, o, xb, yb):
                g = method_grad_mom(p, scales, skips, blocks, xb, yb, "pc",
                                    state_lr=slr, rho=1.0, budget=max(T, 1), phi=phi,
                                    variant=a.variant, beta=beta)
                return adam_apply(p, g, o, lr)

            for idx in batch_order(x_tr.shape[0], BS, a.seed, True):
                params, opt = update(params, opt, jnp.asarray(x_tr[idx]),
                                     jnp.asarray(y_tr[idx]))
            te_mse, te_ce, te_acc = evaluate(params, x_te, y_te, BS, eval_batch)
            row = dict(depth=L, width=W, seed=a.seed, method="pc", variant=a.variant,
                       budget=T, beta=beta, beta_gap_scale=a.beta_gap_scale, beta_from=a.beta_from,
                       kappa_used_for_beta=kappa_used,
                       state_lr=slr, state_lr_gd=slr_gd,
                       state_lr_provenance=prov[L], lambda_max=lmax, lambda_min=lmin,
                       kappa=kappa, learning_rate=lr, dataset=a.dataset, epochs=1,
                       test_acc=float(te_acc), test_mse=float(te_mse), test_ce=float(te_ce),
                       finite=bool(np.isfinite(te_acc) and np.isfinite(te_mse)),
                       wall_sec=round(time.time() - t0, 2))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[mom {a.variant} L={L} s={a.seed}] T={T} beta={beta:.4f} "
                  f"acc={100*float(te_acc):.2f}% ({row['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
