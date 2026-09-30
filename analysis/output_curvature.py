"""Is the output-loss block B indefinite wherever the upper conditioning bound fails?

The upper bound kappa <= ((1 + J_max)^2 + |B|) M^2 (n + 1/2)^2 assumes B is positive
semidefinite. isometry_kappa.py records only max(lambda_max(B), 0). This recomputes B at
the forward-pass point for every cell of the isometry runs and records both extreme
eigenvalues next to whether the bound held there.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp, numpy as np

from pcalm.data import load_dataset
from analysis.isometry_kappa import build, layer_jacobians, output_block


def main():
    print("file,arm,act,gain,depth,seed,kappa,kappa_upper,upper_holds,B_lambda_min,B_lambda_max")
    for fn in sorted(glob.glob("results/strengthen/isometry_[abr]*.jsonl")):
        for line in open(fn):
            if not line.strip():
                continue
            r = json.loads(line)
            if not np.isfinite(r.get("kappa", float("nan"))):
                continue
            xt, yt, _, _ = load_dataset("mnist", train_subset=8, test_subset=8, seed=r["seed"],
                                        data_dir="data", input_dim=784, output_dim=10)
            x, y = jnp.asarray(xt[0:1]), jnp.asarray(yt[0:1])
            params, scales, skips = build(r["depth"], r["seed"], r["arm"], r["act"], r["gain"])
            free, _ = layer_jacobians(params, scales, skips, x, r["act"])
            B = output_block(params, scales, skips, x, y, free, r["act"])
            ev = np.linalg.eigvalsh(0.5 * (B + B.T))
            holds = r["kappa"] <= r["kappa_upper"] * (1 + 1e-6)
            print(f"{Path(fn).name},{r['arm']},{r['act']},{r['gain']},{r['depth']},{r['seed']},"
                  f"{r['kappa']:.3f},{r['kappa_upper']:.4g},{holds},{ev[0]:.6f},{ev[-1]:.6f}",
                  flush=True)


if __name__ == "__main__":
    main()
