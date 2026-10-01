"""The depth-scaling protocol at four times the reference width.

Residual MLP of width 128 (the widest cell of the reference implementation's frozen step
table), MNIST, depths 8-128, three seeds, one epoch, the geometric budget ladder. Four arms:
backpropagation (the accuracy bar), plain PC, PC-ALM, and PC with Nesterov momentum whose
beta is set from depth alone (the closed-form condition number of an orthogonal chain),
so the recipe needs no measurement at all.

Every cell is one call to the existing runners (run_sweep.py for bp/pc/pcalm,
run_momentum.py for Nesterov); this script only schedules them, longest first, on a
process pool. Rows land in results/wide128/L{L}_s{seed}_{method}.jsonl and a rerun skips
budgets already done. Each job runs with the same XLA threading as the main sweep
(Eigen multithreading off, two intra-op threads), seven at a time, which keeps runs
bitwise reproducible and the machine from oversubscribing.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="2",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")


def jobs(width, depths, seeds, out):
    for L in depths:
        for s in seeds:
            base = ["--depth", str(L), "--seed", str(s), "--width", str(width)]
            f = lambda m: str(out / f"L{L}_s{s}_{m}.jsonl")
            # rough cost ~ L * (largest budget): PC's ladder reaches 24L, the others 4L
            yield (L * 24 * L, [PY, "analysis/run_sweep.py", *base, "--method", "pc",
                                "--ref-mult", "24", "--out", f("pc")])
            yield (L * 4 * L * 1.7, [PY, "analysis/run_sweep.py", *base, "--method", "pcalm",
                                     "--ref-mult", "4", "--out", f("pcalm")])
            yield (L * 4 * L * 1.7, [PY, "analysis/run_momentum.py", *base, "--variant", "nag",
                                     "--beta-from", "depth", "--ref-mult", "4", "--out", f("nag")])
            yield (L, [PY, "analysis/run_sweep.py", *base, "--method", "bp", "--out", f("bp")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depths", default="8,16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="results/wide128")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    todo = sorted(jobs(a.width, [int(d) for d in a.depths.split(",")],
                       [int(s) for s in a.seeds.split(",")], out), key=lambda j: -j[0])
    logs = ROOT / "logs" / "wide128"
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        cmd = job[1]
        name = Path(cmd[-1]).stem
        with open(logs / f"{name}.log", "a") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
        print(f"{'done' if rc == 0 else 'FAILED'} {name}", flush=True)
        return rc

    with ThreadPoolExecutor(a.workers) as pool:
        rcs = list(pool.map(run, todo))
    print(f"{sum(r == 0 for r in rcs)} of {len(rcs)} cells finished cleanly", flush=True)


if __name__ == "__main__":
    main()
