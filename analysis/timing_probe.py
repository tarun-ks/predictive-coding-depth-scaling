"""Wall-clock cost of one inner iteration, PC-ALM against PC.

PC-ALM's dual update needs an extra residual pass, so its iterations cost more. This
times full training steps at T = 2L for both methods, at two run lengths so that the
compile time cancels: per-step time is (t_long - t_short) / (steps_long - steps_short).
At equal T the per-step ratio is the per-iteration ratio. Run on an otherwise idle
machine; parallel load distorts it.
"""
from __future__ import annotations
import sys, tempfile, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcalm.config import load_config
from pcalm.training import train_one

BASE = load_config("configs/headline_mnist.yaml")


def run(depth, T, method, subset, scratch):
    cfg = replace(
        BASE,
        output_dir=f"{scratch}/l{depth}_t{T}_{method}_{subset}",
        model=replace(BASE.model, depth=depth),
        method=replace(BASE.method, name=method, budget=T),
        training=replace(BASE.training, train_subset=subset, test_subset=512),
    )
    t0 = time.time()
    out = train_one(cfg, data_dir="data")
    return time.time() - t0, out["steps"]


def main():
    print("depth,T,method,compile_s,per_step_s", flush=True)
    with tempfile.TemporaryDirectory() as scratch:
        for depth in (32, 64, 128, 256):
            T = 2 * depth
            for method in ("pc", "pcalm"):
                t_a, n_a = run(depth, T, method, 1280, scratch)     # 20 steps
                t_b, n_b = run(depth, T, method, 12800, scratch)    # 200 steps
                per = (t_b - t_a) / max(n_b - n_a, 1)
                print(f"{depth},{T},{method},{t_a - n_a * per:.1f},{per:.5f}", flush=True)


if __name__ == "__main__":
    main()
