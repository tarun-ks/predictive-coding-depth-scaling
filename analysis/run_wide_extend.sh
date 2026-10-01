#!/usr/bin/env bash
# Extend the width-128 ladders where the strict (97-98% of BP) bars were censored by the
# ladder's end: PC and Nesterov at L=128, plus PC and Nesterov at L=64. Same XLA threading
# as run_wide.py; rows append to the same files and budgets already run are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
PY=.venv/bin/python
{
  for s in 0 1 2; do echo "analysis/run_sweep.py --depth 128 --seed $s --width 128 --method pc --budgets 4096,6144 --out results/wide128/L128_s${s}_pc.jsonl"; done
  for s in 0 1 2; do echo "analysis/run_momentum.py --depth 128 --seed $s --width 128 --variant nag --beta-from depth --budgets 768,1024 --out results/wide128/L128_s${s}_nag.jsonl"; done
  for s in 0 1 2; do echo "analysis/run_sweep.py --depth 64 --seed $s --width 128 --method pc --budgets 2048,3072 --out results/wide128/L64_s${s}_pc.jsonl"; done
  for s in 0 1 2; do echo "analysis/run_momentum.py --depth 64 --seed $s --width 128 --variant nag --beta-from depth --budgets 384,512 --out results/wide128/L64_s${s}_nag.jsonl"; done
} | xargs -P 7 -I{} sh -c "$PY {} >> logs/wide128/extend.log 2>&1"
echo "extension finished"
