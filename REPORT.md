# Inner-iteration scaling with depth: PC vs PC-ALM

Measurement only. No update rule, hyperparameter or algorithm was altered;
all dynamics are imported verbatim from `pcalm/`.

## Provenance

- reference implementation: SakanaAI/pc-alm (MIT), arXiv 2605.31022
- reproduction gate: Fashion-MNIST N=32 L=32 relu seed 0, T=2L -> BP 78.66% / PC 68.13% / PC-ALM 77.75%, matching the repo's published table exactly (0.00 pp on all three)
- config: MNIST, residual MLP, width 32 (their default), relu, 1 epoch, batch 64, Adam lr = 1e-3*sqrt(W/L), gamma0=1, full 60k/10k split
- `state_lr` (eta_h): the reference implementation's frozen eta_best_by_cell.csv at depths 8-128; depths 4 and 256 via the reference implementation's own eta=1/lambda_max rule (calibrated power iteration). Fits are reported both over all depths and restricted to the frozen-table depths.
- total training runs: **925**  |  residual cells: **70**  |  conv cells: **30**

## Headline caveats

1. **Metric (a) as specified is degenerate.** `free_init` sets activities to
   the forward pass, so the constraint residual is *exactly 0* at T=0. For PC
   it then *grows* to a nonzero equilibrium - those equilibrium residuals are
   PC's error signals. `resid_argmin == 1` in every PC cell at both init and
   trained parameters, so "first T below tol" is either 1 or never, at every
   depth. PC is fully censored below; this is a property of the method, not a
   run failure.
2. **PC-ALM has no fixed point in T** at the reference defaults (alpha=1, rho=1,
   inner_steps=1). Its gradient cosine to its own T=100k limit peaks near
   T~1024 then decays; duals grow without settling. It is censored in every
   convergence cell. PC-ALM is accurate at a *budget*, not in a limit.
3. Consequently the load-bearing comparison is the task metric, which is
   well-posed for both methods.

## (b) T_task - literal: first T within 1% absolute of accuracy at largest T


**T_task (literal) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 12.0 | 0.0 | 12 | 12 |
| 16 | 5 | 0 | 48.0 | 0.0 | 48 | 48 |
| 32 | 5 | 0 | 192.0 | 0.0 | 192 | 192 |
| 64 | 5 | 0 | 768.0 | 0.0 | 768 | 768 |
| 128 | 0 | 5 | - | - | - | - |

log-log fit: `+2.000  (residuals identically zero on 5 depth means; CI undefined, not infinitely precise)  n=25/5d`


**T_task (literal) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 24.0 | 0.0 | 24 | 24 |
| 32 | 2 | 3 | 48.0 | 0.0 | 48 | 48 |
| 64 | 3 | 2 | 96.0 | 0.0 | 96 | 96 |
| 128 | 1 | 4 | 256.0 | - | 256 | 256 |

log-log fit: `+1.252  [+1.114, +1.390]  R2=0.994  n=21/6d (clustered, df=4)`

> **On the SD column.** The budget ladder is geometric (ratio ~1.4), so when every seed falls in the same ladder bin the reported SD is 0. That means the seed-to-seed spread is *smaller than one ladder step*, not that it is zero. Read SD=0 as an upper bound of roughly 40% of the mean.

- PC restricted to paper-frozen eta depths: `+2.000  (residuals identically zero on 4 depth means; CI undefined, not infinitely precise)  n=20/4d`
- PC seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 300% per depth step)
- PCALM restricted to paper-frozen eta depths: `+1.258  [+1.100, +1.417]  R2=0.995  n=25/5d (clustered, df=3)`  (recomputed from `results/t_target_bp_relative.csv` by `analysis/report.py`; an earlier value of +1.200 was from a superseded pipeline state)
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 138% per depth step)

### Plateau / censoring audit

| method | depth | n | T_ref | plateau_reached | censored | acc_at_T_ref |
|---|---|---|---|---|---|---|
| pc | 4 | 5 | 16 | 5 | 0 | 0.8975 |
| pc | 8 | 5 | 32 | 5 | 0 | 0.8901 |
| pc | 16 | 5 | 64 | 5 | 0 | 0.8745 |
| pc | 32 | 5 | 512 | 5 | 0 | 0.8545 |
| pc | 64 | 5 | 1024 | 5 | 0 | 0.8304 |
| pc | 128 | 5 | 3072 | 0 | 5 | 0.7955 |
| pcalm | 4 | 5 | 16 | 5 | 0 | 0.898 |
| pcalm | 8 | 5 | 32 | 5 | 0 | 0.8902 |
| pcalm | 16 | 5 | 64 | 5 | 0 | 0.8781 |
| pcalm | 32 | 5 | 512 | 2 | 3 | 0.8241 |
| pcalm | 64 | 5 | 1024 | 3 | 2 | 0.7975 |
| pcalm | 128 | 5 | 1024 | 1 | 4 | 0.7875 |

## (b') T_task - robust: first T reaching a fraction of depth-matched BP


**T_target @ 90% of BP — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 32.0 | 0.0 | 32 | 32 |
| 32 | 5 | 0 | 128.0 | 0.0 | 128 | 128 |
| 64 | 5 | 0 | 512.0 | 0.0 | 512 | 512 |
| 128 | 5 | 0 | 2048.0 | 0.0 | 2048 | 2048 |

log-log fit: `+1.916  [+1.782, +2.050]  R2=0.997  n=30/6d (clustered, df=4)`


**T_target @ 90% of BP — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 16.0 | 0.0 | 16 | 16 |
| 32 | 5 | 0 | 48.0 | 0.0 | 48 | 48 |
| 64 | 5 | 0 | 96.0 | 0.0 | 96 | 96 |
| 128 | 5 | 0 | 256.0 | 0.0 | 256 | 256 |

log-log fit: `+1.269  [+1.177, +1.361]  R2=0.997  n=30/6d (clustered, df=4)`

> **On the SD column.** The budget ladder is geometric (ratio ~1.4), so when every seed falls in the same ladder bin the reported SD is 0. That means the seed-to-seed spread is *smaller than one ladder step*, not that it is zero. Read SD=0 as an upper bound of roughly 40% of the mean.

- PC restricted to frozen-eta depths: `+2.000  (residuals identically zero on 5 depth means; CI undefined, not infinitely precise)  n=25/5d`
- PC seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 269% per depth step)
- PCALM restricted to frozen-eta depths: `+1.258  [+1.100, +1.417]  R2=0.995  n=25/5d (clustered, df=3)`
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 143% per depth step)

**T_target @ 95% of BP — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 32.0 | 0.0 | 32 | 32 |
| 32 | 5 | 0 | 192.0 | 0.0 | 192 | 192 |
| 64 | 5 | 0 | 768.0 | 0.0 | 768 | 768 |
| 128 | 5 | 0 | 3072.0 | 0.0 | 3072 | 3072 |

log-log fit: `+2.067  [+1.863, +2.271]  R2=0.995  n=30/6d (clustered, df=4)`


**T_target @ 95% of BP — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 24.0 | 0.0 | 24 | 24 |
| 32 | 5 | 0 | 48.0 | 0.0 | 48 | 48 |
| 64 | 5 | 0 | 96.0 | 0.0 | 96 | 96 |
| 128 | 5 | 0 | 256.0 | 0.0 | 256 | 256 |

log-log fit: `+1.252  [+1.114, +1.390]  R2=0.994  n=30/6d (clustered, df=4)`

> **On the SD column.** The budget ladder is geometric (ratio ~1.4), so when every seed falls in the same ladder bin the reported SD is 0. That means the seed-to-seed spread is *smaller than one ladder step*, not that it is zero. Read SD=0 as an upper bound of roughly 40% of the mean.

- PC restricted to frozen-eta depths: `+2.175  [+1.989, +2.362]  R2=0.998  n=25/5d (clustered, df=3)`
- PC seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 300% per depth step)
- PCALM restricted to frozen-eta depths: `+1.200  [+1.008, +1.392]  R2=0.993  n=25/5d (clustered, df=3)`
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 143% per depth step)

**T_target @ 98% of BP — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 11.2 | 1.79 | 8 | 12 |
| 16 | 5 | 0 | 48.0 | 0.0 | 48 | 48 |
| 32 | 4 | 1 | 192.0 | 0.0 | 192 | 192 |
| 64 | 4 | 1 | 768.0 | 0.0 | 768 | 768 |
| 128 | 1 | 4 | 3072.0 | - | 3072 | 3072 |

log-log fit: `+2.010  [+1.977, +2.043]  R2=1.000  n=24/6d (clustered, df=4)`


**T_target @ 98% of BP — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 3.0 | 0.0 | 3 | 3 |
| 8 | 5 | 0 | 8.0 | 0.0 | 8 | 8 |
| 16 | 5 | 0 | 24.0 | 0.0 | 24 | 24 |
| 32 | 5 | 0 | 64.0 | 35.78 | 48 | 128 |
| 64 | 5 | 0 | 166.4 | 35.05 | 128 | 192 |
| 128 | 5 | 0 | 409.6 | 57.24 | 384 | 512 |

log-log fit: `+1.421  [+1.370, +1.472]  R2=0.999  n=30/6d (clustered, df=4)`

> **On the SD column.** The budget ladder is geometric (ratio ~1.4), so when every seed falls in the same ladder bin the reported SD is 0. That means the seed-to-seed spread is *smaller than one ladder step*, not that it is zero. Read SD=0 as an upper bound of roughly 40% of the mean.

- PC restricted to frozen-eta depths: `+2.023  [+1.980, +2.066]  R2=1.000  n=19/5d (clustered, df=3)`
- PC seed-vs-depth: seed spread smaller than depth effect (median relative SD 0% vs 300% per depth step)
- PCALM restricted to frozen-eta depths: `+1.410  [+1.327, +1.494]  R2=0.999  n=25/5d (clustered, df=3)`
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 7% vs 167% per depth step)

## (a) T_res - literal constraint-residual threshold

Tolerances 0.001, 0.000316, 0.0001 span one order of magnitude,
chosen from the observed residual range ({'median_resid_at_T1': 0.004954661708325148, 'min_observed': 7.282393653440522e-06, 'max_observed_min': 0.00471119862049818}) and then applied
uniformly to every depth, method, seed and parameter point.


**T_res @ tol=0.001 (init params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.001 (init params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 4 | 1 | 109.75 | 13.33 | 96 | 128 |
| 8 | 4 | 1 | 594.0 | 614.06 | 144 | 1498 |
| 16 | 5 | 0 | 66.8 | 14.1 | 51 | 80 |
| 32 | 5 | 0 | 84.8 | 9.26 | 76 | 97 |
| 64 | 5 | 0 | 72.6 | 23.37 | 41 | 96 |
| 128 | 5 | 0 | 74.6 | 18.76 | 46 | 96 |
| 256 | 5 | 0 | 66.2 | 17.24 | 41 | 87 |

log-log fit: `-0.258  [-0.671, +0.155]  R2=0.340  n=33/7d (clustered, df=5)`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 25% vs 44% per depth step)

**T_res @ tol=0.000316 (init params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.000316 (init params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 264.0 | - | 264 | 264 |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 3 | 2 | 469.33 | 336.79 | 240 | 856 |
| 128 | 5 | 0 | 456.6 | 40.75 | 389 | 492 |
| 256 | 5 | 0 | 670.8 | 65.9 | 610 | 769 |

log-log fit: `+0.200  [+0.003, +0.396]  R2=0.905  n=14/4d (clustered, df=2)`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 10% vs 21% per depth step)

**T_res @ tol=0.0001 (init params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.0001 (init params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: not assessable (too few uncensored depths)

**T_res @ tol=0.001 (trained params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 1.0 | - | 1 | 1 |
| 8 | 1 | 4 | 1.0 | - | 1 | 1 |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.001 (trained params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 11.6 | 8.79 | 1 | 18 |
| 8 | 5 | 0 | 3.6 | 2.61 | 1 | 7 |
| 16 | 5 | 0 | 5.4 | 0.89 | 5 | 7 |
| 32 | 5 | 0 | 4.6 | 1.67 | 3 | 7 |
| 64 | 5 | 0 | 4.6 | 1.67 | 3 | 7 |
| 128 | 5 | 0 | 4.6 | 1.67 | 3 | 7 |
| 256 | 5 | 0 | 5.4 | 3.29 | 3 | 11 |

log-log fit: `-0.021  [-0.242, +0.200]  R2=0.012  n=35/7d (clustered, df=5)`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: **SEED VARIANCE COMPARABLE TO DEPTH EFFECT**: median within-depth relative SD = 36%, vs 22% mean change per depth doubling. The depth trend is NOT cleanly separated from seed noise for this metric.

**T_res @ tol=0.000316 (trained params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.000316 (trained params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 181.8 | 85.16 | 89 | 284 |
| 8 | 4 | 1 | 334.5 | 105.96 | 227 | 428 |
| 16 | 4 | 1 | 680.5 | 120.36 | 501 | 754 |
| 32 | 5 | 0 | 91.6 | 9.24 | 81 | 104 |
| 64 | 5 | 0 | 119.2 | 57.42 | 50 | 161 |
| 128 | 5 | 0 | 74.8 | 22.92 | 57 | 113 |
| 256 | 5 | 0 | 76.8 | 46.63 | 45 | 155 |

log-log fit: `-0.387  [-0.868, +0.094]  R2=0.461  n=33/7d (clustered, df=5)`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 32% vs 44% per depth step)

**T_res @ tol=0.0001 (trained params) — PC** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 0 | 5 | - | - | - | - |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 0 | 5 | - | - | - | - |
| 256 | 0 | 5 | - | - | - | - |

log-log fit: `insufficient uncensored data`


**T_res @ tol=0.0001 (trained params) — PCALM** (mean ± SD over seeds; censored = excluded, counted)

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 2 | 3 | 247.5 | 62.93 | 203 | 292 |
| 8 | 2 | 3 | 830.5 | 62.93 | 786 | 875 |
| 16 | 0 | 5 | - | - | - | - |
| 32 | 0 | 5 | - | - | - | - |
| 64 | 0 | 5 | - | - | - | - |
| 128 | 5 | 0 | 1898.4 | 984.5 | 431 | 3206 |
| 256 | 5 | 0 | 803.8 | 66.76 | 748 | 907 |

log-log fit: `+0.271  [-0.555, +1.097]  R2=0.499  n=14/4d (clustered, df=2)`

- PC seed-vs-depth: not assessable (too few uncensored depths)
- PCALM seed-vs-depth: seed spread smaller than depth effect (median relative SD 17% vs 97% per depth step)

## (a') T_conv - EXPLORATORY ONLY, no exponent quoted

This was my own substitute for the degenerate metric (a), not one of the two
requested measurements. **It failed validation; no slope should be read off
it.** T_conv is measured against the activity state after a reference solve
of budget B, and a valid measurement must be independent of B. It is not:

| depth | n_distinct_B | B_ratio | T_conv_ratio | B_dependent |
|---|---|---|---|---|
| 4 | 1 | 1.0 | - | untestable (single B) |
| 8 | 1 | 1.0 | - | untestable (single B) |
| 16 | 3 | 4.0 | 2.67 | YES |
| 32 | 1 | 1.0 | - | untestable (single B) |
| 64 | 3 | 4.0 | 2.29 | YES |
| 128 | 2 | 2.0 | 1.62 | YES |

At depth 64 one seed stopped at B=131072 and reports T_conv=43329 while two
others ran to B=524288 and report ~99400 - a 2.3x spread driven entirely by
how far the reference solve ran, not by the seed. The apparent seed variance
at depths 16 and 64 is therefore mostly this artefact, not seed noise.

Cause: PC's inner loop has slow modes, and float32 accumulation over
million-step scans stops the fixed point being pinned down at L>=16. The
margin needed grows with depth; 3x is not enough. Raising it costs ~10x more
compute with no guarantee of convergence, and float64 would change the
numerics of the object being measured.

Raw values are in `t_conv.csv`. What survives qualitatively: PC does reach a
fixed point and its iteration count grows steeply with depth; PC-ALM never
stabilises at any depth or seed (30/30 censored). Neither statement needs a
slope. T_conv is excluded from fits.csv.


**T_conv @ tol=0.1 (PC, EXPLORATORY - not a measurement)**

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 0 | 5 | - | - | - | - |
| 8 | 5 | 0 | 60.8 | 17.87 | 29 | 71 |
| 16 | 5 | 0 | 671.2 | 117.68 | 529 | 818 |
| 32 | 5 | 0 | 3411.6 | 450.29 | 2944 | 3979 |
| 64 | 5 | 0 | 17135.4 | 3303.61 | 13677 | 22172 |
| 128 | 5 | 0 | 83696.8 | 8811.64 | 71844 | 92425 |

**T_conv @ tol=0.03 (PC, EXPLORATORY - not a measurement)**

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 31.4 | 2.07 | 29 | 34 |
| 8 | 5 | 0 | 300.2 | 25.01 | 265 | 328 |
| 16 | 5 | 0 | 1925.2 | 256.39 | 1574 | 2190 |
| 32 | 5 | 0 | 8992.0 | 1150.21 | 7253 | 10041 |
| 64 | 5 | 0 | 45262.2 | 9799.76 | 29343 | 53142 |
| 128 | 5 | 0 | 173473.6 | 28144.41 | 141396 | 215508 |

**T_conv @ tol=0.01 (PC, EXPLORATORY - not a measurement)**

| depth | n | censored | mean | SD | min | max |
|---|---|---|---|---|---|---|
| 4 | 5 | 0 | 78.6 | 6.27 | 72 | 87 |
| 8 | 5 | 0 | 562.2 | 26.73 | 515 | 580 |
| 16 | 5 | 0 | 4027.0 | 1701.48 | 2594 | 6938 |
| 32 | 5 | 0 | 12928.0 | 1271.47 | 11427 | 14392 |
| 64 | 5 | 0 | 74808.8 | 24323.88 | 43329 | 99522 |
| 128 | 5 | 0 | 246438.8 | 65880.39 | 193047 | 355541 |


## Divergence / failure audit

No depth is silently dropped. `n_nonfinite` counts runs with non-finite
loss or accuracy; `n_at_or_below_chance` counts runs that never learned.

| method | depth | n_runs | n_nonfinite | n_at_or_below_chance | best_test_acc | max_T_run |
|---|---|---|---|---|---|---|
| bp | 4 | 5 | 0 | 0 | 0.9053 | 0 |
| bp | 8 | 5 | 0 | 0 | 0.8983 | 0 |
| bp | 16 | 5 | 0 | 0 | 0.8887 | 0 |
| bp | 32 | 5 | 0 | 0 | 0.8783 | 0 |
| bp | 64 | 5 | 0 | 0 | 0.8539 | 0 |
| bp | 128 | 5 | 0 | 0 | 0.8393 | 0 |
| pc | 4 | 40 | 0 | 0 | 0.9041 | 16 |
| pc | 8 | 50 | 0 | 0 | 0.8974 | 32 |
| pc | 16 | 60 | 0 | 0 | 0.8809 | 64 |
| pc | 32 | 90 | 0 | 0 | 0.8605 | 512 |
| pc | 64 | 100 | 0 | 0 | 0.8437 | 1024 |
| pc | 128 | 115 | 0 | 0 | 0.8081 | 3072 |
| pcalm | 4 | 40 | 0 | 0 | 0.9049 | 16 |
| pcalm | 8 | 50 | 0 | 0 | 0.897 | 32 |
| pcalm | 16 | 60 | 0 | 0 | 0.8834 | 64 |
| pcalm | 32 | 90 | 0 | 0 | 0.8725 | 512 |
| pcalm | 64 | 100 | 0 | 0 | 0.8474 | 1024 |
| pcalm | 128 | 100 | 0 | 0 | 0.8312 | 1024 |

## Plain-language summary of the slopes

What the exponent means: T ~ depth^slope, so slope 1 means doubling depth
doubles the inner iterations needed; slope 2 means it quadruples them.
No judgement is offered here about whether any slope is good.

- **At 90% of BP accuracy:**
  - PC: slope +1.916 (95% CI [+1.782, +2.050], R2=0.997)
  - PCALM: slope +1.269 (95% CI [+1.177, +1.361], R2=0.997)
- **At 95% of BP accuracy:**
  - PC: slope +2.067 (95% CI [+1.863, +2.271], R2=0.995)
  - PCALM: slope +1.252 (95% CI [+1.114, +1.390], R2=0.994)
- **At 98% of BP accuracy:**
  - PC: slope +2.010 (95% CI [+1.977, +2.043], R2=1.000), 6 censored cell(s)
  - PCALM: slope +1.421 (95% CI [+1.370, +1.472], R2=0.999)

In words: PC's inner-iteration requirement grows about quadratically with
depth (slope near 2); PC-ALM's grows about linearly, with an exponent near
1.25 rather than exactly 1. The two confidence intervals are far apart at
every threshold. Metric (a) yields no exponent for PC at any tolerance
because its residual never decreases (see caveat 1); for PC-ALM metric (a)
is roughly flat in depth, and its seed spread is large enough that the
depth trend there is not separable from seed noise.

**Depth 256 was dropped from both task metrics.** Its budget ladder alone
costs ~36 h of serial CPU (~5 h even at 7x parallelism) against ~2.5 h for
depths 4-128 combined. It is retained in the residual measurement (a),
where it needs one training run per cell rather than a full ladder.


---

## Known weakness in this workflow: numbers are typed, not generated

Every fitted value in the write-up is a hand-typed literal. Nothing connects a number
in the `.tex` to the data it came from, so a value that was correct against an earlier
state of the pipeline can silently stop being correct.

This is not hypothetical. One restricted-depth PC-ALM fit once read
`+1.200 [1.008, 1.392]` where the raw data gives `+1.258 [1.100, 1.417]` (1.25850); every other
fit in that table reproduced to three or four decimals. It was caught by recomputing
the whole table from `results/t_target_bp_relative.csv`, which is the only way this
class of error can be caught: prose review cannot falsify a fitted value, only its
inputs can.

All eight rows of that table have since been re-derived from raw data and match, and
the same check is re-run after any edit that touches a number.

**The fix.** `analysis/summary_numbers.py` recomputes every fitted exponent and summary
statistic from `results/`, one function per group of results, and with `--check` compares
each against its reference value at the precision it is reported:

    .venv/bin/python analysis/summary_numbers.py --check

A stale value is then a MISMATCH line rather than a silent error. Its first run found
thirteen, all corrected: rounding slips, fits quoted over a different depth range than
stated, one censoring rule applied inconsistently, and a handful of values whose inputs
had never been saved (now regenerated: `verify_linear.csv`, `output_curvature.csv`,
`timing.csv` under `results/strengthen/`).


---

## Additional experiments

All scripts are external callers; `pcalm/` is unmodified. Run everything with the
project virtualenv.

### Residual-connection strength
`analysis/skip_strength.py` -> `results/strengthen/skip_strength.jsonl`. Block
`z_i = s_i f_i(z_{i-1}) + c z_{i-1}`, c in {0, 0.25, 0.5, 0.75, 1}, 6 depths x 3 seeds.
kappa exponent: c=1 `+2.036 [+1.926, +2.146]`; every c<1 flat or falling. Every c<1
network underflows float32 in the forward pass and is at chance under backpropagation.

### Spectra to depth 512
`results/strengthen/kappa_512.csv` (float64). Fit over L=4-512: `L^2.015 [1.973, 2.058]`.

### Hopfield energy
`analysis/kappa_family.py`, `analysis/hopfield_chain.py`. Activity Hessian I - (C + C^T).
  * residual connection carried in as a unit coupling (`kappa_hopfield.csv`): indefinite
    at every depth 4-128, lambda_max + lambda_min = 2.00.
  * on the orthogonal no-skip networks that carry a signal and train
    (`kappa_hopfield_orth.csv`): indefinite at every depth; for the linear one
    lambda_min = 1 - 2cos(pi/(n+1)).
  * on the reference weights without the skip (`kappa_hopfield_noskip.csv`): positive
    definite, kappa FALLING from 35 at L=4 to 1.5 at L=128, like no-skip PC on the same
    weights, whose forward signal has underflowed by L=32.
  * linear chain of orthogonal layers with gain g, exact: eigenvalues
    1 - 2g cos(k pi/(n+1)); indefinite above g ~ 1/2, geometrically decaying response
    below it, and at g = 1/2 half the path Laplacian, kappa = cot^2(pi/(2n+2)) = Theta(n^2)
    (`hopfield_chain.csv`).
So the Hopfield energy is indefinite wherever a signal crosses depth.

### Slow-mode smoothness
`analysis/eigvec_smoothness.py` -> `results/strengthen/eigvec_smoothness.jsonl`.
Envelope roughness ~ L^-0.871; linear-interpolation error 0.036 at L=128.

### Inner-solve cost: steepest descent, Nesterov, multigrid
`analysis/multigrid_pc.py`, `analysis/run_multigrid.py` ->
`results/strengthen/multigrid/`. L=8-256, three seeds, cost to 99% of the achievable
energy reduction. Rounds charge a level-k operation 2^k hops and each exact line search a
global reduction, 2(n-1) hops; Nesterov (fixed step) needs none.

  steepest descent   work   L^1.792 [1.699, 1.885]   rounds L^2.825 [2.733, 2.918]
  Nesterov           work = rounds   L^0.964 [0.829, 1.100]
  multigrid          work   L^1.289 [0.729, 1.849]   rounds L^2.428 [1.753, 3.102]

The energy is piecewise quadratic and not convex away from the forward pass, and no
solver is monotone near its minimum (`analysis/estar_wander.py`: steepest descent reaches
its lowest point partway through and then wanders above it). E* is therefore the lowest
energy any iterate of any reference run (sd, nag, mg) reached. 20k and 80k reference
budgets agree to < 1e-4 of the gap in 17 of 18 seed-depth cells; in the last (L=64, seed
2) the longer run finds a deeper basin and 80k agrees with 320k instead
(`multigrid/checks/`). A depth where a seed ends in a higher basin is censored for that
solver: steepest descent at L=64, multigrid at L=128. Multigrid costs 5-17x Nesterov's
work at every uncensored depth. Gradient-norm tolerances are degenerate on this energy
(the minimiser sits on a ReLU kink); use energy against a verified minimum.

### Weight-gradient budget of the inner solve
`analysis/gradient_tax.py` -> `results/strengthen/gradient_tax_{nag,gd}.jsonl`. Iterations
until the weight gradient stays within 10% of its converged value over [t, 2t], at init,
batch 64; each layer group's reference checked converged to < delta/3.
  Nesterov, L=4-512: whole network `L^1.002 [0.906, 1.098]`, t/sqrt(kappa) 2.0-3.1;
  input layer `L^1.059`, output-side hidden quarter `L^1.147`, read-out `L^1.207`
  (the last two fall to 1.09 and 1.08 over L=16-512).
  Gradient descent, L=4-64: `L^1.985 [1.722, 2.249]`.

### Conditioning at trained weights
`analysis/kappa_trained.py`, `analysis/kappa_trained_traj.py` ->
`results/strengthen/kappa_trained*.csv`. PC-ALM trained by the reference pipeline;
kappa at 0/10/25/50/75/100% of the epoch. Depth exponent over L=4-256: 2.024 at init,
2.04-2.06 at every later stage, 2.051 [1.943, 2.159] at the end; per-depth growth 1.1-1.5x.

### CIFAR-10
`analysis/cifar_data.py`, `analysis/run_cifar.py` -> `results/strengthen/cifar/`.
PC `L^2.000` (exact, 5 depths), PC-ALM `L^1.258 [1.018, 1.499]`; every budget above the
L-2 floor. With the +/-0.066 ladder systematic this does not exclude linear scaling.

### Conditioning versus signal distortion
`analysis/isometry_kappa.py`, `analysis/isometry_train.py` ->
`results/strengthen/isometry_*.jsonl`. At the forward-pass point H = A^T A + B exactly.
For any direction u with distortion D_u, kappa * D_u^2 >= n^2 / (pi^2 (1 + 2 beta/n));
holds in all 132 positive-definite cells across 8 networks. Orthogonal linear network
without a skip: kappa equals the closed-form Laplacian value (1604.9 at L=32, all seeds)
and trains at every depth. The upper bound kappa <= ((1+J_max)^2+|B|) M^2 (n+1/2)^2
needs B positive semidefinite; it fails only in 7 orthogonal-tanh cells, all with
indefinite B.

### Depth 256, PC
`results/d256_ext/`. All three seeds reach 90% of depth-matched BP at T=12288 and stay
above threshold at T=16384. PC fit over L=4-256: `+2.000  [+1.848, +2.152]  n=33/7d`.
