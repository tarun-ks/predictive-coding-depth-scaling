"""Recompute every fitted exponent and summary statistic from results/.

No computed value is typed in: each is recomputed from the saved per-run data, using
the estimators the other analysis scripts define (analysis/report.py for the
cluster-corrected log-log fit, analysis/refine_ttarget.py for the hold-window
T_target, and so on). One function per group of results; each returns a dict of named
values and prints them. Groups whose data is still being regenerated are placeholders
that raise NotImplementedError.

    .venv/bin/python analysis/summary_numbers.py            # print every group
    .venv/bin/python analysis/summary_numbers.py --check    # compare with EXPECTED

--check compares each computed value with a reference value at the precision the
reference is written in, and prints MATCH or MISMATCH per value.
"""
from __future__ import annotations
import argparse, contextlib, csv, io, json, math, os, re, sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import stats

from analysis.report import loglog_fit, load_sweep, build_task, build_conv2, conv_b_dependence
from analysis.refine_ttarget import t_target, bp_accuracy, load_curves
from analysis import robust
from analysis.eta_table import CALIBRATION, eta_for_depths
from analysis.run_momentum import spectrum_by_depth
from analysis.momentum_pc import beta_nag

HOLD = 2.0                              # hold factor W of the T_target definition
FRAC = 0.90                             # headline threshold: 90% of depth-matched BP
DEPTHS7 = (4, 8, 16, 32, 64, 128, 256)
DEPTHS6 = (4, 8, 16, 32, 64, 128)
FROZEN = (8, 16, 32, 64, 128)           # depths covered by configs/eta_best_by_cell.csv
ROBUST_DEPTHS = (8, 16, 32, 64)
CHANCE_BAR = 0.15                       # 10 classes; the majority class is 11.35% of test
LADDER_RATIO = 1.4                      # nominal ratio of the geometric budget ladder
KAPPA_F32 = ("kappa/standard_w32.csv", "kappa/mup_256.csv")
KAPPA_F64 = ("strengthen/kappa_512.csv",)


# ----------------------------------------------------------------------------- helpers

@contextlib.contextmanager
def _at_root():
    """Several imported helpers read results/ through relative paths."""
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def _import_quietly(name):
    """Import a module whose top level runs an analysis and prints it."""
    with _at_root(), contextlib.redirect_stdout(io.StringIO()):
        return __import__(name, fromlist=["_"])


@lru_cache(maxsize=None)
def _jsonl(pattern):
    return tuple(json.loads(l) for f in sorted(RES.glob(pattern))
                 for l in f.read_text().splitlines() if l.strip())


@lru_cache(maxsize=None)
def _bp():
    """BP accuracy per (depth, seed): the main sweep plus the depth-256 runs."""
    with _at_root():
        return bp_accuracy()


@lru_cache(maxsize=None)
def _sweep():
    return load_sweep()


def _curves(dirs, method, variant=None, seeds=None):
    """budget -> test_acc per (depth, seed), merged across result directories."""
    cur = {}
    for d in dirs:
        for r in _jsonl(f"{d}/L*_s*.jsonl"):
            if r.get("method") != method or (variant and r.get("variant") != variant):
                continue
            if (seeds is not None and r["seed"] not in seeds) or not r.get("finite", True):
                continue
            cur.setdefault((r["depth"], r["seed"]), {})[r["budget"]] = r["test_acc"]
    return cur


def _rel(frac):
    """Threshold at frac x the BP accuracy of the same depth and seed."""
    bp = _bp()
    return lambda L, s: frac * bp[(L, s)] if (L, s) in bp else None


def _t_targets(cur, thr, hold=HOLD, depths=None):
    """T_target per depth (one entry per seed) and censored seeds per depth.
    thr(L, s) returns the accuracy threshold, or None to skip the cell."""
    det, cens = {}, {}
    for (L, s), c in sorted(cur.items()):
        if depths is not None and L not in depths:
            continue
        tau = thr(L, s)
        if tau is None:
            continue
        t = t_target(c, tau, hold)
        if t is None:
            cens[L] = cens.get(L, 0) + 1
        else:
            det.setdefault(L, []).append(t)
    return det, cens


def _fit(det, cens=None, depths=None, keep_censored=False):
    """Cluster-corrected log-log fit. A depth with any censored seed is dropped unless
    keep_censored, in which case only its uncensored seeds enter."""
    cens = cens or {}
    Ls = sorted(L for L in det if (keep_censored or not cens.get(L))
                and (depths is None or L in depths))
    return loglog_fit([L for L in Ls for _ in det[L]], [t for L in Ls for t in det[L]])


def _fit_log_of_means(depths, values):
    """Same inference as loglog_fit, but on log(arithmetic mean) per depth."""
    d, v = np.asarray(depths, float), np.asarray(values, float)
    Ls = np.unique(d)
    y = np.log([v[d == L].mean() for L in Ls])
    r = stats.linregress(np.log(Ls), y)
    half = stats.t.ppf(0.975, len(Ls) - 2) * r.stderr
    return dict(slope=r.slope, ci_lo=r.slope - half, ci_hi=r.slope + half,
                r2=r.rvalue ** 2, n_depths=len(Ls), exact_fit=False)


def _fmt(f):
    if f is None:
        return "n/a"
    if f["exact_fit"]:
        return f"{f['slope']:+.3f} (exact, residuals zero), {f['n_depths']} depths"
    return (f"{f['slope']:+.3f} [{f['ci_lo']:.3f}, {f['ci_hi']:.3f}], "
            f"R2={f['r2']:.4f}, {f['n_depths']} depths")


def _put(out, key, f):
    out[key] = f["slope"]
    out[key + "_lo"], out[key + "_hi"] = f["ci_lo"], f["ci_hi"]
    out[key + "_r2"], out[key + "_exact"] = f["r2"], bool(f["exact_fit"])


def _say(quiet, title):
    if quiet:
        return lambda *a: None
    print(f"\n== {title}")
    return lambda label, text: print(f"  {label}: {text}")


def _mean(v):
    return float(np.mean(v))


def _gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def _kappa_rows(files):
    rows = []
    for f in files:
        for r in csv.DictReader(open(RES / f)):
            # some spectrum logs carry a solver traceback after the data rows
            if (r.get("depth") or "").strip().isdigit() and \
                    (r.get("kappa") or "").strip() not in ("", "nan"):
                rows.append({k: float(r[k]) for k in ("depth", "lambda_max", "lambda_min", "kappa")})
    return rows


def _kappa_isometric(L):
    """Closed-form condition number of an isometric chain with n = L - 1 free layers."""
    n = L - 1
    return math.cos(math.pi / (2 * n + 1)) ** 2 / math.sin(math.pi / (4 * n + 2)) ** 2


def _endpoint_limit(depths, ratio):
    """Slope change from moving one endpoint of a log-log fit by one ladder rung."""
    x = np.log(np.asarray(depths, float))
    dev = x - x.mean()
    return math.log(ratio) * float(np.abs(dev).max()) / float(np.sum(dev ** 2))


def _rung(t, ladder=robust.LADDER):
    return min(range(len(ladder)), key=lambda i: abs(math.log(ladder[i]) - math.log(t)))


# ------------------------------------------------------------------ spectrum at init

def kappa_spectrum(quiet=False):
    """Condition number, lambda_min and lambda_max of the activity Hessian at init."""
    say, out = _say(quiet, "activity-Hessian spectrum at initialisation"), {}
    rows = _kappa_rows(KAPPA_F32)
    d = [r["depth"] for r in rows]
    fits = {k: loglog_fit(d, [r[k] for r in rows]) for k in ("kappa", "lambda_min", "lambda_max")}
    _put(out, "kappa", fits["kappa"])
    say("kappa vs depth at init, float32, L=4-256", _fmt(fits["kappa"]))
    rows64 = rows + _kappa_rows(KAPPA_F64)
    f = loglog_fit([r["depth"] for r in rows64], [r["kappa"] for r in rows64])
    _put(out, "kappa512", f)
    say("kappa vs depth, adding float64 L=384, 512", _fmt(f))
    _put(out, "lmin", fits["lambda_min"])
    say("lambda_min vs depth, L=4-256", _fmt(fits["lambda_min"]))
    _put(out, "lmax", fits["lambda_max"])
    say("lambda_max vs depth, L=4-256", _fmt(fits["lambda_max"]))
    f = loglog_fit(d, [r["lambda_max"] - 4.0 for r in rows])
    out["lmax_excess"] = f["slope"]
    say("(lambda_max - 4) vs depth, L=4-256", _fmt(f))

    per = {}
    for r in rows64:
        per.setdefault(int(r["depth"]), []).append(r)
    lmax_means = [_mean([r["lambda_max"] for r in per[L]]) for L in DEPTHS7]
    out["lmax_L4"], out["lmax_L256"] = lmax_means[0], lmax_means[-1]
    out["lmax_monotone"] = bool(np.all(np.diff(lmax_means) < 0))
    out["lmax_above_4_all"] = bool(min(_mean([r["lambda_max"] for r in v]) for v in per.values()) > 4)
    say("lambda_max depth means, L=4 -> 256",
        f"{lmax_means[0]:.3f} -> {lmax_means[-1]:.3f}; monotone={out['lmax_monotone']}, "
        f"above 4 at every depth to 512={out['lmax_above_4_all']}")
    for L in sorted(per):
        v = per[L]
        sd = lambda k: float(np.std([r[k] for r in v], ddof=1))
        lm, ls = _mean([r["lambda_min"] for r in v]), sd("lambda_min")
        e = math.floor(math.log10(lm))
        out[f"t_L{L}_lmax"], out[f"t_L{L}_lmax_sd"] = _mean([r["lambda_max"] for r in v]), sd("lambda_max")
        out[f"t_L{L}_lmin_m"], out[f"t_L{L}_lmin_sd_m"] = lm / 10 ** e, ls / 10 ** e
        out[f"t_L{L}_kappa"], out[f"t_L{L}_kappa_sd"] = _mean([r["kappa"] for r in v]), sd("kappa")
        say(f"L={L:3d}, {len(v)} seeds, mean +/- SD",
            f"lambda_max {out[f't_L{L}_lmax']:.3f} +/- {out[f't_L{L}_lmax_sd']:.3f}, "
            f"lambda_min ({lm / 10 ** e:.2f} +/- {ls / 10 ** e:.2f})e{e}, "
            f"kappa {out[f't_L{L}_kappa']:.0f} +/- {out[f't_L{L}_kappa_sd']:.0f}")

    s8 = list(csv.DictReader(open(RES / "kappa/mup_w32_samples.csv")))
    f = loglog_fit([int(r["depth"]) for r in s8], [float(r["kappa"]) for r in s8])
    _put(out, "kappa_8inputs", f)
    say("kappa vs depth, 8 inputs x 3 seeds per depth, L=4-256", _fmt(f))
    conv = _kappa_rows(("kappa/conv.csv",))
    f = loglog_fit([r["depth"] for r in conv], [r["kappa"] for r in conv])
    _put(out, "kappa_conv", f)
    say("kappa vs depth, conv stack, L=4-32", _fmt(f))
    dense = [r for r in rows if r["depth"] <= 32]
    f = loglog_fit([r["depth"] for r in dense], [r["kappa"] for r in dense])
    _put(out, "kappa_dense_4_32", f)
    say("kappa vs depth, dense network, same depths L=4-32", _fmt(f))

    E = _jsonl("strengthen/eigvec_smoothness.jsonl")
    out["eigvec_interp_L128"] = _mean([r["linear_interp_rel_error"] for r in E if r["depth"] == 128])
    say("slow eigenvector, error of every-other-layer linear interpolation, L=128",
        f"{out['eigvec_interp_L128']:.4f}")

    sp = {}
    for r in _kappa_rows(("kappa/sp_w32.csv",)):
        sp.setdefault(int(r["depth"]), []).append(r["kappa"])
    Ls = sorted(sp)
    ratios = [math.log2(_mean(sp[b]) / _mean(sp[a])) for a, b in zip(Ls, Ls[1:])]
    for i, rr in enumerate(ratios):
        out[f"sp_log2_ratio_{i + 1}"] = rr
    say(f"standard parameterisation, log2 kappa ratio per doubling, L={Ls}",
        ", ".join(f"{r:.2f}" for r in ratios))
    return out


def step_size_offsets(quiet=False):
    """Spectral lambda_max estimate against the frozen per-cell step sizes."""
    say, out = _say(quiet, "step size: spectral lambda_max vs frozen eta_h"), {}
    with _at_root():
        eta, _ = eta_for_depths(list(FROZEN))
    lmax = {}
    for r in _kappa_rows(KAPPA_F32):
        lmax.setdefault(int(r["depth"]), []).append(r["lambda_max"])
    off = [100.0 * (float(np.median(lmax[L])) * eta[L] - 1.0) for L in FROZEN]
    tr = stats.linregress(np.log(FROZEN), off)
    out.update(off_min=min(off), off_max=max(off), off_mean=_mean(off),
               off_sd=float(np.std(off, ddof=1)), off_trend_p=tr.pvalue)
    say("median-over-seeds lambda_max x frozen eta_h - 1, L=8-128 (%)",
        ", ".join(f"{o:.2f}" for o in off))
    say("range / mean / SD / trend in log L",
        f"{out['off_min']:.1f}% to {out['off_max']:.1f}%, mean {out['off_mean']:.1f}%, "
        f"SD {out['off_sd']:.1f}%, p = {out['off_trend_p']:.2f}")
    return out


# ----------------------------------------------------- skip strength and no-skip arms

def skip_strength(quiet=False):
    """Residual connection of strength c: conditioning, forward signal, trainability."""
    say, out = _say(quiet, "skip strength c in z_i = s_i f_i(z_{i-1}) + c z_{i-1}"), {}
    R = _jsonl("strengthen/skip_strength.jsonl")
    cs = sorted({r["skip_strength"] for r in R})
    under, chance = {}, {}
    for c in cs:
        rs = [r for r in R if r["skip_strength"] == c]
        f = loglog_fit([r["depth"] for r in rs], [r["kappa"] for r in rs])
        by = {}
        for r in rs:
            by.setdefault(r["depth"], []).append(r)
        under[c] = next((L for L in sorted(by) if all(r["act_rms_last"] == 0 for r in by[L])), None)
        chance[c] = next((L for L in sorted(by)
                          if _mean([r["bp_test_acc"] for r in by[L]]) < CHANCE_BAR), None)
        out[f"c{c:g}_slope"] = f["slope"]
        out[f"underflow_c{c:g}"] = under[c]
        say(f"c={c:g}: kappa vs depth, L=4-128", _fmt(f))
        say(f"c={c:g}: first depth with exact float32 underflow / BP below {CHANCE_BAR:.0%}",
            f"{under[c]} / {chance[c]}")
    rs = [r for r in R if r["skip_strength"] == 1.0]
    _put(out, "c1", loglog_fit([r["depth"] for r in rs], [r["kappa"] for r in rs]))
    f = _fit_log_of_means([r["depth"] for r in rs], [r["kappa"] for r in rs])
    out["c1_logmean_lo"], out["c1_logmean_hi"] = f["ci_lo"], f["ci_hi"]
    say("c=1 with log(arithmetic mean kappa) per depth instead", _fmt(f))
    weak = [c for c in cs if c <= 0.5]
    out["underflow_by_L_weak"] = max(under[c] or 10 ** 9 for c in weak)
    out["chance_by_L_weak"] = max(chance[c] or 10 ** 9 for c in weak)
    out["chance_by_L_all_below_1"] = max(chance[c] or 10 ** 9 for c in cs if c < 1)
    out["bp_c1_L128"] = 100 * _mean([r["bp_test_acc"] for r in rs if r["depth"] == 128])
    say("depth by which every c <= 0.5 has underflowed / is at chance",
        f"{out['underflow_by_L_weak']} / {out['chance_by_L_weak']}")
    say("depth by which every c < 1 is at chance", out["chance_by_L_all_below_1"])
    return out


def isometry(quiet=False):
    """No-skip arms and the two forward-pass bounds, 8 arms x 6 depths x 3 seeds."""
    say, out = _say(quiet, "no-skip arms and the forward-pass bounds"), {}
    R = [r for f in ("isometry_a", "isometry_b", "isometry_relu")
         for r in _jsonl(f"strengthen/{f}.jsonl")]
    T = _jsonl("strengthen/isometry_train.jsonl")
    arm = lambda r: (r["arm"], r["act"], r["gain"])
    pd = [r for r in R if r["kappa"] is not None and np.isfinite(r["kappa"]) and r["lambda_min"] > 0]
    # Theorem 1 with the test vector sin(pi (k-1)/(n-1)), which vanishes at both ends so the
    # output block drops out: kappa D_u^2 >= (n-1)^2 / pi^2 for n >= 3
    lower = lambda r: (r["n"] - 1) ** 2 / math.pi ** 2
    ok1 = [r for r in pd if r["kappa"] * r["distortion"] ** 2 >= lower(r) * (1 - 1e-9)]
    bad2 = [r for r in pd if r["kappa"] > r["kappa_upper"] * (1 + 1e-6)]
    out.update(n_arms=len({arm(r) for r in R}), n_cells=len(R), n_pd=len(pd),
               lower_holds=len(ok1), upper_fails=len(bad2),
               upper_fail_arms=sorted({f"{a}:{b}:{g:g}" for a, b, g in map(arm, bad2)}))
    say("arms / cells / positive-definite cells", f"{out['n_arms']} / {out['n_cells']} / {out['n_pd']}")
    say("kappa D_u^2 >= (n-1)^2 / pi^2 holds in",
        f"{out['lower_holds']} of {out['n_pd']} positive-definite cells")
    say("upper bound exceeded in", f"{out['upper_fails']} cells, arms {out['upper_fail_arms']} "
        "(lambda_min of the output block is not stored, so B >= 0 cannot be checked here)")
    lin32 = sorted((r["seed"], r["kappa"]) for r in R
                   if arm(r) == ("orth", "linear", 1.0) and r["depth"] == 32)
    for s, k in lin32:
        out[f"lin_L32_s{s}"] = k
    out["lin_L32_closed"] = _kappa_isometric(32)
    say("orthogonal linear, L=32, kappa per seed vs closed form",
        ", ".join(f"{k:.3f}" for _, k in lin32) + f" vs {out['lin_L32_closed']:.3f}")
    tanh = [r for r in pd if arm(r) == ("orth", "tanh", 1.0)]
    f = loglog_fit([r["depth"] for r in tanh], [r["kappa"] for r in tanh])
    _put(out, "tanh", f)
    say("orthogonal tanh, no skip, kappa vs depth, L=4-128", _fmt(f))
    con = [r for r in R if arm(r) == ("orth", "tanh", 0.8) and r["depth"] == 128]
    out["contract_kappa_L128"] = _mean([r["kappa"] for r in con])
    out["contract_cap"] = max(((1 + r["J_max"]) ** 2 + r["beta"]) / (1 - r["J_max"]) ** 2 for r in con)
    say("contractive tanh |J|=0.8, L=128: kappa / proved cap",
        f"{out['contract_kappa_L128']:.2f} / {out['contract_cap']:.2f}")
    ind = sorted({r["depth"] for r in R if arm(r) == ("orth", "tanh", 1.2) and r not in pd})
    out["gain12_indefinite_from"] = ind[0] if ind else None
    say("orthogonal tanh gain 1.2: indefinite at depths", ind)
    acc = lambda a, L: 100 * _mean([r["bp_test_acc"] for r in T if arm(r) == a and r["depth"] == L])
    out["bp_lin_L128"] = acc(("orth", "linear", 1.0), 128)
    out["bp_tanh_L64"] = acc(("orth", "tanh", 1.0), 64)
    out["bp_ref_L64"] = acc(("reference", "relu", 1.0), 64)
    say("BP after one epoch: orth linear L=128 / orth tanh L=64 / reference L=64 (%)",
        f"{out['bp_lin_L128']:.2f} / {out['bp_tanh_L64']:.2f} / {out['bp_ref_L64']:.2f}")
    return out


# --------------------------------------------------------------- training-side budgets

def budget_ladder(quiet=False):
    """T_target at 90% of depth-matched BP, per depth and method."""
    say, out = _say(quiet, "T_target at 90% of depth-matched BP, per depth"), {}
    bp = _bp()
    for L in DEPTHS7:
        v = [100 * bp[(L, s)] for s in range(5) if (L, s) in bp]
        out[f"bp_L{L}"], out[f"bp_L{L}_sd"] = _mean(v), float(np.std(v, ddof=1))
    say("BP accuracy mean +/- SD (%)", ", ".join(
        f"L={L}: {out[f'bp_L{L}']:.2f} +/- {out[f'bp_L{L}_sd']:.2f}" for L in DEPTHS7))
    arms = {
        "pc": _curves(("sweep", "d256", "d256_ext"), "pc"),
        "alm": _curves(("sweep", "d256"), "pcalm"),
        "nag": _curves(("momentum", "momentum_fine"), "pc", "nag"),
    }
    with _at_root():
        arms["ref"] = load_curves(set(DEPTHS7), "pcalm")
    for name, cur in arms.items():
        det, cens = _t_targets(cur, _rel(FRAC))
        for L in sorted(det):
            out[f"{name}_L{L}_min"], out[f"{name}_L{L}_max"] = min(det[L]), max(det[L])
        say(f"{name}: T_target per seed", {L: sorted(v) for L, v in sorted(det.items())}
            | ({"censored": cens} if cens else {}))
    first, _ = _t_targets(_curves(("momentum",), "pc", "hb"), _rel(FRAC), hold=1.0)
    held, hcens = _t_targets(_curves(("momentum",), "pc", "hb"), _rel(FRAC))
    for L in sorted(first):
        out[f"hb_L{L}_first"] = max(first[L])
    for L in sorted(held):
        out[f"hb_L{L}_min"], out[f"hb_L{L}_max"] = min(held[L]), max(held[L])
    out["hb_depths_run"] = sorted(first)
    out["hb_hold_censored_depths"] = sorted(hcens)
    say("heavy-ball, first crossing", {L: sorted(v) for L, v in sorted(first.items())})
    say("heavy-ball, hold window", {L: sorted(v) for L, v in sorted(held.items())}
        | {"censored": hcens})
    return out


def threshold_fits(quiet=False):
    """Coarse-ladder exponents at 90/95/98% of BP, L=4-128 (estimator of report.py)."""
    say, out = _say(quiet, "coarse ladder, L=4-128, first crossing, per-run censoring"), {}
    _, tgt = build_task(list(_sweep()))
    for fr in (0.90, 0.95, 0.98):
        for m in ("pc", "pcalm"):
            pts = [(r["depth"], r["T_target"]) for r in tgt if r["frac"] == fr
                   and r["method"] == m and r["T_target"] is not None and not r["censored"]]
            f = loglog_fit([p[0] for p in pts], [p[1] for p in pts])
            _put(out, f"{m}_{int(fr * 100)}", f)
            say(f"{m} at {fr:.0%} of BP", _fmt(f))
    for fr in (0.95, 0.98):
        for m in ("pc", "pcalm"):
            det, cens = _t_targets(_curves(("sweep",), m), _rel(fr))
            f = _fit(det, cens)
            _put(out, f"{m}_{int(fr * 100)}_strict", f)
            out[f"{m}_{int(fr * 100)}_strict_depths"] = f["n_depths"] if f else 0
            say(f"{m} at {fr:.0%}, hold window, depths with a censored seed dropped",
                f"{_fmt(f)}; censored seeds {cens}")
    return out


def headline_fits(quiet=False):
    """Depth exponents of the headline budgets, coarse and refined ladders."""
    say, out = _say(quiet, "depth exponents of T_target at 90% of BP"), {}
    # PC at L=256 reaches the threshold only on the extended ladder in results/d256_ext,
    # which refine_ttarget.load_curves does not read; without it one seed is censored.
    pc = _t_targets(_curves(("sweep", "d256", "d256_ext"), "pc"), _rel(FRAC))
    alm = _t_targets(_curves(("sweep", "d256"), "pcalm"), _rel(FRAC))
    with _at_root():
        ref = _t_targets(load_curves(set(DEPTHS7), "pcalm"), _rel(FRAC))
    rows = [("pc7", pc, DEPTHS7, "PC, L=4-256"),
            ("pc6", pc, DEPTHS6, "PC, L=4-128"),
            ("pc_frozen", pc, FROZEN, "PC, L=8-128 (frozen-eta depths)"),
            ("alm7", alm, DEPTHS7, "PC-ALM coarse, L=4-256"),
            ("alm_frozen", alm, FROZEN, "PC-ALM coarse, L=8-128 (frozen-eta depths)"),
            ("ref7", ref, DEPTHS7, "PC-ALM refined, L=4-256"),
            ("ref6", ref, DEPTHS6, "PC-ALM refined, L=4-128")]
    for key, (det, cens), depths, label in rows:
        f = _fit(det, cens, depths)
        _put(out, key, f)
        say(label, _fmt(f))
    out["pc_L256_T"] = pc[0][256]
    out["ref_L128_T"] = sorted(ref[0][128])
    out["alm_L128_T"] = sorted(alm[0][128])
    say("PC at L=256, per seed", out["pc_L256_T"])
    say("PC-ALM at L=128, refined / coarse", f"{out['ref_L128_T']} / {out['alm_L128_T']}")
    out["alm7_width"] = out["alm7_hi"] - out["alm7_lo"]
    out["ref7_width"] = out["ref7_hi"] - out["ref7_lo"]
    out["ref7_half"] = out["ref7_width"] / 2
    say("CI width, coarse -> refined, L=4-256",
        f"{out['alm7_width']:.3f} -> {out['ref7_width']:.3f} (half-width {out['ref7_half']:.3f})")
    for name, (det, _) in (("pc", pc), ("alm", alm), ("ref", ref)):
        r = {L: _mean(det[L]) / (L - 1) for L in det}      # floor: h_1 first moves at t = L-1
        out[f"{name}_floor_ratio"] = r
        out[f"{name}_floor_min"], out[f"{name}_floor_max"] = min(r.values()), max(r.values())
        say(f"{name}: T_target / (L - 1) per depth", {L: round(v, 2) for L, v in r.items()})
    out["pc_floor_L8"], out["pc_floor_L128"] = out["pc_floor_ratio"][8], out["pc_floor_ratio"][128]
    return out


def grid_systematic(quiet=False):
    """Exponent spread across four budget grids, L=8-64, and its analytic limit."""
    say, out = _say(quiet, "budget-grid systematic, L=8-64, first crossing"), {}
    base = list(robust.load(str(RES / "sweep/*.jsonl")))
    bp_rows = [r for r in base if r["method"] == "bp"]   # BP does not depend on the grid
    slopes = {"pc": [], "pcalm": []}
    for g in ("sweep", "offset", "offset09", "offset30"):
        rows = [r for r in robust.load(str(RES / f"{g}/*.jsonl")) if r["method"] != "bp"]
        tt = robust.t_target(rows + bp_rows, FRAC, set(ROBUST_DEPTHS))
        for m in slopes:
            f = robust.fit_of(tt[m])
            slopes[m].append(f["slope"])
            out[f"{g}_{m}"] = f["slope"]
        say(f"grid {g}", f"PC {out[g + '_pc']:.3f}, PC-ALM {out[g + '_pcalm']:.3f}")
    for m in slopes:
        out[f"{m}_span_lo"], out[f"{m}_span_hi"] = min(slopes[m]), max(slopes[m])
        out[f"{m}_half_range"] = (max(slopes[m]) - min(slopes[m])) / 2
        say(f"{m} across the four grids",
            f"{min(slopes[m]):.3f} to {max(slopes[m]):.3f}, half-range {out[m + '_half_range']:.3f}")
    out["limit_4"] = _endpoint_limit(ROBUST_DEPTHS, LADDER_RATIO)
    out["limit_6"] = _endpoint_limit(DEPTHS6, LADDER_RATIO)
    out["limit_6_ratio15"] = _endpoint_limit(DEPTHS6, 1.5)
    say("one-rung endpoint shift, ratio 1.4: L=8-64 / L=4-128",
        f"{out['limit_4']:.3f} / {out['limit_6']:.3f}")
    say("one-rung endpoint shift, ratio 1.5, L=4-128", f"{out['limit_6_ratio15']:.3f}")
    h = headline_fits(quiet=True)
    out["ref7_lo_net"] = h["ref7_lo"] - out["pcalm_half_range"]
    say("refined PC-ALM lower edge net of the grid systematic", f"{out['ref7_lo_net']:.3f}")
    return out


def log_correction(quiet=False):
    """Free power law against a L (1 + b ln L) on the refined PC-ALM budgets (AICc)."""
    say, out = _say(quiet, "refined PC-ALM budgets: power law vs log-corrected linear"), {}
    M = _import_quietly("analysis.log_correction_fit")
    out.update(p=M.f.slope, a=float(M.p[0]), b=float(M.p[1]),
               aicc_pow=M.aicc(M.r_pow, 2), aicc_log=M.aicc(M.r_log, 2),
               aicc_llnl=M.aicc(M.r_lnl, 1), aicc_lin=M.aicc(M.r_lin, 1))
    out["d_aicc"] = out["aicc_pow"] - out["aicc_log"]
    say("power law p", f"{out['p']:.3f}")
    say("a L (1 + b ln L)", f"a = {out['a']:.3f}, b = {out['b']:.3f}")
    say("AICc power / log-corrected / L ln L / L", " / ".join(
        f"{out[k]:.2f}" for k in ("aicc_pow", "aicc_log", "aicc_llnl", "aicc_lin")))
    say("Delta AICc, power law minus log-corrected linear", f"{out['d_aicc']:.2f}")
    out["pc_control_p"] = headline_fits(quiet=True)["pc7"]
    say("PC control, free power law on its seven-depth budgets", f"{out['pc_control_p']:.3f}")
    return out


def gradient_alignment(quiet=False):
    """Budget for PC-ALM's weight update to reach cosine 0.90 with the BP gradient."""
    say, out = _say(quiet, "gradient-alignment budget at frozen trained parameters"), {}
    cur = {}
    for f in sorted((RES / "item1").glob("L*_s*.npz")):
        L, s = map(int, re.match(r"L(\d+)_s(\d+)", f.name).groups())
        z = np.load(f)
        cur[(L, s)] = {int(t): float(c) for t, c in zip(z["Ts"], z["total"])}
    det, cens = _t_targets(cur, lambda L, s: 0.90)
    f = _fit(det, cens)
    _put(out, "cos90", f)
    out["n_censored"] = sum(cens.values())
    say("first T with cos(g, g_BP) >= 0.90 held to 2T, per seed",
        {L: sorted(v) for L, v in sorted(det.items())})
    say("depth exponent, L=4-128", f"{_fmt(f)}, censored seeds {out['n_censored']}")
    h = headline_fits(quiet=True)
    out["accuracy_budget_same_depths"] = h["ref6"]
    out["accuracy_budget_same_depths_lo"], out["accuracy_budget_same_depths_hi"] = \
        h["ref6_lo"], h["ref6_hi"]
    say("refined accuracy budget over the same depths, L=4-128",
        f"{out['accuracy_budget_same_depths']:.3f}")
    return out


def robustness(quiet=False):
    """Exponent stability across dataset, width, epochs and threshold definition."""
    say, out = _say(quiet, "robustness, L=8-64 unless stated"), {}
    base = list(robust.load(str(RES / "sweep/*.jsonl")))
    conds = {}
    for r in robust.load(str(RES / "robust/*.jsonl")):
        conds.setdefault(robust.key(r), []).append(r)
    names = {("fashion_mnist", 32, 1): "fashion", ("mnist", 16, 1): "w16",
             ("mnist", 64, 1): "w64", ("mnist", 32, 5): "ep5"}
    for k, rows in sorted(conds.items(), key=lambda kv: names.get(kv[0], "")):
        depths = sorted({r["depth"] for r in rows})
        tt, bt = robust.t_target(rows, FRAC, set(depths)), robust.t_target(base, FRAC, set(depths))
        for m, tag in (("pc", "pc"), ("pcalm", "alm")):
            keep = {L for L, _ in tt[m]} & {L for L, _ in bt[m]}
            fc = robust.fit_of([p for p in tt[m] if p[0] in keep])
            fb = robust.fit_of([p for p in bt[m] if p[0] in keep])
            out[f"{names[k]}_{tag}"], out[f"base_{tag}"] = fc["slope"], fb["slope"]
            out[f"{names[k]}_{tag}_exact"] = bool(fc["exact_fit"])
            out[f"base_{tag}_exact"] = bool(fb["exact_fit"])
        say(f"{names[k]} {k}", f"PC {out[names[k] + '_pc']:.3f}"
            f"{' (exact)' if out[names[k] + '_pc_exact'] else ''}, PC-ALM {out[names[k] + '_alm']:.3f}")
    say("baseline mnist/w32/1ep", f"PC {out['base_pc']:.3f}"
        f"{' (exact)' if out['base_pc_exact'] else ''}, PC-ALM {out['base_alm']:.3f}")
    g = grid_systematic(quiet=True)
    out["offset_pc"], out["offset_alm"] = g["offset_pc"], g["offset_pcalm"]
    say("x1.19 offset grid", f"PC {out['offset_pc']:.3f}, PC-ALM {out['offset_alm']:.3f}")

    variants = (("abs8_64", ROBUST_DEPTHS, False, "L=8-64"),
                ("abs4_128_keep", DEPTHS6, True, "L=4-128, partially censored depths kept"),
                ("abs4_128", DEPTHS6, False, "L=4-128, partially censored depths dropped"))
    for key, depths, keep, label in variants:
        for m, tag in (("pc", "pc"), ("pcalm", "alm")):
            vals = []
            for thr in (0.70, 0.75, 0.80):
                det, cens = _t_targets(_curves(("sweep",), m), lambda L, s: thr, hold=1.0,
                                       depths=set(depths))
                vals.append(_fit(det, cens, keep_censored=keep)["slope"])
            out[f"{key}_{tag}_min"], out[f"{key}_{tag}_max"] = min(vals), max(vals)
            say(f"coarse, absolute 70/75/80%, {m}, {label}", ", ".join(f"{v:.3f}" for v in vals))
    with _at_root():
        ref = load_curves(set(DEPTHS7), "pcalm")
    bp = _bp()
    for thr in (0.70, 0.75, 0.80):
        det, cens = _t_targets(ref, lambda L, s: thr, depths=set(DEPTHS6))
        f = _fit(det, cens)
        out[f"ref_abs{int(thr * 100)}"] = f["slope"]
        det7, cens7 = _t_targets(ref, lambda L, s: thr if (L, s) in bp else None)
        f7 = _fit(det7, cens7)
        out[f"ref_abs{int(thr * 100)}_7"] = f7["slope"]
        say(f"refined PC-ALM, absolute {thr:.0%}", f"L=4-128: {_fmt(f)}; "
            f"L=4-256 on the headline seeds: {_fmt(f7)}")
    main_bp = [100 * _mean([bp[(L, s)] for s in range(5)]) for L in DEPTHS6]
    ep5 = conds[("mnist", 32, 5)]
    ep5_bp = [100 * _mean([r["test_acc"] for r in ep5 if r["method"] == "bp" and r["depth"] == L])
              for L in ROBUST_DEPTHS]
    out.update(bp1_min=min(main_bp), bp1_max=max(main_bp), bp5_min=min(ep5_bp),
               bp5_max=max(ep5_bp), bp_decline=100 * (1 - main_bp[-1] / main_bp[0]))
    say("BP one epoch L=4-128 / five epochs L=8-64 (%)",
        f"{out['bp1_min']:.2f}-{out['bp1_max']:.2f} / {out['bp5_min']:.2f}-{out['bp5_max']:.2f}")
    say("relative BP decline L=4 -> 128", f"{out['bp_decline']:.1f}%")
    return out


def cifar(quiet=False):
    """CIFAR-10 sweep, L=4-64, three seeds, and the MNIST-calibrated budget model."""
    say, out = _say(quiet, "CIFAR-10"), {}
    P = _import_quietly("analysis.predict_cifar")
    bp = {}
    for r in _jsonl("strengthen/cifar/L*_s*_bp.jsonl"):
        bp.setdefault(r["depth"], []).append(r["test_acc"])
    out["bp_L4"], out["bp_L128"] = 100 * _mean(bp[4]), 100 * _mean(bp[128])
    out["bp_decline"] = 100 * (1 - _mean(bp[128]) / _mean(bp[4]))
    say("BP accuracy L=4 / L=128 / relative decline (%)",
        f"{out['bp_L4']:.2f} / {out['bp_L128']:.2f} / {out['bp_decline']:.1f}")
    for m, T in (("pc", P.TCpc), ("alm", P.TCalm)):
        f = loglog_fit(list(T), list(T.values()))
        _put(out, m, f)
        out[f"{m}_T"] = [round(T[L]) for L in sorted(T)]
        out[f"{m}_floor_min"] = min(T[L] / (L - 1) for L in T)
        say(f"{m}: T_target (hold window), L=4-64", out[f"{m}_T"])
        say(f"{m}: depth exponent", _fmt(f))
    out["pc_floor_L64"] = P.TCpc[64] / 63
    say("smallest T / (L - 1), PC / PC-ALM; PC at L=64",
        f"{out['pc_floor_min']:.2f} / {out['alm_floor_min']:.2f}; {out['pc_floor_L64']:.1f}")
    out["alm_lo_net"] = out["alm_lo"] - grid_systematic(quiet=True)["pcalm_half_range"]
    say("PC-ALM lower edge net of the grid systematic", f"{out['alm_lo_net']:.3f}")
    # The CIFAR-10 step is 1/lambda_max from a batch power iteration (run_cifar.py, the
    # same estimator as lambda_max.py). Its per-depth MNIST outputs are not saved; the
    # only record is the mean ratio to the frozen step sizes kept in eta_table.py.
    out["batch_estimator_mean_offset"] = 100 * (CALIBRATION - 1)
    out["batch_estimator_within_3p1"] = out["batch_estimator_mean_offset"] <= 3.1
    say("batch lambda_max estimator vs frozen step sizes on MNIST, L=8-128, mean offset",
        f"{out['batch_estimator_mean_offset']:.2f}% (per-depth values not saved)")
    out.update(kappa_dev=100 * P.dev, c_pc=P.c_pc, a_alm=float(np.exp(P.lc)), q_alm=float(P.q))
    say("max |kappa CIFAR / kappa MNIST - 1| over L=4-128", f"{out['kappa_dev']:.1f}%")
    say("MNIST calibration", f"PC T = {out['c_pc']:.3f} kappa; "
        f"PC-ALM T = {out['a_alm']:.3f} kappa^{out['q_alm']:.3f}")
    off_pc = [P.rung(P.TCpc[L]) - P.rung(P.c_pc * P.kC[L]) for L in sorted(P.TCpc)]
    off_alm = [P.rung(P.TCalm[L]) - P.rung(out["a_alm"] * P.kC[L] ** P.q) for L in sorted(P.TCalm)]
    out.update(alm_rungs_exact=sum(o == 0 for o in off_alm),
               alm_rungs_one=sum(abs(o) == 1 for o in off_alm),
               pc_rungs_one_low=sum(o == 1 for o in off_pc))
    say("measured minus predicted rung, PC / PC-ALM", f"{off_pc} / {off_alm}")
    return out


def momentum(quiet=False):
    """Inner solver swapped for momentum on PC's activities, everything else held."""
    say, out = _say(quiet, "momentum on PC's activities"), {}
    rel = _rel(FRAC)
    arms = (("gd", _curves(("momentum",), "pc", "gd")),
            ("nag", _curves(("momentum", "momentum_fine"), "pc", "nag")),
            ("depthbeta", _curves(("momentum_depthbeta",), "pc", "nag", seeds={0, 1, 2})),
            ("depthbeta_all", _curves(("momentum_depthbeta",), "pc", "nag")))
    det = {}
    for name, cur in arms:
        det[name] = _t_targets(cur, rel)
        f = _fit(*det[name])
        _put(out, name, f)
        say(f"{name}, L=4-128", f"{_fmt(f)}; censored seeds {det[name][1]}")
    first = _t_targets(_curves(("momentum",), "pc", "hb"), rel, hold=1.0)
    _put(out, "hb_first", _fit(*first))
    say("heavy-ball, first crossing, L=4-64", _fmt(_fit(*first)))

    same = tot = 0
    spec_set = _curves(("momentum", "momentum_fine"), "pc", "nag")
    for (L, s), c in _curves(("momentum_depthbeta",), "pc", "nag", seeds={0, 1, 2}).items():
        tot += 1
        same += int(t_target(c, rel(L, s), HOLD) == t_target(spec_set[(L, s)], rel(L, s), HOLD))
    out["depthbeta_same"], out["depthbeta_cells"] = same, tot
    say("depth-set vs spectrum-set beta, identical T_target", f"{same} of {tot} seed-depth cells")

    h = headline_fits(quiet=True)
    # like for like: Nesterov was run over six depths (4-128), so compare with PC-ALM's
    # refined fit over the same six depths, not the seven-depth headline
    out["ref6_minus_nag"] = h["ref6"] - out["nag"]
    out["ref6_inside_nag"] = bool(out["nag_lo"] <= h["ref6_lo"] and h["ref6_hi"] <= out["nag_hi"])
    say("refined PC-ALM (L=4-128) minus NAG exponent",
        f"{out['ref6_minus_nag']:+.3f}; PC-ALM interval inside: {out['ref6_inside_nag']}")
    k = kappa_spectrum(quiet=True)["kappa"]
    out["accelerated_prediction"] = k / 2
    say("ideal accelerated exponent, half the kappa exponent", f"{k / 2:.3f}")

    spec = spectrum_by_depth(ROOT)
    ratio = {L: spec[L][2] / _kappa_isometric(L) for L in spec}
    gap = {L: (1 - beta_nag(_kappa_isometric(L))) / (1 - beta_nag(spec[L][2])) for L in spec}
    out.update(kiso_min=min(ratio.values()), kiso_max=max(ratio.values()),
               gap_min=min(gap.values()), gap_max=max(gap.values()))
    say("measured kappa / isometric closed form, L=4-256", {L: round(r, 2) for L, r in ratio.items()})
    say("momentum-gap misspecification sqrt(ratio) / exact 1-beta ratio",
        f"{out['gap_min']:.2f}-{out['gap_max']:.2f} / "
        f"{min(gap.values()):.2f}-{max(gap.values()):.2f}")

    hb64 = {r["budget"]: r["test_acc"] for r in _jsonl("momentum/L64_s0_hb.jsonl")}
    out["hb_L64_T64"], out["hb_L64_T96"] = 100 * hb64[64], 100 * hb64[96]
    say("heavy-ball L=64 seed 0, accuracy at T=64 / T=96 (%)",
        f"{out['hb_L64_T64']:.2f} / {out['hb_L64_T96']:.2f}")

    shift = 0
    coarse = _curves(("momentum",), "pc", "nag")
    for r_L in (32, 64):
        for s in range(3):
            t1 = t_target(coarse[(r_L, s)], rel(r_L, s), HOLD)
            for g in (0.5, 0.75, 1.5, 2.0):
                c = {r["budget"]: r["test_acc"] for r in _jsonl(f"beta_sens/L{r_L}_s{s}_g{g}.jsonl")}
                shift = max(shift, abs(_rung(t_target(c, rel(r_L, s), HOLD)) - _rung(t1)))
    out["beta_gap_max_rung_shift"] = shift
    out["beta_gap_within_one_rung"] = shift <= 1
    say("momentum gap x0.5 to x2, L=32 and 64: largest coarse-rung shift", shift)
    return out


def prefactors(quiet=False):
    """Prefactors anchored at L=128 and their extrapolation to L=1000."""
    say, out = _say(quiet, "prefactors and extrapolation"), {}
    h = headline_fits(quiet=True)
    pcd = _t_targets(_curves(("sweep", "d256", "d256_ext"), "pc"), _rel(FRAC))[0]
    with _at_root():
        refd = _t_targets(load_curves(set(DEPTHS7), "pcalm"), _rel(FRAC))[0]
    pc_T = _mean(pcd[128])
    out["pc_a"] = pc_T / 128 ** h["pc7"]
    out["alm_a"] = _gmean(h["ref_L128_T"]) / 128 ** h["ref7"]
    out["pc_1000"] = out["pc_a"] * 1000 ** h["pc7"]
    out["alm_1000"] = out["alm_a"] * 1000 ** h["ref7"]
    out["ratio_1000"] = out["pc_1000"] / out["alm_1000"]
    # the same extrapolation anchored at L=256 instead (PC's L=256 cell is an unrefined rung)
    out["pc_pred_256"] = out["pc_a"] * 256 ** h["pc7"]
    out["pc_T256"] = _mean(pcd[256])
    out["pc_a256"] = out["pc_T256"] / 256 ** h["pc7"]
    out["alm_a256"] = _mean(refd[256]) / 256 ** h["ref7"]
    out["pc_1000_256"] = out["pc_a256"] * 1000 ** h["pc7"]
    out["alm_1000_256"] = out["alm_a256"] * 1000 ** h["ref7"]
    out["ratio_1000_256"] = out["pc_1000_256"] / out["alm_1000_256"]
    out["vs_2L_256"] = out["alm_1000_256"] / 2000
    say("anchored at L=256", f"PC {out['pc_a256']:.3f}, PC-ALM {out['alm_a256']:.3f}; L=1000: "
        f"{out['pc_1000_256']:.3g} / {out['alm_1000_256']:.3g} / ratio {out['ratio_1000_256']:.1f}; "
        f"L=128 fit predicts {out['pc_pred_256']:.0f} at L=256 against {out['pc_T256']:.0f}")
    out["per_update_pc"], out["per_update_alm"] = 1 + h["pc7"], 1 + h["ref7"]
    out["doubling_alm"] = 2 ** out["per_update_alm"]
    out["vs_2L"] = out["alm_1000"] / 2000
    out["excess_over_linear"] = h["ref7"] - 1
    say("anchored at L=128", f"PC T = {out['pc_a']:.3f} L^{h['pc7']:.3f}; "
        f"PC-ALM T = {out['alm_a']:.3f} L^{h['ref7']:.3f}")
    say("at L=1000, PC / PC-ALM / ratio",
        f"{out['pc_1000']:.3g} / {out['alm_1000']:.3g} / {out['ratio_1000']:.1f}")
    say("per-update cost exponents, PC / PC-ALM",
        f"{out['per_update_pc']:.2f} / {out['per_update_alm']:.2f}")
    say("PC-ALM per-update cost per depth doubling", f"{out['doubling_alm']:.2f}x")
    say("PC-ALM at L=1000 against T = 2L", f"{out['vs_2L']:.2f}x")
    return out


def measurement_checks(quiet=False):
    """Checks behind the metric choices: residuals, reference budgets, depth floor."""
    say, out = _say(quiet, "metric checks"), {}
    R = list(csv.DictReader(open(RES / "t_res.csv")))
    pc = [r for r in R if r["method"] == "pc"]
    out["pc_resid_cells"] = len({(r["depth"], r["seed"]) for r in pc})
    out["pc_resid_min_at_T1"] = all(r["resid_argmin"] == "1" for r in pc)
    say("PC cells / residual smallest at T=1 in all", f"{out['pc_resid_cells']} / "
        f"{out['pc_resid_min_at_T1']}")
    arg = lambda L: float(np.median([int(r["resid_argmin"]) for r in R if r["method"] == "pcalm"
                                     and int(r["depth"]) == L and r["point"] == "trained"]))
    out["alm_argmin_L4"], out["alm_argmin_L256"] = arg(4), arg(256)
    say("PC-ALM residual argmin at trained parameters, median, L=4 / L=256",
        f"{out['alm_argmin_L4']:.0f} / {out['alm_argmin_L256']:.0f}")
    dep = {r["depth"]: r for r in conv_b_dependence(build_conv2())}
    out["tconv_L64_ratio"], out["tconv_L16_ratio"] = dep[64]["T_conv_ratio"], dep[16]["T_conv_ratio"]
    out["tconv_L16_B"] = dep[16]["B_ratio"]
    say("distance-to-fixed-point value ratio across reference budgets, L=64 / L=16",
        f"{out['tconv_L64_ratio']} / {out['tconv_L16_ratio']} (B ratio {out['tconv_L16_B']})")

    C = _jsonl("conv/L*_s*.jsonl")
    cbp = {(r["depth"], r["seed"]): r["test_acc"] for r in C if r["method"] == "bp"}
    for m, tag in (("pc", "pc"), ("pcalm", "alm")):
        cur = {}
        for r in C:
            if r["method"] == m:
                cur.setdefault((r["depth"], r["seed"]), {})[r["budget"]] = r["test_acc"]
        det, _ = _t_targets(cur, lambda L, s: FRAC * cbp[(L, s)], depths={32})
        out[f"conv_{tag}_L32"] = _mean(det[32])
    cm = lambda L: 100 * _mean([cbp[(L, s)] for s in range(5)])
    out["conv_bp_drop"] = cm(8) - cm(32)
    b = budget_ladder(quiet=True)
    out["mlp_bp_drop"] = b["bp_L8"] - b["bp_L32"]
    say("conv stack at L=32: mean T_target PC / PC-ALM (floor L-1 = 31)",
        f"{out['conv_pc_L32']:.1f} / {out['conv_alm_L32']:.1f}")
    say("BP accuracy drop L=8 -> 32, conv / residual MLP (pp)",
        f"{out['conv_bp_drop']:.2f} / {out['mlp_bp_drop']:.2f}")

    gate = {r["method"]: r for r in csv.DictReader(open(RES / "repro_fashion_n32_l32/cells.csv"))}
    for m in ("bp", "pc", "pcalm"):
        out[f"gate_{m}_acc"] = 100 * float(gate[m]["final_test_acc"])
        out[f"gate_{m}_cos"] = float(gate[m]["grad_cos_to_bp"])
    say("reproduction cell, test accuracy BP / PC / PC-ALM (%)",
        " / ".join(f"{out[f'gate_{m}_acc']:.2f}" for m in ("bp", "pc", "pcalm")))
    say("reproduction cell, gradient cosine to BP",
        " / ".join(f"{out[f'gate_{m}_cos']:.3f}" for m in ("bp", "pc", "pcalm")))
    vl = {(r["activation"], int(r["T"])): float(r["cos_to_TRUE_BP"])
          for r in csv.DictReader(open(RES / "strengthen/verify_linear.csv"))}
    out.update(lin_cos_16384=vl[("linear", 16384)], relu_cos_1024=vl[("relu", 1024)],
               relu_cos_16384=vl[("relu", 16384)])
    say("L=32 PC-ALM gradient cosine to BP: linear at T=16384 / ReLU at T=1024 / T=16384",
        f"{out['lin_cos_16384']:.4f} / {out['relu_cos_1024']:.3f} / {out['relu_cos_16384']:.3f}")
    oc = list(csv.DictReader(open(RES / "strengthen/output_curvature.csv")))
    fails = [r for r in oc if r["upper_holds"] == "False"]
    out["upper_fail_cells"] = len(fails)
    out["upper_fail_all_B_indefinite"] = all(float(r["B_lambda_min"]) < 0 for r in fails)
    say("cells where the upper bound fails / all with lambda_min(B) < 0; range",
        f"{len(fails)} / {out['upper_fail_all_B_indefinite']}; "
        f"{min(float(r['B_lambda_min']) for r in fails):.4f} to "
        f"{max(float(r['B_lambda_min']) for r in fails):.4f}")
    return out


def hopfield(quiet=False):
    """Activity Hessian I - (C + C^T) of the Hopfield-type energy at initialisation."""
    say, out = _say(quiet, "Hopfield-type energy"), {}
    fam = {}
    for f in ("kappa_hopfield.csv", "kappa_hopfield_orth.csv", "kappa_hopfield_noskip.csv"):
        for r in csv.DictReader(open(RES / "strengthen" / f)):
            fam.setdefault(r["family"], []).append(r)
    for name in ("hopfield", "hopfield_orth_linear", "hopfield_orth_tanh"):
        rs = fam.get(name, [])
        out[f"{name}_indefinite_all"] = bool(rs) and all(float(r["lambda_min"]) < 0 for r in rs)
        say(f"{name}: indefinite in every cell, depths {sorted({int(r['depth']) for r in rs})}",
            out[f"{name}_indefinite_all"])
    k = {}
    for r in fam["hopfield_noskip"]:
        k.setdefault(int(r["depth"]), []).append(float(r["kappa"]))
    out["noskip_kappa_L4"], out["noskip_kappa_L128"] = _mean(k[4]), _mean(k[128])
    say("reference weights without the skip: kappa at L=4 / L=128",
        f"{out['noskip_kappa_L4']:.2f} / {out['noskip_kappa_L128']:.2f}")
    ch = list(csv.DictReader(open(RES / "strengthen/hopfield_chain.csv")))
    half = [r for r in ch if float(r["g"]) == 0.5]
    out["chain_half_equals_cot2"] = all(abs(float(r["kappa"]) / float(r["cot2"]) - 1) < 1e-3
                                        for r in half)
    out["chain_indefinite_above_half"] = all(float(r["lambda_min"]) < 0 for r in ch
                                             if float(r["g"]) > 0.5)
    out["chain_definite_at_or_below_half"] = all(float(r["lambda_min"]) > 0 for r in ch
                                                 if float(r["g"]) <= 0.5)
    say("orthogonal linear chain, gain g: kappa = cot^2(pi/(2n+2)) at g=1/2 / indefinite "
        "for g > 1/2 / definite for g <= 1/2", f"{out['chain_half_equals_cot2']} / "
        f"{out['chain_indefinite_above_half']} / {out['chain_definite_at_or_below_half']}")
    return out


def mechanisms(quiet=False):
    """Candidate explanations for PC-ALM's post-knee accuracy decay."""
    say, out = _say(quiet, "post-knee decay: tested mechanisms"), {}
    rec = _jsonl("reconcile/*.jsonl")
    for L in (64, 128):
        at = lambda sc, k: _mean([r[k] for r in rec if r["depth"] == L and r["lr_scale"] == sc])
        out[f"dual_peak_over_L_L{L}"] = at(1.0, "peak_over_L")
        out[f"fluct_ratio_L{L}"] = at(0.1, "fluct_rel") / at(1.0, "fluct_rel")
        say(f"L={L}: dual-norm peak at T/L / tail fluctuation at eta_h x0.1 relative to x1",
            f"{out[f'dual_peak_over_L_L{L}']:.2f} / {out[f'fluct_ratio_L{L}']:.3f}")
    out["fluct_ratio_min"] = min(out["fluct_ratio_L64"], out["fluct_ratio_L128"])
    out["fluct_ratio_max"] = max(out["fluct_ratio_L64"], out["fluct_ratio_L128"])
    T2 = _import_quietly("analysis.task2_analysis")
    out["decay_ratio_L64"] = T2.summary[(64, 0.1)] / T2.summary[(64, 1.0)]
    say("L=64: accuracy drop at 4x knee, eta_h x0.1 relative to x1",
        f"{T2.summary[(64, 0.1)]:.2f} / {T2.summary[(64, 1.0)]:.2f} pp = "
        f"{out['decay_ratio_L64']:.1f}x")
    fall = []
    for L in (16, 64, 128):
        zs = [np.load(f) for f in sorted((RES / "task3").glob(f"L{L}_s*.npz"))]
        fall.append(_mean([z["trained_resid_max"][2 * L] for z in zs]) /
                    _mean([z["trained_resid_max"][-1] for z in zs]))
    out["resid_fall_min"], out["resid_fall_max"] = min(fall), max(fall)
    say("constraint residual at T=2L over T=4096, L=16/64/128", ", ".join(f"{x:.2f}" for x in fall))
    R = _jsonl("item2/L64_s*.jsonl")
    for arm in ("none", "input", "output"):
        d = []
        for s in sorted({r["seed"] for r in R}):
            rr = [r for r in R if r["arm"] == arm and r["seed"] == s]
            d.append(100 * (max(r["test_acc"] for r in rr)
                            - max(rr, key=lambda r: r["mult_of_knee"])["test_acc"]))
        out[f"freeze_{arm}_L64"] = _mean(d)
    say("L=64: peak minus accuracy at 32x knee, duals frozen none / input half / output half (pp)",
        " / ".join(f"{out[f'freeze_{a}_L64']:.2f}" for a in ("none", "input", "output")))
    # main sweep: where PC-ALM's accuracy peaks, in units of the knee T = 2L, and how far it
    # has fallen by the end of the ladder, at the depths whose ladder extends past 2x the knee
    with _at_root():
        cur = load_curves(set(DEPTHS7), "pcalm")
    peaks, drops = [], {}
    for L in (32, 64, 128):
        for (d, s), c in cur.items():
            if d != L:
                continue
            Ts = sorted(c)
            i = int(np.argmax([c[t] for t in Ts]))
            peaks.append(Ts[i] / (2 * L))
            drops.setdefault(L, []).append(100 * (c[Ts[i]] - c[Ts[-1]]))
    out["peak_over_knee_min"], out["peak_over_knee_max"] = min(peaks), max(peaks)
    out["drop_mean_min"] = min(_mean(v) for v in drops.values())
    out["drop_mean_max"] = max(_mean(v) for v in drops.values())
    say("PC-ALM accuracy peak, L=32-128, in units of T=2L / mean drop by end of ladder (pp)",
        f"{out['peak_over_knee_min']:.0f}-{out['peak_over_knee_max']:.0f} / "
        + ", ".join(f"L={L}: {_mean(v):.2f}" for L, v in drops.items()))
    return out


def width_and_networks(quiet=False):
    """Width dependence of kappa, and the eight-network test of the conditioning theorems."""
    say, out = _say(quiet, "kappa against width; the eight networks"), {}
    for w, fn in ((256, "kappa/mup_w256.csv"), (1024, "kappa/mup_w1024.csv")):
        R = [r for r in csv.DictReader(open(RES / fn))]
        f = loglog_fit([int(r["depth"]) for r in R], [float(r["kappa"]) for r in R])
        out[f"w{w}"], out[f"w{w}_depths"] = f["slope"], f["n_depths"]
        out[f"w{w}_L64"] = _mean([float(r["kappa"]) for r in R if int(r["depth"]) == 64])
        say(f"width {w}: kappa vs depth / kappa at L=64", f"{_fmt(f)} / {out[f'w{w}_L64']:.0f}")
    out["w32_L64"] = _mean([float(r["kappa"]) for r in csv.DictReader(open(RES / "kappa/standard_w32.csv"))
                            if int(r["depth"]) == 64])
    rows = _jsonl("strengthen/isometry_[abr]*.jsonl")
    arm = lambda a, act, g: [r for r in rows if r["arm"] == a and r["act"] == act and r["gain"] == g]
    for key, (a, act, g) in {"orth_relu": ("orth", "relu", 1.0), "gauss_relu": ("gauss", "relu", 1.0),
                             "orth_tanh": ("orth", "tanh", 1.0), "reference": ("reference", "relu", 1.0),
                             "contract_relu": ("orth", "relu", 0.8)}.items():
        rs = [r for r in arm(a, act, g) if math.isfinite(r["kappa"])]
        out[f"{key}_slope"] = loglog_fit([r["depth"] for r in rs], [r["kappa"] for r in rs])["slope"]
    out["distorted_faster_than_L2"] = out["orth_relu_slope"] > 2 and out["gauss_relu_slope"] > 2
    out["contract_relu_far_below_L2"] = out["contract_relu_slope"] < 1.5
    pdd = sorted({r["depth"] for r in arm("orth", "tanh", 1.2) if math.isfinite(r["kappa"])})
    out["expansive_indefinite_from"] = min(r["depth"] for r in arm("orth", "tanh", 1.2)
                                           if not math.isfinite(r["kappa"]))
    tr = _jsonl("strengthen/isometry_train.jsonl")
    def fails_by(a, act, g):
        by = {}
        for r in tr:
            if (r["arm"], r["act"], r["gain"]) == (a, act, g):
                by.setdefault(r["depth"], []).append(r["bp_test_acc"])
        return min(L for L, v in by.items() if _mean(v) < 0.25)
    out["orth_relu_fails_by"], out["contract_tanh_fails_by"] = (fails_by("orth", "relu", 1.0),
                                                              fails_by("orth", "tanh", 0.8))
    say("kappa exponents: reference / orth tanh / orth ReLU / Gaussian ReLU / contractive ReLU",
        " / ".join(f"{out[k + '_slope']:.2f}" for k in
                   ("reference", "orth_tanh", "orth_relu", "gauss_relu", "contract_relu")))
    say("orthogonal ReLU / contractive tanh fall below 25% accuracy by L; expansive tanh "
        "indefinite from L", f"{out['orth_relu_fails_by']} / {out['contract_tanh_fails_by']}; "
        f"{out['expansive_indefinite_from']} (positive definite at {pdd})")
    return out


def wide128(quiet=False):
    """The protocol at width 128 (analysis/run_wide.py, run_wide_extend.sh, timing_wide.py)."""
    say, out = _say(quiet, "width 128"), {}
    W = _import_quietly("analysis.wide_analysis")
    bp = {k: v[0] for k, v in W.curves("bp").items()}
    cur = {m: W.curves(m) for m in W.ARMS}
    below = {}
    for frac in W.BARS:
        r = {m: W.targets(cur[m], bp, frac) for m in W.ARMS}
        below[frac] = sum(t < L - 1 for m in W.ARMS for L, ts in r[m][0].items() for t in ts)
    out["below_90"], out["below_95"] = below[0.90], below[0.95]
    bar = next(f for f in W.BARS if below[f] == 0)
    out["bar"] = 100 * bar
    say("cells below the L-1 floor by bar; bar used", f"{below}; {bar:.0%}")
    for frac in (0.95, bar, 0.98):
        for m in W.ARMS:
            f, Ls = W.fit(*W.targets(cur[m], bp, frac))
            key = f"{m}_{round(100 * frac)}"
            if f:
                _put(out, key, f)
            out[key + "_depths"] = len(Ls)
            say(f"{m} at {frac:.0%}", f"{_fmt(f) if f else 'n/a'} over {Ls}")
    out["bp_L8"] = 100 * _mean([a for (L, s), a in bp.items() if L == 8])
    out["bp_L128"] = 100 * _mean([a for (L, s), a in bp.items() if L == 128])
    det = {m: W.targets(cur[m], bp, bar) for m in W.ARMS}
    for m in W.ARMS:
        for L, ts in det[m][0].items():
            out[f"{m}_L{L}_min"], out[f"{m}_L{L}_max"] = min(ts), max(ts)
    out["nag_censored_L128"] = det["nag"][1].get(128, 0)
    c0 = cur["nag"][(128, 0)]
    out["nag_L128_s0_T256"], out["nag_L128_s0_T1024"] = 100 * c0[256], 100 * c0[1024]
    say("Nesterov L=128 seed 0, accuracy at T=256 / T=1024 (%)",
        f"{out['nag_L128_s0_T256']:.2f} / {out['nag_L128_s0_T1024']:.2f}")
    cost = {(int(r["depth"]), r["method"]): (float(r["fixed_ms"]), float(r["per_iter_ms"]))
            for r in csv.DictReader(open(RES / "wide128_timing.csv"))}
    mins = {m: W.STEPS * (cost[(64, m)][0] + _mean(det[m][0][64]) * cost[(64, m)][1]) / 6e4
            for m in W.ARMS}
    out.update(min_pc_L64=mins["pc"], min_pcalm_L64=mins["pcalm"], min_nag_L64=mins["nag"],
               speedup_pc_nag_L64=mins["pc"] / mins["nag"],
               iter_ratio_nag_pc_L64=cost[(64, "nag")][1] / cost[(64, "pc")][1])
    say("L=64 minutes per epoch at the budget needed, PC / PC-ALM / Nesterov; speedup",
        f"{mins['pc']:.2f} / {mins['pcalm']:.2f} / {mins['nag']:.2f}; "
        f"{out['speedup_pc_nag_L64']:.1f}x (Nesterov iteration {out['iter_ratio_nag_pc_L64']:.2f}x PC)")
    hours = sum(r.get("wall_sec", 0) for f in (RES / "wide128").glob("*.jsonl")
                for r in map(json.loads, filter(str.strip, open(f))))
    out["process_hours"] = hours / 3600
    return out


def timing(quiet=False):
    """Wall-clock cost per inner iteration, PC-ALM against PC, at T = 2L (timing_probe.py)."""
    say, out = _say(quiet, "wall-clock cost per inner iteration"), {}
    per = {}
    for r in csv.DictReader(open(RES / "strengthen/timing.csv")):
        per.setdefault(int(r["depth"]), {})[r["method"]] = float(r["per_step_s"])
    ratio = {L: v["pcalm"] / v["pc"] for L, v in sorted(per.items())}
    deep = [ratio[L] for L in ratio if L >= 64]
    out.update(ratio_L32=ratio[32], ratio_deep_min=min(deep), ratio_deep_max=max(deep))
    say("PC-ALM / PC per-iteration wall clock", {L: round(r, 2) for L, r in ratio.items()})
    with _at_root():
        cur = load_curves(set(DEPTHS7), "pcalm")
    det, _ = _t_targets(cur, _rel(FRAC))
    t_alm = _mean(det[128])
    out["alm_pc_equiv_L128"] = t_alm * ratio[128]
    out["alm_advantage_L128"] = 2048 / out["alm_pc_equiv_L128"]
    say("L=128: PC-ALM refined T in PC-equivalent iterations / advantage over PC's 2048",
        f"{out['alm_pc_equiv_L128']:.0f} / {out['alm_advantage_L128']:.2f}x")
    return out


def solvers(quiet=False):
    """Inner-solve cost against depth for steepest descent, Nesterov and a V-cycle,
    at the 80k reference budget; a depth with an unconverged seed is censored."""
    say, out = _say(quiet, "inner solve alone: steepest descent, Nesterov, multigrid"), {}
    R = _jsonl("strengthen/multigrid/mg_L*.jsonl")
    es = {}
    for r in R:
        es.setdefault((r["depth"], r["seed"]), {})[r["ref_work"]] = (r["e_star"], r["e0"])
    agree = {k: (v[20000][0] - v[80000][0]) / (v[80000][1] - v[80000][0]) for k, v in es.items()}
    good = [a for a in agree.values() if a < 1e-4]
    out["ref_agree_cells"], out["ref_cells"] = len(good), len(agree)
    out["ref_agree_max"] = max(good)
    chk = _jsonl("strengthen/multigrid/checks/*.jsonl")
    e = {r["ref_work"]: (r["e_star"], r["e0"]) for r in chk}
    out["ref320_agree"] = (e[80000][0] - e[320000][0]) / (e[320000][1] - e[320000][0])
    say("reference E*: 20k vs 80k agree to <1e-4 of the gap in / cells; largest; 80k vs 320k "
        "in the other", f"{len(good)} / {len(agree)}; {max(good):.1e}; {out['ref320_agree']:.1e}")
    rw = 80000
    for solver in ("sd", "nag", "mg"):
        for metric in ("work", "rounds"):
            d, cens = {}, []
            for L in sorted({r["depth"] for r in R}):
                g = [r for r in R if r["solver"] == solver and r["depth"] == L
                     and r["ref_work"] == rw]
                if all(r["converged"] for r in g):
                    d[L] = [r[metric] for r in g]
                else:
                    cens.append(L)
            f = loglog_fit([L for L in d for _ in d[L]], [v for L in d for v in d[L]])
            _put(out, f"{solver}_{metric}", f)
            out[f"{solver}_censored"] = cens
            say(f"{solver} {metric}", f"{_fmt(f)}; censored depths {cens}")
    gm = lambda s, L, m: math.exp(_mean([math.log(r[m]) for r in R if r["solver"] == s
                                         and r["depth"] == L and r["ref_work"] == rw]))
    ratio = [gm("mg", L, "work") / gm("nag", L, "work") for L in sorted({r["depth"] for r in R})
             if L not in out["mg_censored"]]
    out["mg_over_nag_work_min"], out["mg_over_nag_work_max"] = min(ratio), max(ratio)
    say("multigrid work / Nesterov work at uncensored depths",
        f"{min(ratio):.1f} to {max(ratio):.1f}")
    W = list(csv.DictReader(open(RES / "strengthen/estar_wander.csv")))
    sd = [r for r in W if r["solver"] == "sd"]
    out["sd_final_above_min"] = all(float(r["e_final"]) > float(r["e_min"]) for r in sd)
    out["sd_increase_frac_max"] = max(int(r["n_increases"]) / int(r["n_iter"]) for r in sd)
    say("steepest descent reference: final above lowest in every run / largest share of "
        "steps raising the energy", f"{out['sd_final_above_min']} / "
        f"{out['sd_increase_frac_max']:.2f}")
    return out


def gradient_tax(quiet=False):
    """Iterations until the inner solve's weight gradient stays within 10% of its converged
    value (hold window), whole network and by layer group, at initialisation."""
    say, out = _say(quiet, "weight-gradient budget of the inner solve"), {}
    for solver in ("nag", "gd"):
        R = _jsonl(f"strengthen/gradient_tax_{solver}.jsonl")
        out[f"{solver}_ref_ok"] = all(v < 0.1 / 3 for r in R
                                      for v in r["ref_converged_group"].values())
        out[f"{solver}_hold_equals_first"] = all(
            r["t_delta"][g]["0.1"] == r["t_first"][g]["0.1"] for r in R for g in r["t_delta"])
        for g in ("all", "input_layer", "input_hidden", "output_hidden", "readout"):
            f = loglog_fit([r["depth"] for r in R], [r["t_delta"][g]["0.1"] for r in R])
            _put(out, f"{solver}_{g}", f)
            say(f"{solver} {g}, L={min(r['depth'] for r in R)}-{max(r['depth'] for r in R)}",
                _fmt(f))
        norm = (lambda r: math.sqrt(r["kappa"])) if solver == "nag" else (lambda r: r["kappa"])
        per = {}
        for r in R:
            per.setdefault(r["depth"], []).append(r["t_delta"]["all"]["0.1"] / norm(r))
        out[f"{solver}_t_norm_min"] = min(_mean(v) for v in per.values())
        out[f"{solver}_t_norm_max"] = max(_mean(v) for v in per.values())
        say(f"{solver}: t / {'sqrt(kappa)' if solver == 'nag' else 'kappa'} by depth",
            {L: round(_mean(v), 2) for L, v in sorted(per.items())})
        say(f"{solver}: every group's reference converged to < delta/3 / hold == first crossing",
            f"{out[f'{solver}_ref_ok']} / {out[f'{solver}_hold_equals_first']}")
    R = [r for r in _jsonl("strengthen/gradient_tax_nag.jsonl") if r["depth"] >= 16]
    for g in ("output_hidden", "readout"):
        f = loglog_fit([r["depth"] for r in R], [r["t_delta"][g]["0.1"] for r in R])
        out[f"nag_{g}_16"] = f["slope"]
        say(f"nag {g}, L=16-512", _fmt(f))
    return out


def trained_kappa(quiet=False):
    """Condition number at trained weights (PC-ALM, reference pipeline), and through
    training at fixed fractions of the epoch."""
    say, out = _say(quiet, "kappa at trained weights"), {}
    rows = []
    for f in ("kappa_trained_traj.csv", "kappa_trained_traj_256.csv"):
        rows += list(csv.DictReader(open(RES / "strengthen" / f)))
    K = {}
    for r in rows:
        K.setdefault((int(r["depth"]), int(r["seed"])), {})[float(r["frac"])] = float(r["kappa"])
    fracs = sorted({float(r["frac"]) for r in rows})
    for fr in fracs:
        pts = [(L, d[fr]) for (L, s), d in K.items() if fr in d]
        f = loglog_fit([p[0] for p in pts], [p[1] for p in pts])
        _put(out, f"frac{fr:g}", f)
        say(f"kappa vs depth at {fr:.0%} of the epoch, L=4-256", _fmt(f))
    mid = [out[f"frac{fr:g}"] for fr in fracs if fr > 0]
    out["during_min"], out["during_max"] = min(mid), max(mid)
    ratio = {}
    for (L, s), d in K.items():
        ratio.setdefault(L, []).append(d[1.0] / d[0.0])
    out["growth_min"] = min(_mean(v) for v in ratio.values())
    out["growth_max"] = max(_mean(v) for v in ratio.values())
    say("kappa end / start of training, per-depth mean", {L: round(_mean(v), 2)
                                                          for L, v in sorted(ratio.items())})
    return out


GROUPS = (kappa_spectrum, step_size_offsets, skip_strength, isometry, budget_ladder,
          threshold_fits, headline_fits, grid_systematic, log_correction, gradient_alignment,
          robustness, cifar, momentum, prefactors, measurement_checks, hopfield, mechanisms,
          width_and_networks, wide128, timing, solvers, gradient_tax, trained_kappa)


# ------------------------------------------------------------------------ --check

# Reference values, written at the precision they are reported; "exact" marks a fit
# whose residuals vanish. Keys are "<group>.<name>".
EXPECTED = {
    "kappa_spectrum.kappa": "2.024", "kappa_spectrum.kappa_lo": "1.953",
    "kappa_spectrum.kappa_hi": "2.095", "kappa_spectrum.kappa_r2": "0.9991",
    "kappa_spectrum.kappa512": "2.015", "kappa_spectrum.kappa512_lo": "1.973",
    "kappa_spectrum.kappa512_hi": "2.058", "kappa_spectrum.kappa512_r2": "0.9994",
    "kappa_spectrum.lmin": "-2.072", "kappa_spectrum.lmin_lo": "-2.150",
    "kappa_spectrum.lmin_hi": "-1.994", "kappa_spectrum.lmax": "-0.048",
    "kappa_spectrum.lmax_lo": "-0.059", "kappa_spectrum.lmax_hi": "-0.037",
    "kappa_spectrum.lmax_excess": "-0.58", "kappa_spectrum.lmax_L4": "4.941",
    "kappa_spectrum.lmax_L256": "4.095", "kappa_spectrum.lmax_monotone": True,
    "kappa_spectrum.lmax_above_4_all": True,
    "kappa_spectrum.kappa_8inputs": "2.036", "kappa_spectrum.kappa_8inputs_lo": "1.955",
    "kappa_spectrum.kappa_8inputs_hi": "2.116",
    "kappa_spectrum.kappa_conv": "2.228", "kappa_spectrum.kappa_conv_lo": "1.929",
    "kappa_spectrum.kappa_conv_hi": "2.526", "kappa_spectrum.kappa_dense_4_32": "2.121",
    "kappa_spectrum.kappa_dense_4_32_lo": "1.934", "kappa_spectrum.kappa_dense_4_32_hi": "2.308",
    "kappa_spectrum.eigvec_interp_L128": "0.036",
    "kappa_spectrum.sp_log2_ratio_1": "3.66", "kappa_spectrum.sp_log2_ratio_2": "3.77",
    "kappa_spectrum.sp_log2_ratio_3": "5.95",
    "step_size_offsets.off_min": "0.4", "step_size_offsets.off_max": "3.1",
    "step_size_offsets.off_mean": "2.4", "step_size_offsets.off_sd": "1.2",
    "step_size_offsets.off_trend_p": "0.24",
    "skip_strength.c1": "2.036", "skip_strength.c1_lo": "1.925", "skip_strength.c1_hi": "2.147",
    "skip_strength.underflow_c0": "32", "skip_strength.underflow_c0.25": "64",
    "skip_strength.underflow_c0.5": "128", "skip_strength.chance_by_L_weak": "16",
    "skip_strength.chance_by_L_all_below_1": "32",
    "isometry.n_arms": "8", "isometry.n_pd": "132", "isometry.lower_holds": "132",
    "isometry.upper_fails": "7", "isometry.lin_L32_s0": "1604.9", "isometry.lin_L32_s1": "1604.8",
    "isometry.lin_L32_s2": "1604.9", "isometry.lin_L32_closed": "1604.9",
    "isometry.bp_lin_L128": "79.0", "isometry.bp_tanh_L64": "81.1", "isometry.bp_ref_L64": "84.3",
    "isometry.tanh": "2.159", "isometry.contract_kappa_L128": "80.0",
    "isometry.contract_cap": "83.1", "isometry.gain12_indefinite_from": "32",
    "threshold_fits.pc_90": "1.916", "threshold_fits.pc_90_lo": "1.782",
    "threshold_fits.pc_90_hi": "2.050", "threshold_fits.pcalm_90": "1.269",
    "threshold_fits.pcalm_90_lo": "1.177", "threshold_fits.pcalm_90_hi": "1.361",
    "threshold_fits.pc_95": "2.067", "threshold_fits.pc_95_lo": "1.863",
    "threshold_fits.pc_95_hi": "2.271", "threshold_fits.pcalm_95": "1.252",
    "threshold_fits.pcalm_95_lo": "1.114", "threshold_fits.pcalm_95_hi": "1.390",
    "threshold_fits.pc_98_strict": "2.000", "threshold_fits.pc_98_strict_lo": "1.142",
    "threshold_fits.pc_98_strict_hi": "2.858", "threshold_fits.pcalm_98_strict": "1.447",
    "threshold_fits.pcalm_98_strict_lo": "1.334", "threshold_fits.pcalm_98_strict_hi": "1.560",
    "threshold_fits.pc_98_strict_depths": "3", "threshold_fits.pcalm_98_strict_depths": "4",
    "headline_fits.pc7": "2.000", "headline_fits.pc7_lo": "1.848", "headline_fits.pc7_hi": "2.152",
    "headline_fits.pc_frozen": "2.000", "headline_fits.pc_frozen_exact": True,
    "headline_fits.alm_frozen": "1.258", "headline_fits.alm_frozen_lo": "1.100",
    "headline_fits.alm_frozen_hi": "1.417", "headline_fits.alm7": "1.244",
    "headline_fits.alm7_lo": "1.173", "headline_fits.alm7_hi": "1.315",
    "headline_fits.ref7": "1.208", "headline_fits.ref7_lo": "1.175",
    "headline_fits.ref7_hi": "1.241", "headline_fits.ref7_r2": "0.999",
    "headline_fits.alm7_width": "0.142", "headline_fits.ref7_width": "0.067",
    "headline_fits.ref7_half": "0.033", "headline_fits.pc_floor_L8": "1.1",
    "headline_fits.pc_floor_L128": "16.1", "headline_fits.alm_floor_min": "1.0",
    "headline_fits.alm_floor_max": "2.0",
    "grid_systematic.pcalm_span_lo": "1.234", "grid_systematic.pcalm_span_hi": "1.366",
    "grid_systematic.pc_span_lo": "1.990", "grid_systematic.pc_span_hi": "2.097",
    "grid_systematic.pcalm_half_range": "0.066", "grid_systematic.pc_half_range": "0.053",
    "grid_systematic.limit_4": "0.146", "grid_systematic.limit_6": "0.069",
    "grid_systematic.limit_6_ratio15": "0.084", "grid_systematic.ref7_lo_net": "1.11",
    "log_correction.d_aicc": "3.5", "log_correction.b": "0.62", "log_correction.p": "1.21",
    "log_correction.pc_control_p": "2.00",
    "gradient_alignment.cos90": "1.110", "gradient_alignment.cos90_lo": "0.951",
    "gradient_alignment.cos90_hi": "1.269", "gradient_alignment.n_censored": "0",
    "gradient_alignment.accuracy_budget_same_depths": "1.227",
    "gradient_alignment.accuracy_budget_same_depths_lo": "1.198",
    "gradient_alignment.accuracy_budget_same_depths_hi": "1.255",
    "robustness.base_pc": "2.000", "robustness.base_pc_exact": True, "robustness.base_alm": "1.234",
    "robustness.fashion_pc": "2.000", "robustness.fashion_pc_exact": True,
    "robustness.fashion_alm": "1.234", "robustness.w16_pc": "2.000", "robustness.w16_alm": "1.244",
    "robustness.w64_pc": "2.017", "robustness.w64_alm": "1.234", "robustness.ep5_pc": "2.033",
    "robustness.ep5_alm": "1.234", "robustness.offset_pc": "2.097", "robustness.offset_alm": "1.366",
    "robustness.abs8_64_pc_min": "2.00", "robustness.abs8_64_pc_max": "2.15",
    "robustness.abs8_64_alm_min": "1.234", "robustness.abs8_64_alm_max": "1.234",
    "robustness.ref_abs70": "1.223", "robustness.ref_abs75": "1.228", "robustness.ref_abs80": "1.250",
    "robustness.bp1_min": "81.9", "robustness.bp1_max": "89.9", "robustness.bp5_min": "92.8",
    "robustness.bp5_max": "94.3", "robustness.bp_decline": "9",
    "cifar.bp_L4": "39.30", "cifar.bp_L128": "30.24", "cifar.bp_decline": "23",
    "cifar.pc": "2.000", "cifar.pc_exact": True, "cifar.alm": "1.258", "cifar.alm_lo": "1.018",
    "cifar.alm_hi": "1.499", "cifar.alm_lo_net": "0.952", "cifar.c_pc": "0.031",
    "cifar.a_alm": "0.227", "cifar.q_alm": "0.630", "cifar.alm_rungs_exact": "4",
    "cifar.alm_rungs_one": "1", "cifar.pc_rungs_one_low": "5",
    "cifar.batch_estimator_mean_offset": "3.5",
    "momentum.gd": "1.916", "momentum.gd_lo": "1.782", "momentum.gd_hi": "2.050",
    "momentum.nag": "1.213", "momentum.nag_lo": "1.156", "momentum.nag_hi": "1.271",
    "momentum.ref6_minus_nag": "0.013", "momentum.ref6_inside_nag": True,
    "momentum.depthbeta": "1.216", "momentum.depthbeta_lo": "1.156",
    "momentum.depthbeta_hi": "1.276", "momentum.depthbeta_same": "16",
    "momentum.depthbeta_cells": "18", "momentum.kiso_min": "2.3", "momentum.kiso_max": "3.1",
    "momentum.gap_min": "1.5", "momentum.gap_max": "1.6", "momentum.hb_first": "1.083",
    "momentum.hb_first_lo": "0.930", "momentum.hb_first_hi": "1.236",
    "momentum.hb_L64_T64": "80.86", "momentum.hb_L64_T96": "24.01",
    "momentum.accelerated_prediction": "1.012", "momentum.beta_gap_within_one_rung": True,
    "prefactors.pc_a": "0.125", "prefactors.alm_a": "0.58", "prefactors.pc_1000": "1.25e5",
    "prefactors.alm_1000": "2.5e3", "prefactors.ratio_1000": "5e1",
    "prefactors.pc_a256": "0.19", "prefactors.alm_a256": "0.54",
    "prefactors.pc_1000_256": "1.9e5", "prefactors.alm_1000_256": "2.3e3",
    "prefactors.ratio_1000_256": "8e1", "prefactors.vs_2L_256": "1.1",
    "prefactors.pc_pred_256": "8192", "prefactors.pc_T256": "12288",
    "prefactors.per_update_pc": "3.00", "prefactors.per_update_alm": "2.21",
    "prefactors.doubling_alm": "4.6", "prefactors.vs_2L": "1.2",
    "prefactors.excess_over_linear": "0.21",
    "measurement_checks.pc_resid_cells": "35", "measurement_checks.pc_resid_min_at_T1": True,
    "measurement_checks.alm_argmin_L4": "1120", "measurement_checks.alm_argmin_L256": "15199",
    "measurement_checks.tconv_L64_ratio": "2.3", "measurement_checks.tconv_L16_ratio": "2.67",
    "measurement_checks.conv_alm_L32": "5", "measurement_checks.conv_pc_L32": "6",
    "measurement_checks.conv_bp_drop": "0.47", "measurement_checks.mlp_bp_drop": "2.76",
    "measurement_checks.gate_bp_acc": "78.66", "measurement_checks.gate_pc_acc": "68.13",
    "measurement_checks.gate_pcalm_acc": "77.75", "measurement_checks.gate_bp_cos": "1.000",
    "measurement_checks.gate_pc_cos": "0.604", "measurement_checks.gate_pcalm_cos": "0.909",
    "hopfield.hopfield_indefinite_all": True, "hopfield.hopfield_orth_linear_indefinite_all": True,
    "hopfield.hopfield_orth_tanh_indefinite_all": True, "hopfield.noskip_kappa_L4": "35",
    "hopfield.noskip_kappa_L128": "1.5", "hopfield.chain_half_equals_cot2": True,
    "hopfield.chain_indefinite_above_half": True, "hopfield.chain_definite_at_or_below_half": True,
    "mechanisms.dual_peak_over_L_L64": "6", "mechanisms.dual_peak_over_L_L128": "6",
    "mechanisms.fluct_ratio_min": "0.49", "mechanisms.fluct_ratio_max": "0.63",
    "mechanisms.decay_ratio_L64": "27", "mechanisms.resid_fall_min": "3",
    "mechanisms.resid_fall_max": "6", "mechanisms.freeze_input_L64": "1.77",
    "mechanisms.freeze_none_L64": "2.04", "mechanisms.freeze_output_L64": "0.92", "mechanisms.peak_over_knee_min": "2",
    "mechanisms.peak_over_knee_max": "3", "mechanisms.drop_mean_min": "2.7",
    "mechanisms.drop_mean_max": "4.2",
    "measurement_checks.lin_cos_16384": "0.9997", "measurement_checks.relu_cos_1024": "0.956",
    "measurement_checks.relu_cos_16384": "0.888", "measurement_checks.upper_fail_cells": "7",
    "measurement_checks.upper_fail_all_B_indefinite": True,
    "solvers.ref_agree_cells": "17", "solvers.ref_cells": "18",
    "solvers.sd_work": "1.792", "solvers.sd_work_lo": "1.699", "solvers.sd_work_hi": "1.885",
    "solvers.sd_rounds": "2.825", "solvers.sd_rounds_lo": "2.733", "solvers.sd_rounds_hi": "2.918",
    "solvers.nag_work": "0.964", "solvers.nag_work_lo": "0.829", "solvers.nag_work_hi": "1.100",
    "solvers.mg_work": "1.289", "solvers.mg_work_lo": "0.729", "solvers.mg_work_hi": "1.849",
    "solvers.mg_rounds": "2.428", "solvers.mg_rounds_lo": "1.753", "solvers.mg_rounds_hi": "3.102",
    "solvers.mg_over_nag_work_min": "5", "solvers.mg_over_nag_work_max": "17",
    "solvers.sd_final_above_min": True, "solvers.sd_increase_frac_max": "0.29",
    "gradient_tax.nag_all": "1.002", "gradient_tax.nag_all_lo": "0.906",
    "gradient_tax.nag_all_hi": "1.098", "gradient_tax.nag_t_norm_min": "2.0",
    "gradient_tax.nag_t_norm_max": "3.1", "gradient_tax.gd_all": "1.985",
    "gradient_tax.gd_all_lo": "1.722", "gradient_tax.gd_all_hi": "2.249",
    "gradient_tax.nag_input_layer": "1.059", "gradient_tax.nag_input_layer_lo": "1.017",
    "gradient_tax.nag_input_layer_hi": "1.100", "gradient_tax.nag_output_hidden": "1.147",
    "gradient_tax.nag_output_hidden_lo": "1.055", "gradient_tax.nag_output_hidden_hi": "1.239",
    "gradient_tax.nag_readout": "1.207", "gradient_tax.nag_readout_lo": "1.054",
    "gradient_tax.nag_readout_hi": "1.359", "gradient_tax.nag_output_hidden_16": "1.09",
    "gradient_tax.nag_readout_16": "1.08", "gradient_tax.nag_ref_ok": True,
    "gradient_tax.gd_ref_ok": True, "gradient_tax.nag_hold_equals_first": True,
    "trained_kappa.frac0": "2.024", "trained_kappa.frac1": "2.051",
    "trained_kappa.frac1_lo": "1.943", "trained_kappa.frac1_hi": "2.159",
    "trained_kappa.during_min": "2.04", "trained_kappa.during_max": "2.06",
    "trained_kappa.growth_min": "1.1", "trained_kappa.growth_max": "1.5",
    "width_and_networks.w256": "2.18", "width_and_networks.w1024": "2.19",
    "width_and_networks.w256_depths": "3", "width_and_networks.w1024_depths": "3",
    "width_and_networks.w32_L64": "17872", "width_and_networks.w256_L64": "22082",
    "width_and_networks.w1024_L64": "22818", "width_and_networks.reference_slope": "2.05",
    "width_and_networks.orth_tanh_slope": "2.16",
    "width_and_networks.distorted_faster_than_L2": True,
    "width_and_networks.contract_relu_far_below_L2": True,
    "width_and_networks.expansive_indefinite_from": "32",
    "width_and_networks.orth_relu_fails_by": "64", "width_and_networks.contract_tanh_fails_by": "32",
    "momentum.beta_gap_max_rung_shift": "0",
    "wide128.below_90": "40", "wide128.below_95": "8", "wide128.bar": "97",
    "wide128.pc_97": "2.175", "wide128.pc_97_lo": "2.057", "wide128.pc_97_hi": "2.294",
    "wide128.pcalm_97": "1.258", "wide128.pcalm_97_lo": "1.100", "wide128.pcalm_97_hi": "1.417",
    "wide128.nag_97": "1.214", "wide128.nag_97_lo": "0.993", "wide128.nag_97_hi": "1.436",
    "wide128.pc_97_depths": "5", "wide128.pcalm_97_depths": "5", "wide128.nag_97_depths": "4",
    "wide128.pc_95": "2.04", "wide128.pcalm_95": "1.25", "wide128.nag_95": "1.32",
    "wide128.pc_98": "2.08", "wide128.pcalm_98": "1.20",
    "wide128.bp_L8": "94.59", "wide128.bp_L128": "93.44", "wide128.nag_censored_L128": "3",
    "wide128.nag_L128_s0_T256": "91.19", "wide128.nag_L128_s0_T1024": "85.94",
    "wide128.min_pc_L64": "5.9", "wide128.min_pcalm_L64": "1.7", "wide128.min_nag_L64": "1.2",
    "wide128.speedup_pc_nag_L64": "5.1", "wide128.iter_ratio_nag_pc_L64": "1.5",
    "wide128.pc_L128_min": "3072", "wide128.pcalm_L128_min": "256", "wide128.nag_L64_max": "96",
    "timing.ratio_L32": "2.7", "timing.ratio_deep_min": "1.6", "timing.ratio_deep_max": "1.8",
    "timing.alm_pc_equiv_L128": "3.6e2", "timing.alm_advantage_L128": "5.7",
}

# Budget ladder entries per depth: (PC, PC-ALM coarse, refined min, refined max, NAG,
# heavy-ball first crossing), None where not run.
_LADDER_ROWS = {4: (3, 3, 3, 3, 3, 3), 8: (8, 8, 7, 7, 8, 8), 16: (32, 16, 16, 16, 16, 16),
                32: (128, 48, 40, 40, 40, None), 64: (512, 96, 92, 92, 92, None),
                128: (2048, 256, 200, 208, 208, None), 256: (12288, 512, 432, 448, None, None)}
_BP_ROWS = {4: ("89.92", "0.63"), 8: ("89.26", "0.40"), 16: ("88.26", "0.42"),
            32: ("86.50", "1.52"), 64: ("84.31", "0.73"), 128: ("81.90", "1.82"),
            256: ("79.97", "0.44")}
_SPECTRUM_ROWS = {  # lambda_max, SD, lambda_min mantissa, SD, kappa, SD
    4: ("4.941", "0.130", "9.69", "0.55", "51", "4"),
    8: ("4.812", "0.048", "1.92", "0.04", "251", "8"),
    16: ("4.543", "0.040", "4.41", "0.55", "1042", "141"),
    32: ("4.364", "0.035", "1.02", "0.09", "4291", "410"),
    64: ("4.237", "0.005", "2.43", "0.43", "17872", "3471"),
    128: ("4.141", "0.005", "6.80", "0.67", "61251", "5676"),
    256: ("4.095", "0.002", "1.66", "0.18", "248727", "28875"),
    384: ("4.078", "0.004", "7.28", "0.34", "560877", "26307"),
    512: ("4.064", "0.003", "3.87", "0.29", "1052775", "74779")}
for _L, _row in _LADDER_ROWS.items():
    for _k, _v in zip(("pc_L{}_max", "alm_L{}_max", "ref_L{}_min", "ref_L{}_max", "nag_L{}_max",
                       "hb_L{}_first"), _row):
        if _v is not None:
            EXPECTED["budget_ladder." + _k.format(_L)] = str(_v)
    EXPECTED[f"budget_ladder.bp_L{_L}"], EXPECTED[f"budget_ladder.bp_L{_L}_sd"] = _BP_ROWS[_L]
EXPECTED["budget_ladder.hb_L8_min"], EXPECTED["budget_ladder.hb_L8_max"] = "8", "32"
for _L, _row in _SPECTRUM_ROWS.items():
    for _k, _v in zip(("lmax", "lmax_sd", "lmin_m", "lmin_sd_m", "kappa", "kappa_sd"), _row):
        EXPECTED[f"kappa_spectrum.t_L{_L}_{_k}"] = _v


def _agrees(value, expected):
    """Does value round to expected at the precision expected is written in?"""
    if isinstance(expected, bool):
        return bool(value) == expected
    if value is None:
        return False
    if "e" in expected:
        mant = expected.split("e")[0]
        nd = len(mant.split(".")[1]) if "." in mant else 0
        return float(f"{float(value):.{nd}e}") == float(expected)
    nd = len(expected.split(".")[1]) if "." in expected else 0
    return f"{float(value):.{nd}f}" == f"{float(expected):.{nd}f}"


def check(values):
    bad = 0
    print(f"\n{'value':52s} {'expected':>10s} {'computed':>14s}  verdict")
    for key, exp in EXPECTED.items():
        v = values.get(key)
        ok = _agrees(v, exp)
        bad += not ok
        shown = v if isinstance(v, (bool, type(None))) else f"{float(v):.5g}"
        print(f"{key:52s} {str(exp):>10s} {str(shown):>14s}  {'MATCH' if ok else 'MISMATCH'}")
    print(f"\n{len(EXPECTED) - bad} MATCH, {bad} MISMATCH of {len(EXPECTED)}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare every computed value with EXPECTED")
    a = ap.parse_args()
    values = {}
    for g in GROUPS:
        try:
            for k, v in g(quiet=a.check).items():
                values[f"{g.__name__}.{k}"] = v
        except NotImplementedError as e:
            print(f"\n== {g.__name__}: not implemented ({e})")
    if a.check:
        sys.exit(1 if check(values) else 0)


if __name__ == "__main__":
    main()
