"""Build every deliverable: raw CSVs, per-depth stats, log-log fits, REPORT.md."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ACC_TOL = 0.01                 # 1% absolute, as specified
BP_FRACS = [0.90, 0.95, 0.98]  # depth-matched BP-relative accuracy targets
CHANCE = 0.11                  # 10-class chance; below this a run learned nothing


def write_csv(rows, path, fields=None):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def loglog_fit(depths, values):
    """Log-log fit with CLUSTER-CORRECT inference.

    The seeds at a given depth are replicates, not independent observations --
    with a geometric budget ladder their within-depth spread in log T is often
    exactly zero. Regressing all n_runs points and taking df = n_runs - 2
    understates the standard error by sqrt(n_runs/n_depths) and uses a
    too-small t critical value; here that made every CI 3.6x too narrow.
    We therefore fit the per-depth means with df = n_depths - 2.
    """
    d, v = np.asarray(depths, float), np.asarray(values, float)
    k = np.isfinite(d) & np.isfinite(v) & (v > 0) & (d > 0)
    d, v = d[k], v[k]
    if len(d) < 3 or len(np.unique(d)) < 3:
        return None
    x_all, y_all = np.log(d), np.log(v)
    Ls = np.unique(d)
    xm = np.log(Ls)
    ym = np.array([y_all[d == L].mean() for L in Ls])
    r = stats.linregress(xm, ym)
    within = float(np.mean([y_all[d == L].std(ddof=1) if (d == L).sum() > 1 else 0.0
                            for L in Ls]))
    resid = ym - (r.intercept + r.slope * xm)
    exact = bool(np.allclose(resid, 0, atol=1e-12))
    half = stats.t.ppf(0.975, len(Ls) - 2) * r.stderr
    return dict(slope=r.slope, ci_lo=r.slope - half, ci_hi=r.slope + half,
                r2=(1.0 if exact else r.rvalue ** 2), n=int(len(d)),
                n_depths=int(len(Ls)), within_depth_sd_logT=round(within, 5),
                exact_fit=exact)


def fmt(f):
    if f is None:
        return "insufficient uncensored data"
    if f.get("exact_fit"):
        return (f"{f['slope']:+.3f}  (residuals identically zero on {f['n_depths']} "
                f"depth means; CI undefined, not infinitely precise)  "
                f"n={f['n']}/{f['n_depths']}d")
    return (f"{f['slope']:+.3f}  [{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}]  "
            f"R2={f['r2']:.3f}  n={f['n']}/{f['n_depths']}d (clustered, "
            f"df={f['n_depths']-2})")


def depth_stats(recs, key, method, only_uncensored=True):
    by = {}
    for r in recs:
        if r["method"] != method:
            continue
        b = by.setdefault(r["depth"], {"v": [], "c": 0})
        v = r.get(key)
        cens = r.get("censored", 0)
        if v is None or (only_uncensored and cens):
            b["c"] += 1
        else:
            b["v"].append(float(v))
    out = []
    for L in sorted(by):
        v = np.array(by[L]["v"])
        out.append(dict(depth=L, n_ok=len(v), n_censored=by[L]["c"],
                        mean=round(float(v.mean()), 2) if len(v) else None,
                        sd=round(float(v.std(ddof=1)), 2) if len(v) > 1 else None,
                        min=int(v.min()) if len(v) else None,
                        max=int(v.max()) if len(v) else None))
    return out


def load_sweep():
    rows = []
    for p in sorted((RES / "sweep").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dir(sub):
    out = []
    for p in sorted((RES / sub).glob("*.json")):
        t = p.read_text().strip()
        if t:
            out.append(json.loads(t))
    return out


def bp_table(rows):
    return {(r["depth"], r["seed"]): r["test_acc"] for r in rows if r["method"] == "bp"}


def build_task(rows):
    """Both task metrics, per (depth, seed, method) cell."""
    bp = bp_table(rows)
    cells = {}
    for r in rows:
        if r["method"] == "bp":
            continue
        cells.setdefault((r["depth"], r["seed"], r["method"]), []).append(r)
    lit, tgt = [], []
    for (L, s, m), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r["budget"])
        accs = [r["test_acc"] for r in rs]
        Ts = [r["budget"] for r in rs]
        T_ref, a_ref = Ts[-1], accs[-1]
        # plateau test: has the curve flattened at the top of the ladder?
        plateau = abs(a_ref - accs[-2]) < ACC_TOL if len(accs) > 1 else False
        hit = next((t for t, a in zip(Ts, accs) if a >= a_ref - ACC_TOL), None)
        lit.append(dict(depth=L, seed=s, method=m, T_ref=T_ref,
                        acc_at_T_ref=round(a_ref, 4),
                        acc_prev=round(accs[-2], 4) if len(accs) > 1 else None,
                        plateau_reached=int(plateau),
                        T_task=hit,
                        censored=int(hit is None or hit == T_ref or not plateau),
                        learned=int(a_ref > CHANCE),
                        all_finite=int(all(r["finite"] for r in rs)),
                        n_budgets=len(rs), max_acc=round(max(accs), 4)))
        a_bp = bp.get((L, s))
        for f in BP_FRACS:
            thr = f * a_bp if a_bp else None
            h = next((t for t, a in zip(Ts, accs) if thr and a >= thr), None)
            tgt.append(dict(depth=L, seed=s, method=m, frac=f,
                            bp_acc=round(a_bp, 4) if a_bp else None,
                            threshold=round(thr, 4) if thr else None,
                            T_target=h, censored=int(h is None),
                            T_max_run=T_ref, max_acc=round(max(accs), 4)))
    return lit, tgt


def build_resid(tols):
    rows, out = load_dir("resid"), []
    for r in rows:
        f = RES / "resid" / f"L{r['depth']}_s{r['seed']}_{r['method']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for tag in ("init", "trained"):
            c = z[tag]
            rec = dict(depth=r["depth"], seed=r["seed"], method=r["method"], point=tag,
                       probe_budget=int(len(c)), resid_T1=float(c[0]),
                       resid_min=float(c.min()), resid_argmin=int(c.argmin()) + 1,
                       resid_final=float(c[-1]), resid_max=float(c.max()),
                       grows_from_T1=int(float(c.max()) > float(c[0])),
                       min_is_at_T1=int(int(c.argmin()) == 0))
            for tol in tols:
                idx = np.nonzero(c < tol)[0]
                rec[f"T_res_tol{tol:g}"] = int(idx[0]) + 1 if len(idx) else None
                rec[f"censored_tol{tol:g}"] = int(len(idx) == 0)
            out.append(rec)
    return out


def build_conv(tols=(0.1, 0.03, 0.01)):
    out = []
    for r in load_dir("conv"):
        rec = dict(depth=r["depth"], seed=r["seed"], method=r["method"],
                   T_ref_used=r["T_ref_used"], ref_gap=r["ref_gap"],
                   ref_stable=int(r["ref_stable"]))
        any_c = 0
        for tol in tols:
            v = r.get(f"T_conv_tol{tol}")
            # T_conv == 1 means the state started already inside the tolerance
            floor = v == 1
            rec[f"T_conv_tol{tol:g}"] = None if (v is None or floor) else v
            rec[f"floor_limited_tol{tol:g}"] = int(bool(floor))
            any_c |= int(v is None)
        rec["censored"] = any_c
        out.append(rec)
    return out


def table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            "-" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def stats_block(recs, key, label, methods=("pc", "pcalm")):
    md, fits = [], []
    for m in methods:
        st = depth_stats(recs, key, m)
        md.append(f"\n**{label} — {m.upper()}** (mean ± SD over seeds; "
                  f"censored = excluded, counted)\n")
        md.append(table(st, ["depth", "n_ok", "n_censored", "mean", "sd", "min", "max"],
                        ["depth", "n", "censored", "mean", "SD", "min", "max"]))
        pts = [(r["depth"], r[key]) for r in recs
               if r["method"] == m and r.get(key) is not None and not r.get("censored", 0)]
        f = loglog_fit([p[0] for p in pts], [p[1] for p in pts])
        md.append(f"\nlog-log fit: `{fmt(f)}`\n")
        fits.append((m, f, st))
    return "\n".join(md), fits


def seed_vs_depth_note(st):
    rows = [r for r in st if r["sd"] is not None and r["mean"] is not None]
    if len(rows) < 2:
        return None
    sds = [r["sd"] for r in rows]
    means = [r["mean"] for r in rows]
    rel_sd = float(np.median([s / m for s, m in zip(sds, means) if m]))
    fold = max(means) / min(means)
    return dict(median_rel_sd=rel_sd, depth_fold_change=fold,
                comparable=rel_sd > 0.5 * np.log(fold) / max(len(rows) - 1, 1))


def pick_res_tols(resid_recs_dir="resid"):
    """Choose 3 tolerances spanning ~1 order of magnitude from the OBSERVED
    residual range, then apply them uniformly to every depth/method/seed."""
    curves = []
    for p in sorted((RES / resid_recs_dir).glob("*.npz")):
        z = np.load(p)
        curves.append((p.name, z["init"], z["trained"]))
    if not curves:
        return [3e-3, 1e-3, 3e-4], {}
    t1 = np.median([c[1][0] for c in curves])
    mins = np.array([min(c[1].min(), c[2].min()) for c in curves])
    hi = 10 ** np.floor(np.log10(t1))          # just under the T=1 residual
    lo = hi / 10.0
    tols = [hi, hi / 3.1623, lo]
    return tols, dict(median_resid_at_T1=float(t1),
                      min_observed=float(mins.min()), max_observed_min=float(mins.max()))


PAPER_ETA_DEPTHS = {8, 16, 32, 64, 128}


def build_conv2():
    out = []
    for r in load_dir("conv2"):
        rec = dict(depth=r["depth"], seed=r["seed"], method=r["method"],
                   B_final=r["B_final"], self_consistent=r["self_consistent"],
                   margin=r.get("margin_achieved"), dist_at_T1=r.get("dist_at_T1"))
        anyc = 0
        for t in (0.1, 0.03, 0.01):
            v = r.get(f"T_conv_tol{t}")
            floor = bool(r.get(f"floor_limited_tol{t}"))
            rec[f"T_conv_tol{t:g}"] = None if (v is None or floor) else v
            rec[f"floor_limited_tol{t:g}"] = int(floor)
            anyc |= int(v is None)
        rec["censored"] = anyc or (not r["self_consistent"])
        out.append(rec)
    return out


def restricted_fit(recs, key, method):
    """Same fit but only at depths where state_lr comes from the paper's frozen
    table, so the two extrapolated endpoints cannot drive the slope."""
    pts = [(r["depth"], r[key]) for r in recs
           if r["method"] == method and r.get(key) is not None
           and not r.get("censored", 0) and r["depth"] in PAPER_ETA_DEPTHS]
    return loglog_fit([p[0] for p in pts], [p[1] for p in pts])


def variance_verdict(st):
    """Is within-depth seed spread comparable to the across-depth effect?"""
    rows = [r for r in st if r["sd"] is not None and r["mean"]]
    if len(rows) < 2:
        return "not assessable (too few uncensored depths)"
    rel = [r["sd"] / r["mean"] for r in rows]
    med_rel = float(np.median(rel))
    means = [r["mean"] for r in rows]
    fold = max(means) / min(means)
    n_steps = len(rows) - 1
    per_step = fold ** (1.0 / n_steps) - 1.0 if n_steps else 0.0
    if med_rel >= per_step:
        return (f"**SEED VARIANCE COMPARABLE TO DEPTH EFFECT**: median within-depth "
                f"relative SD = {med_rel:.0%}, vs {per_step:.0%} mean change per "
                f"depth doubling. The depth trend is NOT cleanly separated from "
                f"seed noise for this metric.")
    return (f"seed spread smaller than depth effect (median relative SD "
            f"{med_rel:.0%} vs {per_step:.0%} per depth step)")


def conv_b_dependence(conv2):
    """T_conv must not depend on the reference budget B. Where it does, the
    reference solve was not converged far enough and T_conv is not a
    measurement."""
    out = []
    for L in sorted({r["depth"] for r in conv2}):
        rs = [r for r in conv2 if r["depth"] == L and r.get("T_conv_tol0.01")]
        by = {}
        for r in rs:
            by.setdefault(r["B_final"], []).append(r["T_conv_tol0.01"])
        bs = sorted(by)
        if len(bs) > 1:
            lo, hi = float(np.mean(by[bs[0]])), float(np.mean(by[bs[-1]]))
            out.append(dict(depth=L, n_distinct_B=len(bs),
                            B_ratio=round(bs[-1] / bs[0], 1),
                            T_conv_ratio=round(hi / lo, 2), B_dependent="YES"))
        else:
            out.append(dict(depth=L, n_distinct_B=1, B_ratio=1.0, T_conv_ratio=None,
                            B_dependent="untestable (single B)"))
    return out


QUANT_NOTE = (
    "> **On the SD column.** The budget ladder is geometric (ratio ~1.4), so when "
    "every seed falls in the same ladder bin the reported SD is 0. That means the "
    "seed-to-seed spread is *smaller than one ladder step*, not that it is zero. "
    "Read SD=0 as an upper bound of roughly 40% of the mean.\n")


def divergence_audit(rows):
    out = []
    for m in ("bp", "pc", "pcalm"):
        for L in sorted({r["depth"] for r in rows}):
            rs = [r for r in rows if r["method"] == m and r["depth"] == L]
            if not rs:
                continue
            nonfinite = [r for r in rs if not r["finite"]]
            chance = [r for r in rs if r["test_acc"] <= CHANCE]
            best = max(r["test_acc"] for r in rs)
            out.append(dict(method=m, depth=L, n_runs=len(rs),
                            n_nonfinite=len(nonfinite),
                            n_at_or_below_chance=len(chance),
                            best_test_acc=round(best, 4),
                            max_T_run=max(r["budget"] for r in rs)))
    return out


def main():
    rows = load_sweep()
    RES.mkdir(exist_ok=True)
    write_csv(rows, RES / "raw_runs.csv")
    lit, tgt = build_task(rows)
    write_csv(lit, RES / "t_task_literal.csv")
    write_csv(tgt, RES / "t_target_bp_relative.csv")
    tols, tol_info = pick_res_tols()
    resid = build_resid(tols)
    write_csv(resid, RES / "t_res.csv")
    conv = build_conv()
    write_csv(conv, RES / "t_conv_loose_reference.csv")
    conv2 = build_conv2()
    write_csv(conv2, RES / "t_conv.csv")

    fits = []
    md = ["# Inner-iteration scaling with depth: PC vs PC-ALM", "",
          "Measurement only. No update rule, hyperparameter or algorithm was altered;",
          "all dynamics are imported verbatim from `pcalm/`.", "",
          "## Provenance", "",
          f"- reference implementation: SakanaAI/pc-alm (MIT), arXiv 2605.31022",
          f"- reproduction gate: Fashion-MNIST N=32 L=32 relu seed 0, T=2L -> "
          f"BP 78.66% / PC 68.13% / PC-ALM 77.75%, matching the repo's published "
          f"table exactly (0.00 pp on all three)",
          f"- config: MNIST, residual MLP, width 32 (their default), relu, 1 epoch, "
          f"batch 64, Adam lr = 1e-3*sqrt(W/L), gamma0=1, full 60k/10k split",
          f"- `state_lr` (eta_h): paper's frozen eta_best_by_cell.csv at depths "
          f"8-128; depths 4 and 256 via the paper's own eta=1/lambda_max rule "
          f"(calibrated power iteration). Fits are reported both over all depths "
          f"and restricted to the frozen-table depths.",
          f"- total training runs: **{len(rows)}**  |  residual cells: "
          f"**{len(resid)//2}**  |  conv cells: **{len(conv2)}**", ""]

    md += ["## Headline caveats", "",
           "1. **Metric (a) as specified is degenerate.** `free_init` sets activities to",
           "   the forward pass, so the constraint residual is *exactly 0* at T=0. For PC",
           "   it then *grows* to a nonzero equilibrium - those equilibrium residuals are",
           "   PC's error signals. `resid_argmin == 1` in every PC cell at both init and",
           "   trained parameters, so \"first T below tol\" is either 1 or never, at every",
           "   depth. PC is fully censored below; this is a property of the method, not a",
           "   run failure.",
           "2. **PC-ALM has no fixed point in T** at the paper's defaults (alpha=1, rho=1,",
           "   inner_steps=1). Its gradient cosine to its own T=100k limit peaks near",
           "   T~1024 then decays; duals grow without settling. It is censored in every",
           "   convergence cell. PC-ALM is accurate at a *budget*, not in a limit.",
           "3. Consequently the load-bearing comparison is the task metric, which is",
           "   well-posed for both methods.", ""]

    md += ["## (b) T_task - literal: first T within 1% absolute of accuracy at largest T", ""]
    b, fl = stats_block(lit, "T_task", "T_task (literal)")
    md.append(b)
    md.append(QUANT_NOTE)
    for m, f, st in fl:
        rf = restricted_fit(lit, "T_task", m)
        md.append(f"- {m.upper()} restricted to paper-frozen eta depths: `{fmt(rf)}`")
        md.append(f"- {m.upper()} seed-vs-depth: {variance_verdict(st)}")
        fits.append(dict(metric="T_task_literal", method=m, tolerance="1% abs",
                         **({} if f is None else {k: round(v, 4) for k, v in f.items()})))
    md.append("\n### Plateau / censoring audit\n")
    aud = []
    for m in ("pc", "pcalm"):
        for L in sorted({r["depth"] for r in lit}):
            rs = [r for r in lit if r["method"] == m and r["depth"] == L]
            if rs:
                aud.append(dict(method=m, depth=L, n=len(rs), T_ref=rs[0]["T_ref"],
                                plateau_reached=sum(r["plateau_reached"] for r in rs),
                                censored=sum(r["censored"] for r in rs),
                                acc_at_T_ref=round(float(np.mean(
                                    [r["acc_at_T_ref"] for r in rs])), 4)))
    md.append(table(aud, ["method", "depth", "n", "T_ref", "plateau_reached",
                          "censored", "acc_at_T_ref"]))

    md += ["", "## (b') T_task - robust: first T reaching a fraction of depth-matched BP", ""]
    for fr in BP_FRACS:
        sub = [r for r in tgt if r["frac"] == fr]
        b, fl = stats_block(sub, "T_target", f"T_target @ {int(fr*100)}% of BP")
        md.append(b)
        md.append(QUANT_NOTE)
        for m, f, st in fl:
            rf = restricted_fit(sub, "T_target", m)
            md.append(f"- {m.upper()} restricted to frozen-eta depths: `{fmt(rf)}`")
            md.append(f"- {m.upper()} seed-vs-depth: {variance_verdict(st)}")
            fits.append(dict(metric="T_target", method=m, tolerance=f"{int(fr*100)}%_of_BP",
                             **({} if f is None else {k: round(v, 4) for k, v in f.items()})))

    md += ["", "## (a) T_res - literal constraint-residual threshold", "",
           f"Tolerances {', '.join(f'{t:.3g}' for t in tols)} span one order of magnitude,",
           f"chosen from the observed residual range ({tol_info}) and then applied",
           "uniformly to every depth, method, seed and parameter point.", ""]
    for tag in ("init", "trained"):
        for tol in tols:
            k = f"T_res_tol{tol:g}"
            sub = [dict(r, censored=r.get(f"censored_tol{tol:g}", 0))
                   for r in resid if r["point"] == tag]
            b, fl = stats_block(sub, k, f"T_res @ tol={tol:.3g} ({tag} params)")
            md.append(b)
            for m, f, st in fl:
                md.append(f"- {m.upper()} seed-vs-depth: {variance_verdict(st)}")
                fits.append(dict(metric=f"T_res_{tag}", method=m, tolerance=f"{tol:.3g}",
                                 **({} if f is None else
                                    {kk: round(v, 4) for kk, v in f.items()})))

    md += ["", "## (a') T_conv - EXPLORATORY ONLY, no exponent quoted", "",
           "This was my own substitute for the degenerate metric (a), not one of the two",
           "requested measurements. **It failed validation; no slope should be read off",
           "it.** T_conv is measured against the activity state after a reference solve",
           "of budget B, and a valid measurement must be independent of B. It is not:", ""]
    md.append(table(conv_b_dependence(conv2),
                    ["depth", "n_distinct_B", "B_ratio", "T_conv_ratio", "B_dependent"]))
    md += ["",
           "At depth 64 one seed stopped at B=131072 and reports T_conv=43329 while two",
           "others ran to B=524288 and report ~99400 - a 2.3x spread driven entirely by",
           "how far the reference solve ran, not by the seed. The apparent seed variance",
           "at depths 16 and 64 is therefore mostly this artefact, not seed noise.", "",
           "Cause: PC's inner loop has slow modes, and float32 accumulation over",
           "million-step scans stops the fixed point being pinned down at L>=16. The",
           "margin needed grows with depth; 3x is not enough. Raising it costs ~10x more",
           "compute with no guarantee of convergence, and float64 would change the",
           "numerics of the object being measured.", "",
           "Raw values are in `t_conv.csv`. What survives qualitatively: PC does reach a",
           "fixed point and its iteration count grows steeply with depth; PC-ALM never",
           "stabilises at any depth or seed (30/30 censored). Neither statement needs a",
           "slope. T_conv is excluded from fits.csv.", ""]
    for tol in (0.1, 0.03, 0.01):
        st = depth_stats(conv2, f"T_conv_tol{tol:g}", "pc")
        md.append(f"\n**T_conv @ tol={tol} (PC, EXPLORATORY - not a measurement)**\n")
        md.append(table(st, ["depth", "n_ok", "n_censored", "mean", "sd", "min", "max"],
                        ["depth", "n", "censored", "mean", "SD", "min", "max"]))
    md.append("")

    md += ["", "## Divergence / failure audit", "",
           "No depth is silently dropped. `n_nonfinite` counts runs with non-finite",
           "loss or accuracy; `n_at_or_below_chance` counts runs that never learned.", ""]
    md.append(table(divergence_audit(rows),
                    ["method", "depth", "n_runs", "n_nonfinite",
                     "n_at_or_below_chance", "best_test_acc", "max_T_run"]))

    md += ["", "## Plain-language summary of the slopes", "",
           "What the exponent means: T ~ depth^slope, so slope 1 means doubling depth",
           "doubles the inner iterations needed; slope 2 means it quadruples them.",
           "No judgement is offered here about whether any slope is good.", ""]
    for fr in BP_FRACS:
        sub = [r for r in tgt if r["frac"] == fr]
        line = [f"- **At {int(fr*100)}% of BP accuracy:**"]
        for m in ("pc", "pcalm"):
            pts = [(r["depth"], r["T_target"]) for r in sub if r["method"] == m
                   and r["T_target"] is not None and not r["censored"]]
            f = loglog_fit([x for x, _ in pts], [y for _, y in pts])
            nc = sum(1 for r in sub if r["method"] == m and r["censored"])
            line.append(f"  - {m.upper()}: slope {f['slope']:+.3f} "
                        f"(95% CI [{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}], R2={f['r2']:.3f})"
                        f"{f', {nc} censored cell(s)' if nc else ''}"
                        if f else f"  - {m.upper()}: not measurable")
        md += line
    md += ["",
           "In words: PC's inner-iteration requirement grows about quadratically with",
           "depth (slope near 2); PC-ALM's grows about linearly, with an exponent near",
           "1.25 rather than exactly 1. The two confidence intervals are far apart at",
           "every threshold. Metric (a) yields no exponent for PC at any tolerance",
           "because its residual never decreases (see caveat 1); for PC-ALM metric (a)",
           "is roughly flat in depth, and its seed spread is large enough that the",
           "depth trend there is not separable from seed noise.", "",
           "**Depth 256 was dropped from both task metrics.** Its budget ladder alone",
           "costs ~36 h of serial CPU (~5 h even at 7x parallelism) against ~2.5 h for",
           "depths 4-128 combined. It is retained in the residual measurement (a),",
           "where it needs one training run per cell rather than a full ladder.", ""]

    write_csv(fits, RES / "fits.csv")
    (ROOT / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"wrote REPORT.md ({len(md)} lines) and CSVs")


if __name__ == "__main__":
    main()
