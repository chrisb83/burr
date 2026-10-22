"""Accuracy benchmark: scipy.stats.burr (installed SciPy) vs burr_new vs mpmath.

For a grid of shape parameters (c, d) and arguments that reach far into both
tails, every method (pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf) is compared
with an mpmath reference (defining formulas at 400 digits). Errors are reported
in ulps,

    ulps = |computed - reference| / (|reference| * eps),   eps = 2.2e-16,

so 1 ulp is a relative error of 2.2e-16 and 1e6 ulps is 2.2e-10. A "fail" is a
result that is 0, inf or nan although the reference is a normal float. Points
whose reference is not a normal float (underflow/overflow) are skipped.

Usage
-----
    python burr_benchmark.py                 # summary tables
    python burr_benchmark.py --verbose       # plus worst case per (c, d)
    python burr_benchmark.py --plot err.png  # plus error-vs-argument plots

Notes on reading the results
----------------------------
* A few hundred ulps where results are extreme (say 1e-200) are expected for any
  double-precision algorithm: there the result changes by about |log(result)|
  ulps when d changes by one ulp, so that error is within the conditioning.
* Relative error of logcdf/logsf/logpdf is measured as well. Near zero it is
  stricter than what most users need, but it matters for logcdf because sf and
  logsf are computed from it.
"""
import argparse
import warnings

import numpy as np
import mpmath as mp
import scipy
from scipy import stats

from burr_new import burr as burr_new

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
HUGE = np.finfo(float).max

IMPLS = {f"scipy {scipy.__version__}": stats.burr, "new": burr_new}
X_FUNCS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]
Q_FUNCS = ["ppf", "isf"]
PARAMS = [(c, d) for c in [0.3, 1.0, 2.0, 10.0, 100.0]
          for d in [1e-3, 0.5, 1.0, 3.0, 1e3, 1e6]]
PLOT_PARAMS = [(2.0, 3.0), (0.5, 1e-3), (3.0, 1e4)]


# ---------------------------------------------------------------- references

# The references use only the defining formulas, deliberately not the
# log1p/expm1 rewrites of the implementation, so an algebra slip cannot be
# reproduced. Cancellation such as 1 - cdf is resolved by precision instead:
# any difference that is still a normal double (>= 2.2e-308) is resolved with
# ~330 digits, so the default of 400 digits leaves a wide margin.

def ref_x(x, c, d):
    """Reference pdf, logpdf, cdf, logcdf, sf, logsf at one point."""
    x, c, d = mp.mpf(x), mp.mpf(c), mp.mpf(d)   # exact conversion of floats
    t = x ** (-c)
    pdf = c * d * x ** (-c - 1) * (1 + t) ** (-d - 1)
    cdf = (1 + t) ** (-d)
    sf = 1 - cdf
    return {"pdf": pdf, "logpdf": mp.log(pdf), "cdf": cdf,
            "logcdf": mp.log(cdf), "sf": sf, "logsf": mp.log(sf)}


def ref_q(q, c, d):
    """Reference ppf(q) and isf(q) from x = (F**(-1/d) - 1)**(-1/c)."""
    q, c, d = mp.mpf(q), mp.mpf(c), mp.mpf(d)
    ppf = (q ** (-1 / d) - 1) ** (-1 / c)
    isf = ((1 - q) ** (-1 / d) - 1) ** (-1 / c)
    return {"ppf": ppf, "isf": isf}


def check_reference():
    """Sanity checks of the reference: round trips, and stability in dps."""
    for c, d in [(2.0, 3.0), (0.3, 1e-3), (100.0, 1e6)]:
        for q in [1e-300, 1e-10, 0.3, 1 - 1e-12]:
            r = ref_q(q, c, d)
            for name in ["ppf", "isf"]:
                x = r[name]
                if not mp.isfinite(x) or x == 0:
                    continue
                back = ref_x(x, c, d)["cdf" if name == "ppf" else "sf"]
                assert abs(back / mp.mpf(q) - 1) < mp.mpf(10) ** (-30), (c, d, q, name)
    # the hardest cancellations: cdf ~ 1 - 1e-300 and pdf ~ 1 - 1e-72
    for args in [(1e100, 3.0, 1.0), (1e-64, 1.0, 1.0), (1e-200, 2.0, 3.0)]:
        lo = ref_x(*args)
        with mp.workdps(mp.mp.dps + 100):
            hi = ref_x(*args)
        for k in lo:
            if lo[k] != 0 and hi[k] != 0:
                assert abs(lo[k] / hi[k] - 1) < mp.mpf(10) ** (-30), (args, k)


# ---------------------------------------------------------------- grids

def x_grid(c):
    # x**-c spans 1e-320 .. 1e320, plus points around the branch at x = 1
    with np.errstate(over="ignore"):
        x = 10.0 ** (-np.linspace(-320, 320, 81) / c)
    x = x[(x >= 1e-300) & (x <= 1e300)]
    near_one = [0.5, 0.9, 1 - 1e-6, 1.0, 1 + 1e-6, 1.1, 2.0]
    return np.unique(np.concatenate([x, near_one]))


def q_grid():
    return np.unique(np.concatenate([
        10.0 ** np.linspace(-300, -3, 30),
        np.linspace(0.01, 0.99, 15),
        1 - 10.0 ** np.linspace(-15.5, -3, 20)]))


def x_region(cdf, sf):
    return "left tail" if cdf < 1e-3 else "right tail" if sf < 1e-3 else "bulk"


def q_region(q):
    return "q < 1e-3" if q < 1e-3 else "q > 1-1e-3" if q > 1 - 1e-3 else "bulk"


# ---------------------------------------------------------------- evaluation

def ulps(got, ref):
    """Error in ulps; nan where the reference is not a normal float."""
    ref = np.asarray(ref, dtype=float)
    valid = np.isfinite(ref) & (np.abs(ref) >= TINY) & (np.abs(ref) <= HUGE)
    with np.errstate(all="ignore"):
        e = np.abs(got - ref) / (np.abs(ref) * EPS)
    e = np.where(np.isfinite(got) & (got != 0), e, np.inf)   # total failures
    return np.where(valid, e, np.nan)


def evaluate(dist, name, args, c, d):
    """Call dist.<name>; return values and whether a RuntimeWarning was emitted."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = np.asarray(getattr(dist, name)(args, c, d), dtype=float)
    return out, any(issubclass(wi.category, RuntimeWarning) for wi in w)


def run(params):
    """Collect one record per (function, impl, c, d): args, regions, errors."""
    records = []
    qs = q_grid()
    for c, d in params:
        xs = x_grid(c)
        rx = [ref_x(x, c, d) for x in xs]
        rq = [ref_q(q, c, d) for q in qs]
        x_reg = np.array([x_region(r["cdf"], r["sf"]) for r in rx])
        q_reg = np.array([q_region(q) for q in qs])
        for fname in X_FUNCS + Q_FUNCS:
            is_x = fname in X_FUNCS
            args, refs, regs = (xs, rx, x_reg) if is_x else (qs, rq, q_reg)
            ref = np.array([float(r[fname]) for r in refs])
            for iname, dist in IMPLS.items():
                got, warned = evaluate(dist, fname, args, c, d)
                records.append(dict(func=fname, impl=iname, c=c, d=d, args=args, ref=ref,
                                    region=regs, err=ulps(got, ref), warned=warned))
    return records


# ---------------------------------------------------------------- reporting

def fmt(v):
    if np.isnan(v):
        return "-"
    if np.isinf(v):
        return "inf"
    return f"{v:.3g}" if v < 1e5 else f"{v:.1e}"


def summary(records):
    impls = list(IMPLS)
    regions = {f: (["left tail", "bulk", "right tail"] if f in X_FUNCS
                   else ["q < 1e-3", "bulk", "q > 1-1e-3"]) for f in X_FUNCS + Q_FUNCS}
    head = f"{'':8}{'region':12}{'n':>5}" + "".join(
        f" | {name:>14} max   median  fails" for name in impls)
    print(head)
    print("-" * len(head))
    for f in X_FUNCS + Q_FUNCS:
        warn = {i: sum(r["warned"] for r in records if r["func"] == f and r["impl"] == i)
                for i in impls}
        for k, reg in enumerate(regions[f]):
            cells, n = [], 0
            for i in impls:
                e = np.concatenate([r["err"][r["region"] == reg] for r in records
                                    if r["func"] == f and r["impl"] == i])
                e = e[~np.isnan(e)]
                n = e.size
                if n == 0:
                    cells.append(f" | {'-':>18} {'-':>8} {'-':>6}")
                    continue
                fails = int(np.isinf(e).sum())
                finite = e[np.isfinite(e)]
                cells.append(f" | {fmt(e.max()):>18} {fmt(np.median(finite) if finite.size else np.inf):>8}"
                             f" {fails:>6}")
            print(f"{f if k == 0 else '':8}{reg:12}{n:>5}" + "".join(cells))
        notes = [f"{i}: RuntimeWarning in {w}/{len(PARAMS)} (c, d) sets"
                 for i, w in warn.items() if w]
        if notes:
            print(f"{'':8}" + "; ".join(notes))
        print()


def regressions(records, limit=8):
    """Points where the new version is clearly less accurate than SciPy."""
    old, new = list(IMPLS)
    print(f"Points where '{new}' is worse than '{old}' by more than max(2x, 8 ulp):")
    any_found = False
    for f in X_FUNCS + Q_FUNCS:
        rows = []
        for c, d in PARAMS:
            (ro,) = [r for r in records if (r["func"], r["impl"], r["c"], r["d"]) == (f, old, c, d)]
            (rn,) = [r for r in records if (r["func"], r["impl"], r["c"], r["d"]) == (f, new, c, d)]
            eo, en = ro["err"], rn["err"]
            bad = ~np.isnan(en) & (en > np.maximum(2 * eo, eo + 8))
            rows += [(en[k], eo[k], c, d, ro["args"][k], ro["ref"][k])
                     for k in np.flatnonzero(bad)]
        if rows:
            any_found = True
            rows.sort(reverse=True)
            print(f"  {f}: {len(rows)} points; worst:")
            for en, eo, c, d, a, ref in rows[:limit]:
                print(f"    c={c:<6g} d={d:<7g} arg={a:<11.4g} value={ref:<11.4g}"
                      f" {old}: {fmt(eo):>8}  {new}: {fmt(en):>8} ulp")
    if not any_found:
        print("  none")
    else:
        print("  For pdf, cdf, sf, ppf and isf, an error below |log(value)| ulp is within the\n"
              "  conditioning: changing d by one ulp moves the value by about that much.")
    print()


def verbose(records):
    print("Worst case per (c, d) [ulps]; 'inf' = 0/inf/nan where the reference is finite\n")
    for f in X_FUNCS + Q_FUNCS:
        print(f"{f}:")
        print(f"  {'c':>6} {'d':>7}" + "".join(f"{i:>16}" for i in IMPLS))
        for c, d in PARAMS:
            row = []
            for i in IMPLS:
                (r,) = [r for r in records if (r["func"], r["impl"], r["c"], r["d"]) == (f, i, c, d)]
                e = r["err"][~np.isnan(r["err"])]
                row.append(fmt(e.max()) if e.size else "-")
            print(f"  {c:>6g} {d:>7g}" + "".join(f"{v:>16}" for v in row))
        print()


def plot(path, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference categorical palette, slots 1-2; markers give a second encoding
    style = {list(IMPLS)[0]: dict(color="#eb6834", marker="x", ms=4),
             list(IMPLS)[1]: dict(color="#2a78d6", marker="o", ms=3, mfc="none")}
    ink, muted, surface = "#1f1f1e", "#6b6a64", "#fcfcfb"
    fail_y, top = 3e17, 1e18

    records = run(params)
    funcs = X_FUNCS + Q_FUNCS
    fig, axes = plt.subplots(len(funcs), len(params), figsize=(4.2 * len(params), 2.0 * len(funcs)),
                             sharey=True, squeeze=False, facecolor=surface)
    for col, (c, d) in enumerate(params):
        for row, f in enumerate(funcs):
            ax = axes[row, col]
            ax.set_facecolor(surface)
            ax.axhspan(1e17, top, color="#e9e8e3", lw=0)
            for i in IMPLS:
                (r,) = [r for r in records if (r["func"], r["impl"], r["c"], r["d"]) == (f, i, c, d)]
                a = r["args"]
                u = np.log10(a) if f in X_FUNCS else np.log10(a) - np.log10(1 - a)
                e = r["err"]
                ok = np.isfinite(e)
                y = np.where(ok, np.maximum(e, 0.1), np.nan)
                ax.plot(u, y, ls="none", label=i, **style[i])
                fails = np.isinf(e)
                ax.plot(u[fails], np.full(fails.sum(), fail_y), ls="none", **style[i])
            if f in X_FUNCS:
                ax.axvline(0, color=muted, lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.set_ylim(0.1, top)
            ax.set_yticks([1, 1e3, 1e6, 1e9, 1e12, 1e15])
            ax.grid(axis="y", color="#e4e3dd", lw=0.5)
            for s in ax.spines.values():
                s.set_color("#c9c8c1")
            ax.tick_params(colors=muted, labelsize=7)
            if col == 0:
                ax.set_ylabel(f"{f}\nerror [ulp]", color=ink, fontsize=8)
            if row == 0:
                ax.set_title(f"c = {c:g}, d = {d:g}", color=ink, fontsize=9)
            if row == len(X_FUNCS) - 1 or row == len(funcs) - 1:
                ax.set_xlabel("log10 x" if f in X_FUNCS else "log10(q / (1 - q))",
                              color=muted, fontsize=8)
            ax.text(1.01, fail_y, "fail", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6, color=muted)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(IMPLS), frameon=False,
               fontsize=9, labelcolor=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130, facecolor=surface)
    print(f"plot written to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verbose", action="store_true", help="worst case per (c, d)")
    p.add_argument("--plot", metavar="FILE", help="write error-vs-argument plots")
    p.add_argument("--dps", type=int, default=400, help="mpmath digits (default 400)")
    a = p.parse_args()

    mp.mp.dps = a.dps
    check_reference()
    print(__doc__.split("Usage")[0].strip(), "\n")
    print(f"(c, d) grid: c in {sorted({c for c, _ in PARAMS})}, "
          f"d in {sorted({d for _, d in PARAMS})}\n")
    records = run(PARAMS)
    summary(records)
    regressions(records)
    if a.verbose:
        verbose(records)
    if a.plot:
        plot(a.plot, PLOT_PARAMS)


if __name__ == "__main__":
    main()
