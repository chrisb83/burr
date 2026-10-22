"""Worked examples: where the initial proposal falls short of the new burr code.

Three implementations are compared at a handful of points against an mpmath
reference (defining formulas at 400 digits):

* "scipy 1.18"  - scipy.stats.burr as installed
* "initial"     - the first proposed fix (_log1p_pow / _log1mexp helpers,
                  np.where fallbacks in _cdf and _ppf; _pdf, _logpdf and _isf
                  unchanged from SciPy)
* "new"         - burr_new.py

Each example states the mechanism behind the error, so the numbers can be read
as evidence for one specific change. Run:  python burr_examples.py
"""
import warnings

import numpy as np
import mpmath as mp
import scipy
import scipy.special as sc
from scipy import stats
from scipy.stats._continuous_distns import burr_gen as _scipy_burr_gen
import scipy._external.array_api_extra as xpx

from burr_new import burr as burr_new
from burr_benchmark import ref_x, ref_q

EPS = np.finfo(float).eps


# ------------------------------------------------- the initial proposal (verbatim)

def _log1p_pow(x, c):
    # log1p(x**c), as c*log(x) + log1p(x**-c) where x**c > 1 so that
    # x**c cannot overflow
    return xpx.apply_where(
        c * np.log(x) > 0, (x, c),
        lambda x, c: c * np.log(x) + sc.log1p(x**-c),
        lambda x, c: sc.log1p(x**c))


def _log1mexp(x):
    # log(1 - exp(x)) for x < 0
    return xpx.apply_where(
        x < -np.log(2), x,
        lambda x: np.log1p(-np.exp(x)),
        lambda x: np.log(-np.expm1(x)))


def _log_expm1(y):
    # log(expm1(y)) for y > 0, finite where expm1(y) overflows
    return y + _log1mexp(-y)


class burr_initial_gen(_scipy_burr_gen):
    # _pdf, _logpdf and _isf are inherited unchanged from SciPy

    def _cdf(self, x, c, d):
        # x**-c overflows to inf for small x, where the log form holds
        with np.errstate(over='ignore'):
            cdf = (1 + x**(-c))**(-d)
        return np.where(cdf == 0, np.exp(self._logcdf(x, c, d)), cdf)

    def _logcdf(self, x, c, d):
        return -d * _log1p_pow(x, -c)

    def _sf(self, x, c, d):
        return -sc.expm1(self._logcdf(x, c, d))

    def _logsf(self, x, c, d):
        return _log1mexp(self._logcdf(x, c, d))

    def _ppf(self, q, c, d):
        # q**(-1/d) overflows to inf for small q, where the log form holds
        with np.errstate(over='ignore'):
            ppf = (q**(-1.0/d) - 1)**(-1.0/c)
        return np.where(ppf == 0, np.exp(-_log_expm1(-np.log(q) / d) / c), ppf)


burr_initial = burr_initial_gen(a=0.0, name='burr')

IMPLS = {f"scipy {scipy.__version__}": stats.burr,
         "initial": burr_initial,
         "new": burr_new}


# ------------------------------------------------- examples

EXAMPLES = [
    dict(func="logpdf", arg=1e-200, c=2.0, d=3.0,
         title="logpdf in the left tail: overflow of x**-c",
         why="SciPy and the initial version evaluate xlog1py(d+1, x**-c). For\n"
             "x = 1e-200, x**-c = 1e400 overflows, so logpdf = -inf. The new\n"
             "version uses (c*d-1)*log(x) - (d+1)*log1p(x**c) for x < 1, where\n"
             "x**c cannot overflow."),
    dict(func="pdf", arg=1e-160, c=2.0, d=0.6,
         title="pdf in the left tail: inf / inf = nan",
         why="pdf = c*d*x**(-c-1) / (1 + x**-c)**(d+1). Both power terms overflow,\n"
             "so the quotient is nan although the true pdf is 1.2e-32. The new\n"
             "version uses the equivalent c*d*x**(c*d-1) * (1 + x**c)**(-d-1) for\n"
             "x < 1, where x**c cannot overflow."),
    dict(func="logpdf", arg=1e-10, c=1.0, d=1.0,
         title="logpdf for c*d = 1 (here c = d = 1): cancellation",
         why="x != 0, so SciPy uses log(c) + log(d) + xlogy(-c-1, x)\n"
             "- xlog1py(d+1, x**-c). The last two terms are 46.0517018598809 and\n"
             "46.0517018600809, each rounded to ~5e-15, and cancel to -2e-10.\n"
             "For small x their difference is (c*d-1)*log(x) - (d+1)*log1p(x**c),\n"
             "so the log(x) parts cancel exactly whenever c*d = 1 (c=2, d=0.5 loses\n"
             "every digit). The new x < 1 form computes (c*d-1)*log(x) directly."),
    dict(func="logpdf", arg=1e-10, c=2.0, d=0.5,
         title="logpdf for c*d = 1 (c = 2, d = 0.5): every digit cancels",
         why="Same branch as above; log(c) + log(d) = log(2) + log(0.5) = 0 exactly.\n"
             "The two large terms, exact values (truncated):\n"
             "  xlogy(-3, 1e-10)       = 69.077552789821370411243151694037703...\n"
             "  xlog1py(1.5, 1e20)     = 69.077552789821370411258151694037703...\n"
             "  difference             = -1.5e-20\n"
             "They first differ in the 22nd significant digit, but a double keeps\n"
             "~16: the spacing of doubles near 69 is 1.4e-14. Both round to the same\n"
             "double, 69.07755278982137, so SciPy returns logpdf = 0.0 (pdf = 1).\n"
             "The new form: (c*d-1)*log(x) = 0*log(x) = 0, minus\n"
             "(d+1)*log1p(x**c) = 1.5*log1p(1e-20) = 1.5e-20, gives -1.5e-20."),
    dict(func="pdf", arg=100.0, c=3.0, d=1e6,
         title="pdf with large d: power form loses d ulp",
         why="(1 + x**-c)**(d+1) multiplies the rounding error of 1 + x**-c (up to\n"
             "1/2 ulp) by d + 1 = 1e6. The new version evaluates this factor as\n"
             "exp(-(d+1)*log1p(x**-c)), which loses at most ~min(d, 745) ulp\n"
             "(here ~1, since (d+1)*log1p(x**-c) = 1)."),
    dict(func="pdf", arg=1300.0, c=100.0, d=1e6,
         title="pdf with large d: subnormal intermediate",
         why="x**(-c-1) = 1300**-101 = 3e-315 is subnormal and keeps only ~30 of\n"
             "53 bits; c*d = 1e8 lifts the product back to a normal 3e-307, so the\n"
             "lost bits show. Where (c+1)*log(x) > 708 the new version evaluates\n"
             "the whole pdf in log space. Its remaining ~100 ulp are below\n"
             "|log pdf| = 705, i.e. within the conditioning of the result."),
    dict(func="cdf", arg=1e3, c=3.0, d=1e4,
         title="cdf near 1 with large d: power form loses d ulp",
         why="(1 + x**-c)**-d: the rounding error of 1 + x**-c (up to 1/2 ulp) is\n"
             "multiplied by d = 1e4. The initial version always uses this form;\n"
             "the new one uses exp(logcdf) for x > 1, where |logcdf| <= d*log(2)\n"
             "and here |logcdf| = 1e-5, so the error stays at about 1 ulp."),
    dict(func="logsf", arg=1e-20, c=2.0, d=3.0,
         title="logsf in the left tail: exp(logcdf) vs the power form",
         why="logsf = log1p(-cdf) with cdf = 1e-120. The initial version gets cdf\n"
             "as exp(logcdf) with logcdf = -276, which can cost up to ~|logcdf|\n"
             "ulp. The new version uses its _cdf there (power form, ~1 ulp).\n"
             "SciPy 1.18 is fine here: it also uses the power form."),
    dict(func="ppf", arg=1 - 1e-8, c=3.0, d=1e6,
         title="ppf near q = 1 with large d: cancellation in q**(-1/d) - 1",
         why="q**(-1/d) = 1 + 1e-14, so subtracting 1 leaves only a few correct\n"
             "digits. The fallback in the initial version only triggers when the\n"
             "result is exactly 0. The new version uses expm1(-log(q)/d)."),
    dict(func="ppf", arg=1 - 1e-8, c=0.3, d=2.0,
         title="ppf near q = 1 with moderate d: same cancellation, amplified by 1/c",
         why="Here q**(-1/d) - 1 = 5e-9 loses ~8 digits and the power -1/c = -3.3\n"
             "amplifies the relative error further."),
    dict(func="isf", arg=0.9, c=100.0, d=1e-3,
         title="isf with small d: overflow of expm1",
         why="isf uses expm1(-log1p(-q)/d) with -log1p(-q)/d = 2303; expm1\n"
             "overflows to inf and inf**(-1/c) = 0. The initial version left _isf\n"
             "unchanged. The new version switches to the log form\n"
             "x = exp(-log(expm1(z))/c) with log(expm1(z)) = z + log1mexp(-z)."),
    dict(func="ppf", arg=0.1, c=100.0, d=1e-3,
         title="ppf with small d: overflow handled by both proposals",
         why="q**(-1/d) = 10**1000 overflows; SciPy returns 0. Both the initial\n"
             "and the new version fall back to the log form here (control case)."),
]


def reference(func, arg, c, d):
    r = ref_x(arg, c, d) if func in ("pdf", "logpdf", "cdf", "logcdf", "sf", "logsf") \
        else ref_q(arg, c, d)
    return float(r[func])


def call(dist, func, arg, c, d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = float(getattr(dist, func)(arg, c, d))
    msgs = sorted({str(wi.message) for wi in w if issubclass(wi.category, RuntimeWarning)})
    return value, msgs


def relerr(value, ref):
    if not np.isfinite(value) or (value == 0 and ref != 0):
        return "total loss"
    e = abs(value - ref) / abs(ref)
    return f"{e:.1e}  ({e / EPS:.3g} ulp)"


def main():
    mp.mp.dps = 400
    for k, ex in enumerate(EXAMPLES, 1):
        func, arg, c, d = ex["func"], ex["arg"], ex["c"], ex["d"]
        ref = reference(func, arg, c, d)
        print(f"{k}. {ex['title']}")
        print(f"   burr.{func}({arg!r}, c={c:g}, d={d:g})")
        print("   " + ex["why"].replace("\n", "\n   "))
        print(f"   {'mpmath reference':<16} {ref!r}")
        for name, dist in IMPLS.items():
            value, msgs = call(dist, func, arg, c, d)
            note = f"   [{'; '.join(msgs)}]" if msgs else ""
            print(f"   {name:<16} {value!r:<24} rel. error {relerr(value, ref)}{note}")
        print()


if __name__ == "__main__":
    main()