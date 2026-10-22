"""Burr Type III distribution with improved numerical accuracy in the tails.

Drop-in replacement for ``burr_gen`` in ``scipy/stats/_continuous_distns.py``.
The imports below are only needed to run this file on its own; inside SciPy the
module already provides all of them.

Changes compared with SciPy 1.18:

* ``_pdf``, ``_logpdf``: for ``x < 1`` use the algebraically equivalent form
  ``c*d*x**(c*d-1) * (1 + x**c)**(-d-1)``, so ``x**-c`` never overflows
  (SciPy 1.18 returns nan / -inf there). ``x == 0`` is covered by the same branch.
  In ``_pdf``, ``(1 + u)**(-d-1)`` is evaluated as ``exp(-(d+1)*log1p(u))``,
  which avoids a loss of about d ulp and overflow warnings for large d.
* ``_cdf``, ``_logcdf``, ``_sf``, ``_logsf``: everything is derived from an
  overflow-free ``logcdf``; ``sf`` and ``logsf`` no longer cancel in ``1 - cdf``.
  ``_cdf`` uses the power form for ``x <= 1`` and ``exp(logcdf)`` for ``x > 1``,
  whichever loses less; ``_logsf`` reuses ``_cdf`` where ``cdf < 1/2``.
* ``_ppf``, ``_isf``: share one inversion of ``F = (1 + x**-c)**-d`` with three
  forms: ``(F**(-1/d) - 1)**(-1/c)`` where it neither cancels nor overflows
  (most accurate), ``expm1(z)**(-1/c)`` with ``z = -log(F)/d`` where it would
  cancel (F near 1), and ``exp(-log(expm1(z))/c)`` where it would overflow.
"""
import numpy as np
import scipy.special as sc
import scipy.special._ufuncs as scu
import scipy._external.array_api_extra as xpx
from scipy.stats._distn_infrastructure import rv_continuous, _ShapeInfo


def _log_expm1(y):
    # log(expm1(y)) for y > 0, finite where expm1(y) overflows
    return y + scu._log1mexp(-y)


def _burr_inv(z, c):
    # Solve log1p(x**-c) == z for x, i.e. x = expm1(z)**(-1/c), for z > 0.
    # ppf and isf call this with z = -log(F)/d. The pow form is the more
    # accurate one, but expm1(z) overflows for z > ~709; there
    # log(expm1(z)) = z + log1mexp(-z) is used instead. The threshold is not
    # critical: for z > ~37 both branches agree to a few ulp.
    return xpx.apply_where(
        z < 700, (z, c),
        lambda z, c: sc.expm1(z) ** (-1.0 / c),
        lambda z, c: np.exp(-_log_expm1(z) / c))


def _burr_quantile(p, logp, c, d, p_exact):
    # x with cdf(x) = p, i.e. x = (p**(-1/d) - 1)**(-1/c), given log(p).
    # Where p**(-1/d) is in [2, ~1e304], subtracting 1 loses at most one bit,
    # and pow is the most accurate form: it works with log(p)/d internally in
    # extra precision, while _burr_inv rounds z = -log(p)/d first, which costs
    # about z ulp. Elsewhere (cancellation or overflow) _burr_inv is used.
    # p_exact marks where p is exactly the caller's argument and not rounded.
    z = -logp / d
    return xpx.apply_where(
        p_exact & (z > np.log(2)) & (z < 700), (p, z, c, d),
        lambda p, z, c, d: (p ** (-1.0 / d) - 1) ** (-1.0 / c),
        lambda p, z, c, d: _burr_inv(z, c))


class burr_gen(rv_continuous):
    r"""A Burr (Type III) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr12 : Burr Type XII distribution
    mielke : Mielke Beta-Kappa / Dagum distribution

    Notes
    -----
    The probability density function for `burr` is:

    .. math::

        f(x; c, d) = c d \frac{x^{-c - 1}}
                              {{(1 + x^{-c})}^{d + 1}}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr` takes ``c`` and ``d`` as shape parameters for :math:`c` and
    :math:`d`.

    This is the PDF corresponding to the third CDF given in Burr's list;
    specifically, it is equation (11) in Burr's paper [1]_. The distribution
    is also commonly referred to as the Dagum distribution [2]_. If the
    parameter :math:`c < 1` then the mean of the distribution does not
    exist and if :math:`c < 2` the variance does not exist [2]_.
    The PDF is finite at the left endpoint :math:`x = 0` if :math:`c * d >= 1`.

    %(after_notes)s

    References
    ----------
    .. [1] Burr, I. W. "Cumulative frequency functions", Annals of
       Mathematical Statistics, 13(2), pp 215-232 (1942).
    .. [2] "Dagum distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Dagum_distribution
    .. [3] Kleiber, Christian. "A guide to the Dagum distributions."
       Modeling Income Distributions and Lorenz Curves  pp 97-117 (2008).

    %(example)s

    """
    # Do not set _support_mask to rv_continuous._open_support_mask
    # Whether the left-hand endpoint is suitable for pdf evaluation is dependent
    # on the values of c and d: if c*d >= 1, the pdf is finite, otherwise infinite.

    def _shape_info(self):
        ic = _ShapeInfo("c", False, (0, np.inf), (False, False))
        id = _ShapeInfo("d", False, (0, np.inf), (False, False))
        return [ic, id]

    def _pdf(self, x, c, d):
        # burr.pdf(x, c, d) = c * d * x**(-c-1) * (1+x**(-c))**(-d-1)
        #                   = c * d * x**(c*d-1) * (1+x**c)**(-d-1)
        # The second form is used for x < 1, where x**-c can overflow; it also
        # gives the limit at x = 0.
        # (1 + u)**(-d-1) is evaluated as exp(-(d+1)*log1p(u)): the power form
        # multiplies the rounding error of 1 + u by d + 1, while the exp form
        # loses at most about min(d, 745) ulp and cannot overflow.
        def right(x_, c_, d_):
            # x**(-c-1) is subnormal, with few significant bits, once
            # (c+1)*log(x) > 708.4; c*d can lift the product back into the
            # normal range, so there the whole pdf is evaluated in log space
            return xpx.apply_where(
                (c_ + 1) * np.log(x_) < 708, (x_, c_, d_),
                lambda x_, c_, d_: (c_ * d_ * x_**(-c_ - 1.0)
                                    * np.exp(-(d_ + 1) * sc.log1p(x_**(-c_)))),
                lambda x_, c_, d_: np.exp(np.log(c_) + np.log(d_)
                                          - (c_ + 1) * np.log(x_)
                                          - (d_ + 1) * sc.log1p(x_**(-c_))))

        output = xpx.apply_where(
            x < 1, (x, c, d),
            lambda x_, c_, d_: (c_ * d_ * x_**(c_*d_ - 1.0)
                                * np.exp(-(d_ + 1) * sc.log1p(x_**c_))),
            right)
        return output[()] if output.ndim == 0 else output

    def _logpdf(self, x, c, d):
        # Same split as _pdf. For x < 1 this also avoids the cancellation
        # between -(c+1)*log(x) and -(d+1)*log1p(x**-c) when c*d is close to 1.
        output = xpx.apply_where(
            x < 1, (x, c, d),
            lambda x_, c_, d_: (np.log(c_) + np.log(d_)
                                + sc.xlogy(c_*d_ - 1, x_)
                                - (d_+1) * sc.log1p(x_**c_)),
            lambda x_, c_, d_: (np.log(c_) + np.log(d_)
                                - (c_ + 1) * np.log(x_)
                                - (d_+1) * sc.log1p(x_**(-c_))))
        return output[()] if output.ndim == 0 else output

    def _cdf(self, x, c, d):
        def power(x, c, d):
            # x**-c overflows to inf for small x, where the log form holds
            with np.errstate(over='ignore'):
                cdf = (1 + x**(-c))**(-d)
            return np.where(cdf == 0, np.exp(self._logcdf(x, c, d)), cdf)

        # The power form loses about d ulp (rounding of 1 + x**-c is raised to
        # the power -d); exp(logcdf) loses about |logcdf| ulp, which is at most
        # d*log(2) for x >= 1 but larger than d for x < 1. At x = 1 nothing is
        # rounded and the power form gives 2**-d exactly.
        return xpx.apply_where(
            x <= 1, (x, c, d), power,
            lambda x, c, d: np.exp(self._logcdf(x, c, d)))

    def _logcdf(self, x, c, d):
        # log1p(x**c) - c*log(x) is log1p(x**-c) without overflow for x < 1
        return -d * xpx.apply_where(
            x < 1, (x, c),
            lambda x, c: sc.log1p(x**c) - c * np.log(x),
            lambda x, c: sc.log1p(x**-c))

    def _sf(self, x, c, d):
        return -sc.expm1(self._logcdf(x, c, d))

    def _logsf(self, x, c, d):
        # log1mexp(logcdf), except that where cdf < 1/2 the cdf itself is used:
        # in the left tail _cdf uses the power form, which is more accurate
        # than exp(logcdf) (|logcdf| ulp) and keeps logsf = log1p(-cdf) exact
        # to a few ulp. Near cdf = 1, log(-expm1(logcdf)) avoids cancellation.
        logcdf = self._logcdf(x, c, d)
        return xpx.apply_where(
            logcdf < -np.log(2), (x, c, d, logcdf),
            lambda x, c, d, _: sc.log1p(-self._cdf(x, c, d)),
            lambda x, c, d, logcdf: scu._log1mexp(logcdf))

    def _ppf(self, q, c, d):
        # cdf = (1 + x**-c)**-d = q
        return _burr_quantile(q, np.log(q), c, d, True)

    def _isf(self, q, c, d):
        # cdf = 1 - q; 1 - q is exact for q >= 1/2 (Sterbenz)
        return _burr_quantile(1 - q, sc.log1p(-q), c, d, q >= 0.5)

    def _stats(self, c, d):
        nc = np.arange(1, 5).reshape(4,1) / c
        # ek is the kth raw moment, e1 is the mean e2-e1**2 variance etc.
        e1, e2, e3, e4 = sc.beta(d + nc, 1. - nc) * d
        mu = np.where(c > 1.0, e1, np.nan)
        mu2_if_c = e2 - mu**2
        mu2 = np.where(c > 2.0, mu2_if_c, np.nan)
        g1 = xpx.apply_where(
            c > 3.0, (e1, e2, e3, mu2_if_c),
            lambda e1, e2, e3, mu2_if_c: ((e3 - 3*e2*e1 + 2*e1**3)
                                           / np.sqrt((mu2_if_c)**3)),
            fill_value=np.nan)
        g2 = xpx.apply_where(
            c > 4.0, (e1, e2, e3, e4, mu2_if_c),
            lambda e1, e2, e3, e4, mu2_if_c: (
                ((e4 - 4*e3*e1 + 6*e2*e1**2 - 3*e1**4) / mu2_if_c**2) - 3),
            fill_value=np.nan)
        if np.ndim(c) == 0:
            return mu.item(), mu2.item(), g1.item(), g2.item()
        return mu, mu2, g1, g2

    def _munp(self, n, c, d):
        def __munp(n, c, d):
            nc = 1. * n / c
            return d * sc.beta(1.0 - nc, d + nc)
        n, c, d = np.asarray(n), np.asarray(c), np.asarray(d)
        return xpx.apply_where((c > n) & (n == n) & (d == d),
                               (n, c, d), __munp, fill_value=np.nan)


burr = burr_gen(a=0.0, name='burr')
