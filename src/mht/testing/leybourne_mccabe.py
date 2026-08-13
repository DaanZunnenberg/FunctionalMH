"""
Leybourne-McCabe stationarity test.

Single canonical implementation -- imported by hypothesis.py.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.tsatools import lagmat


class Leybourne:
    """
    Leybourne-McCabe stationarity test.

    H0: series is stationary.

    References
    ----------
    Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a
    unit root. Journal of Business and Economic Statistics, 12: 157-166.

    Leybourne, S.J., & McCabe, B.P.M. (1999). Modified stationarity tests
    with data-dependent model-selection rules. Journal of Business and
    Economic Statistics, 17: 264-270.

    Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
    Testing the null hypothesis of stationarity against the alternative of
    a unit root. Journal of Econometrics, 54: 159-178.

    Schwert, G W. (1987). Effects of model specification on tests for unit
    roots in macroeconomic data. Journal of Monetary Economics, 20: 73-103.

    Notes
    -----
    Asymptotic critical values are the same as the asymptotic CVs for the
    KPSS stationarity test.  The p-values are generated through Monte Carlo
    simulation using 1,000,000 replications and 10,000 data points.
    """

    def __init__(self) -> None:
        self._critical_values: dict = {}

        # constant-only model
        critical_values_constant = (
            (99.9999, 0.00819), (99.999, 0.01050), (99.99, 0.01298),
            (99.9, 0.01701), (99.8, 0.01880), (99.7, 0.02005),
            (99.6, 0.02102), (99.5, 0.02186), (99.4, 0.02258),
            (99.3, 0.02321), (99.2, 0.02382), (99.1, 0.02437),
            (99.0, 0.02488), (97.5, 0.03045), (95.0, 0.03662),
            (92.5, 0.04162), (90.0, 0.04608), (87.5, 0.05024),
            (85.0, 0.05429), (82.5, 0.05827), (80.0, 0.06222),
            (77.5, 0.06621), (75.0, 0.07026), (72.5, 0.07439),
            (70.0, 0.07859), (67.5, 0.08295), (65.0, 0.08747),
            (62.5, 0.09214), (60.0, 0.09703), (57.5, 0.10212),
            (55.0, 0.10750), (52.5, 0.11315), (50.0, 0.11907),
            (47.5, 0.12535), (45.0, 0.13208), (42.5, 0.13919),
            (40.0, 0.14679), (37.5, 0.15503), (35.0, 0.16403),
            (32.5, 0.17380), (30.0, 0.18443), (27.5, 0.19638),
            (25.0, 0.20943), (22.5, 0.22440), (20.0, 0.24132),
            (17.5, 0.26123), (15.0, 0.28438), (12.5, 0.31242),
            (10.0, 0.34699), (7.5, 0.39354), (5.0, 0.45995),
            (2.5, 0.58098), (1.0, 0.74573), (0.9, 0.76453),
            (0.8, 0.78572), (0.7, 0.81005), (0.6, 0.83863),
            (0.5, 0.87385), (0.4, 0.91076), (0.3, 0.96501),
            (0.2, 1.03657), (0.1, 1.16658), (0.01, 1.60211),
            (0.001, 2.03312), (0.0001, 2.57878),
        )
        self._critical_values['c'] = np.asarray(critical_values_constant)

        # constant + trend model
        critical_values_trend = (
            (99.9999, 0.00759), (99.999, 0.00870), (99.99, 0.01023),
            (99.9, 0.01272), (99.8, 0.01378), (99.7, 0.01454),
            (99.6, 0.01509), (99.5, 0.01559), (99.4, 0.01598),
            (99.3, 0.01637), (99.2, 0.01673), (99.1, 0.01704),
            (99.0, 0.01731), (97.5, 0.02029), (95.0, 0.02342),
            (92.5, 0.02584), (90.0, 0.02791), (87.5, 0.02980),
            (85.0, 0.03158), (82.5, 0.03327), (80.0, 0.03492),
            (77.5, 0.03653), (75.0, 0.03813), (72.5, 0.03973),
            (70.0, 0.04135), (67.5, 0.04298), (65.0, 0.04464),
            (62.5, 0.04631), (60.0, 0.04805), (57.5, 0.04981),
            (55.0, 0.05163), (52.5, 0.05351), (50.0, 0.05546),
            (47.5, 0.05753), (45.0, 0.05970), (42.5, 0.06195),
            (40.0, 0.06434), (37.5, 0.06689), (35.0, 0.06962),
            (32.5, 0.07252), (30.0, 0.07564), (27.5, 0.07902),
            (25.0, 0.08273), (22.5, 0.08685), (20.0, 0.09150),
            (17.5, 0.09672), (15.0, 0.10285), (12.5, 0.11013),
            (10.0, 0.11917), (7.5, 0.13104), (5.0, 0.14797),
            (2.5, 0.17775), (1.0, 0.21801), (0.9, 0.22282),
            (0.8, 0.22799), (0.7, 0.23387), (0.6, 0.24109),
            (0.5, 0.24928), (0.4, 0.25888), (0.3, 0.27173),
            (0.2, 0.28939), (0.1, 0.32200), (0.01, 0.43218),
            (0.001, 0.54708), (0.0001, 0.69538),
        )
        self._critical_values['ct'] = np.asarray(critical_values_trend)

    def _interpolate_pvalue(self, stat: float, model: str = 'c') -> tuple:
        """
        Linear interpolation for Leybourne p-values and critical values.

        Parameters
        ----------
        stat : float
            The test statistic.
        model : {'c', 'ct'}
            Model used when computing the statistic. 'c' is default.

        Returns
        -------
        pvalue : float
        cvdict : dict
            Critical values at 1%, 5%, and 10% levels.
        """
        table = self._critical_values[model]
        y = table[:, 0]
        x = table[:, 1]
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, np.flip(y), np.flip(x))
        cvdict = {"1%": crit_value[0], "5%": crit_value[1], "10%": crit_value[2]}
        return pvalue, cvdict

    def _tsls_arima(
        self, x: npt.NDArray[np.floating], arlags: int, model: str
    ) -> tuple[npt.NDArray[np.floating], float, npt.NDArray[np.floating]]:
        """
        Two-stage least-squares approach for estimating ARIMA(p, 1, 1)
        parameters as an alternative to MLE in case of solver non-convergence.

        Parameters
        ----------
        x : array_like
            Data series.
        arlags : int
            AR(p) order.
        model : {'c', 'ct'}

        Returns
        -------
        arparams : ndarray
        theta : float
            MA(1) coefficient.
        residuals : ndarray
        """
        endog = np.diff(x, axis=0)
        exog = lagmat(endog, arlags, trim='both')
        if model == 'ct':
            exog = add_constant(exog)
        endog = endog[arlags:]
        if arlags > 0:
            resids = lagmat(OLS(endog, exog).fit().resid, 1, trim='forward')
        else:
            resids = lagmat(-endog, 1, trim='forward')
        exog = np.append(exog, -resids, axis=1)
        olsfit = OLS(endog, exog).fit()
        if model == 'ct':
            arparams = olsfit.params[1:(len(olsfit.params) - 1)]
        else:
            arparams = olsfit.params[0:(len(olsfit.params) - 1)]
        theta = olsfit.params[len(olsfit.params) - 1]
        return arparams, theta, olsfit.resid

    def _autolag(self, x: npt.NDArray[np.floating]) -> int:
        """
        Empirical AR lag detection.

        Set the number of AR lags equal to the first PACF falling within the
        95% confidence interval.  Maximum is min(10, len(x)//2); minimum is 0.
        """
        p = pacf(x, nlags=min(int(len(x) / 2), 10), method='ols')
        ci = 1.960 / np.sqrt(len(x))
        arlags = max(
            0,
            ([n - 1 for n, val in enumerate(p) if abs(val) < ci] + [len(p) - 1])[0],
        )
        return arlags

    def run(
        self,
        x: npt.ArrayLike,
        arlags: Optional[int] = 1,
        regression: str = 'c',
        method: str = 'mle',
        varest: str = 'var94',
    ) -> tuple[float, float, int, dict[str, float]]:
        """
        Run the Leybourne-McCabe stationarity test.

        Parameters
        ----------
        x : array_like
            1-D data series.
        arlags : int, optional
            Number of AR terms.  Pass ``None`` for automatic selection.
        regression : {'c', 'ct'}
            'c'  = constant only (default)
            'ct' = constant and trend
        method : {'mle', 'ols'}
            'mle' = conditional sum-of-squares MLE (default)
            'ols' = two-stage least squares
        varest : {'var94', 'var99'}
            'var94' = original 1994 LM paper (default)
            'var99' = follow-up 1999 paper

        Returns
        -------
        lmstat : float
        pvalue : float
        arlags : int
        cvdict : dict
            Critical values at 1%, 5%, 10%.
        """
        if regression not in ('c', 'ct'):
            raise ValueError("regression must be 'c' or 'ct', got %r" % regression)
        if method not in ('mle', 'ols'):
            raise ValueError("method must be 'mle' or 'ols', got %r" % method)
        if varest not in ('var94', 'var99'):
            raise ValueError("varest must be 'var94' or 'var99', got %r" % varest)

        x = np.asarray(x)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError('x must be 1-D or a 2-D column vector')
        x = np.reshape(x, (-1, 1))

        if arlags is None:
            arlags = self._autolag(x)
        elif not isinstance(arlags, int) or arlags < 0 or arlags > int(len(x) / 2):
            raise ValueError(
                'arlags must be an integer in [0, %d]' % int(len(x) / 2)
            )

        # Fit ARIMA(p, 1, 1)
        if method == 'mle':
            trend = 't' if regression == 'ct' else None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                arfit = ARIMA(x, order=(arlags, 1, 1), trend=trend).fit()
            resids = arfit.resid
            arcoeffs = arfit.arparams if arlags > 0 else []
            theta = arfit.maparams[0]
        else:
            arcoeffs, theta, resids = self._tsls_arima(x, arlags, model=regression)

        var99 = abs(theta * np.sum(resids ** 2) / len(resids))

        # Build filtered series z(t) = x(t) - sum_j arcoeffs[j] * x(t-j-1)
        n_z = len(x) - arlags
        z = x[arlags:arlags + n_z, 0].astype(float).copy()
        for j, coef in enumerate(arcoeffs):
            z -= coef * x[arlags - j - 1:arlags - j - 1 + n_z, 0]

        if regression == 'c':
            resids = z - z.mean()
        else:
            resids = OLS(z, add_constant(np.arange(1, len(z) + 1))).fit().resid

        var94 = np.sum(resids ** 2) / len(resids)
        eta = np.sum(resids.cumsum() ** 2) / (len(resids) ** 2)
        lmstat = eta / (var99 if varest == 'var99' else var94)

        pvalue, cvdict = self._interpolate_pvalue(lmstat, regression)
        return lmstat, pvalue, arlags, cvdict

    def __call__(
        self,
        x: npt.ArrayLike,
        arlags: Optional[int] = None,
        regression: str = 'c',
        method: str = 'mle',
        varest: str = 'var94',
    ) -> tuple[float, float, int, dict[str, float]]:
        """Alias for :meth:`run`. Note the default ``arlags`` differs (None => autolag)."""
        return self.run(x, arlags=arlags, regression=regression,
                        method=method, varest=varest)
