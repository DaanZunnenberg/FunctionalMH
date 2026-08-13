"""
Kernel-based stationarity test for multivariate diffusion processes.

Classes
-------
Kernel
    Box-kernel for state-domain smoothing.
Test
    Original (deprecated) test implementation.
KernelTest
    Current test implementation (was TestV2).
Simulator
    Monte Carlo simulation driver (was simulate).
ResultSummary
    Post-simulation rejection-rate summary (was repl).
"""
from __future__ import annotations

import itertools
from itertools import accumulate
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.neighbors import KernelDensity
from deprecated import deprecated
from tqdm import tqdm

from mht.models import processes as _models

__VERSION__: str = '1.0.5'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def running_maximum(X: npt.ArrayLike) -> list:
    """Cumulative running maximum of ``|X|``."""
    return list(accumulate(np.abs(X), max))


def simple_sequence(X: npt.NDArray[np.floating], pct: float = 0.3) -> float:
    """``pct`` fraction of the range of ``X`` (flattened)."""
    return pct * (np.max(X.flatten()) - np.min(X.flatten()))


def isiterable(obj: Any, type: type = list) -> bool:
    """Return ``True`` if ``obj`` is an instance of ``type`` (default: list)."""
    return isinstance(obj, type)


def gumbel_running_max_bound(n: int, alpha: float) -> list:
    """
    Asymptotic (Gumbel-type) critical bound for the running maximum of a
    standardised Gaussian process of length ``n``, at confidence ``alpha``.

    Shared by :class:`ResultSummary`, :class:`Simulator`, and
    :class:`TestPlotter`, all of which compare an empirical running maximum
    against this theoretical threshold (see Pickands 1969; Berman 1964).
    """
    x = -np.log(np.log(1 / alpha))
    an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
    bn = [
        np.sqrt(2 * np.log(z))
        - ((np.log(np.log(z)) + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(z))))
        for z in range(1, n + 1)
    ]
    return [np.nan] + [(x / an[i]) + bn[i] for i in range(1, n)]


# ---------------------------------------------------------------------------
# Shared bookkeeping mixin (Test and KernelTest share this machinery)
# ---------------------------------------------------------------------------

class _EstimatorBookkeepingMixin:
    """
    Shared introspection/repr machinery and fixed algebraic constants used
    by both the deprecated :class:`Test` and current :class:`KernelTest`.

    ``duplication`` and ``VECH_VEC`` are fixed 3x4 / 4x4 matrices that
    implement the vech/vec duplication mapping for symmetric 2x2 matrices
    (see Magnus & Neudecker, duplication matrix theory) used throughout the
    covariance transformations of the diffusion-matrix estimators.
    """

    @staticmethod
    def _form(shape: tuple[int, int], sep: str = '\n', fmt: str = r'{}') -> str:
        f: str = ''
        for _ in range(shape[1]):
            f += shape[0] * fmt + sep
        return f

    def __repr__(self) -> str:
        _repr: str = self._form(shape=(2, len(self.__dict__)))
        return _repr.format(
            *list(itertools.chain(*list(zip(self.__dict__.keys(), self.__dict__.values()))))
        )

    def info(self) -> None:
        """Print a summary of every attribute currently set on the instance."""
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                print(key)
                for k, v in val.items():
                    print('{:>10} :: {}'.format(k, v))
            elif isinstance(val, pd.DataFrame):
                print(key, '::', val.__dict__.get(
                    'name', 'Nameless dataframe at 0x{:x}'.format(id(val))
                ))
            else:
                print('{:>10} :: {}'.format(key, val))

    def rename_attribute(self, old_name: str) -> None:
        """Remove attribute ``old_name`` from the instance dict."""
        self.__dict__.pop(old_name)

    def class_operators(self, remove: bool = False, **kwargs: Any) -> '_EstimatorBookkeepingMixin':
        """Bulk-set (or, if ``remove``, bulk-delete) attributes from kwargs."""
        for name, value in kwargs.items():
            if not remove:
                self.__setattr__(name, value)
            else:
                if hasattr(self, name):
                    self.__dict__.pop(name)
        return self

    @property
    def duplication(self) -> np.ndarray:
        """3x4 duplication matrix mapping vec(2x2 symmetric) to vech."""
        return np.array([
            [.5, .0, .0, .0],
            [.0, .5, .5, .0],
            [.0, .0, .0, .5],
        ])

    @property
    def VECH_VEC(self) -> np.ndarray:
        """4x4 selection matrix used in the vech/vec transformation."""
        return np.array([
            [1., .0, .0, .0],
            [.0, 1., .0, .0],
            [.0, .0, .0, 1.],
        ])


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class Kernel:
    """Indicator (box) kernel for two-dimensional state-domain smoothing."""

    def __init__(self, *, kernel_params: dict) -> None:
        self.kernel_params = kernel_params

    def BaseKernel(self) -> Callable[[Any, Any], float]:
        """Return the bivariate box kernel ``1{|x-y| <= bw} / bw^2`` (per axis)."""
        bandwidth = self.kernel_params.get('bandwidth', NameError)
        select_kernel: Callable[[Any, Any], bool] = lambda x, y: (np.abs(x - y) <= bandwidth)
        base_kernel: Callable[[Any, Any], float] = (
            lambda x, y:
            1 * select_kernel(x.item(0), y.item(0))
            * select_kernel(x.item(1), y.item(1))
            / (bandwidth ** 2)
        )
        return base_kernel


# ---------------------------------------------------------------------------
# Deprecated legacy Test class
# ---------------------------------------------------------------------------

@deprecated('Class Test is deprecated; use KernelTest instead.')
class Test(_EstimatorBookkeepingMixin):
    """Original test implementation -- kept for backward compatibility."""

    def __init__(
        self,
        data: pd.DataFrame,
        kernel_params: dict,
        time_params: dict,
        disable: bool = False,
        reachable: bool = False,
        user: str = 'root',
        show_object: bool = True,
    ) -> None:
        self.data = data
        self.kernel_params = kernel_params
        self.time_params = time_params
        self.disable = disable
        self.reachable = reachable
        self.use = user
        self.kernel_estimates: dict = {}
        self.time_estimates: dict = {}
        if show_object:
            self.info()

    @staticmethod
    def integrate_kernel(bandwidth: float, p: int = 2) -> float:
        """Analytic integral of the box kernel (indicator kernel only)."""
        return (2 * bandwidth) / (bandwidth ** (2 * p))

    def state_domain_estimator(self) -> None:
        bandwidth, n_obs, T, _kernel = self.kernel_params.values()
        X = self.data
        DELTA = self.data.diff().fillna(0)
        dt = T / n_obs
        estimate: list = []

        for x in tqdm(X.values, disable=self.disable):
            sliced = DELTA.iloc[X[(X - x).abs() <= bandwidth].dropna().index].to_numpy()
            norm: float = len(sliced)
            sub_estimate = sliced.T @ sliced if norm else np.zeros((2, 2))
            estimate.append(sub_estimate / (norm * dt))

        variance = [
            2 * self.duplication @ np.kron(s, s) @ self.duplication.T
            for s in estimate
        ]
        self.kernel_estimates = {**self.kernel_estimates,
                                  **{'estimate': estimate, 'variance': variance}}

    def time_domain_estimator(self) -> None:
        bandwidth, n_obs, T = self.time_params.values()
        X = self.data
        DELTA = self.data.diff().fillna(0).to_numpy()
        dt = T / n_obs
        est = [np.outer(row, row) for row in DELTA]

        runup: int = np.ceil(bandwidth / dt).astype(int)
        estimate: list = [np.full((2, 2), np.nan)] * runup

        for idx in tqdm(range(runup, len(X)), disable=self.disable):
            estimate.append(np.sum(est[idx - runup:idx], axis=0) / bandwidth)

        variance = [
            2 * self.duplication @ np.kron(s, s) @ self.duplication.T
            for s in estimate
        ]
        self.time_estimates = {**self.time_estimates,
                                **{'estimate': estimate, 'variance': variance}}

    def gauss(self) -> None:
        """Compute the standardised Gaussian process from the two estimators."""
        bandwidth_time, _, __ = self.time_params.values()
        bandwidth_state, n_obs, _, __ = self.kernel_params.values()

        L = self.VECH_VEC
        estimate_time, _ = self.time_estimates.values()
        estimate_kernel, _ = self.kernel_estimates.values()
        state_kernel_integral = self.integrate_kernel(bandwidth=bandwidth_state, p=2)

        diff_vec, var, std_inv = [], [], []
        for time_pt, state_pt in zip(estimate_time, estimate_kernel):
            diff_vec.append(L @ (time_pt - state_pt).reshape(4, 1))
            v = 2 * self.duplication @ np.kron(state_pt, state_pt) @ self.duplication.T
            v *= (1 + state_kernel_integral) / (n_obs * (bandwidth_state ** 2))
            var.append(v)
            try:
                std_inv.append(np.linalg.inv(sqrtm(v)))
            except Exception:
                std_inv.append(np.full((3, 3), np.nan))

        gaussian = []
        for i, dv in enumerate(diff_vec):
            try:
                vi = std_inv[i]
            except Exception:
                vi = np.full((3, 3), np.nan)
            gaussian.append(vi @ dv)
        self.gaussian = gaussian

    def transform_1D_gauss(self, alpha: float = 0.95) -> tuple:
        """Map the multivariate Gaussian process to a scalar statistic + Gumbel bound."""
        x = np.log(1 / np.log(1 / alpha))
        if not hasattr(self, 'gaussian'):
            self.gauss()
        n = len(self.gaussian)
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [
            np.sqrt(2 * np.log(z))
            - (np.log(np.pi * np.log(z)) / (2 * np.sqrt(2 * np.log(z))))
            for z in range(1, n + 1)
        ]
        bound = [np.nan] + [(x / an[i]) + bn[i] for i in range(1, n)]
        final = [
            0 if np.isnan(g).any() else np.sum(g) / np.sqrt(3)
            for g in self.gaussian
        ]
        return bound, final


# ---------------------------------------------------------------------------
# KernelTest  (was TestV2)
# ---------------------------------------------------------------------------

class KernelTest(_EstimatorBookkeepingMixin):
    """
    Kernel-based test for time-homogeneity of the diffusion matrix.

    The test compares a state-domain smoother with a time-domain smoother
    of the integrated diffusion matrix.  Under H0 (time-homogeneous
    diffusion) the standardised difference converges to a Gaussian process
    whose running maximum has a known Gumbel-type limit distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Bivariate path with columns ``['process 1', 'process 2']``.
    kernel_params : dict
        Keys: bandwidth, n, T, kernel.
    time_params : dict
        Keys: bandwidth, n, T.
    disable : bool
        Suppress tqdm progress bars.
    show_object : bool
        Print a summary on construction.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        kernel_params: dict,
        time_params: dict,
        disable: bool = False,
        reachable: bool = False,
        user: str = 'root',
        show_object: bool = True,
    ) -> None:
        self.data = data
        self.kernel_params = kernel_params
        self.time_params = time_params
        self.disable = disable
        self.reachable = reachable
        self.use = user
        self.kernel_estimates: dict = {}
        self.time_estimates: dict = {}
        if show_object:
            self.info()

    @property
    def projection_mat(self) -> np.ndarray:
        """3x4 projection matrix used to build the univariate limiting variance."""
        return np.array([
            [1., .0, .0, .0],
            [.0, .5, .5, .0],
            [.0, .0, .0, 1.],
        ])

    def emp_dist(self, dist: bool) -> np.ndarray:
        """
        Empirical (joint) density estimate used in the covariance of the
        weak limit.  Returns ones when ``dist=False``.
        """
        n_obs = self.kernel_params['n']
        if not dist:
            return np.ones(n_obs)
        bandwidth = self.kernel_params['bandwidth']
        x = self.data[['process 1']]
        y = self.data[['process 2']]
        kde1 = KernelDensity(kernel='gaussian', bandwidth=2 * bandwidth).fit(x)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=2 * bandwidth).fit(y)
        return np.exp(kde1.score_samples(x) + kde2.score_samples(y))

    @staticmethod
    def tau_scalar(tau: float) -> float:
        """Variance-inflation scalar for the EWMA time-domain smoother."""
        return tau * (1 + np.exp(tau)) / (np.exp(tau) - 1)

    def univ_var(self, A: Union[np.ndarray, list]) -> Union[np.ndarray, list]:
        """Project a 2x2-covariance (or list thereof) into the 3x3 vech-space limiting variance."""
        P_P = self.projection_mat
        if isiterable(A):
            return [P_P @ np.kron(cov, cov) @ P_P.T for cov in A]
        return P_P @ np.kron(A, A) @ P_P.T

    def time_domain_smoother(
        self,
        lamb: float = 0.94,
        allow_true_var: bool = False,
        true_var: Optional[list] = None,
    ) -> None:
        """
        Exponentially-weighted time-domain estimator of the diffusion matrix.

        Parameters
        ----------
        lamb : float
            Decay factor for the EWMA smoother (0 < lamb < 1).
        allow_true_var : bool
            Use an externally supplied true variance instead of the estimate.
        true_var : list, optional
            True variance matrices (same length as the data).
        """
        bandwidth, n_obs, T = self.time_params.values()
        X = self.data
        DELTA = self.data.diff().fillna(0).to_numpy()
        dt = T / n_obs
        est = [np.outer(row, row) for row in DELTA]

        runup: int = np.ceil(bandwidth / dt).astype(int)
        tau = runup * (1 - lamb)

        nan_mat = np.full((2, 2), np.nan)
        estimate: list = [nan_mat.copy()] * runup
        variance: list = [nan_mat.copy()] * runup
        true_pvar: list = [nan_mat.copy()] * runup
        fact_pvar: list = [nan_mat.copy()] * runup

        if allow_true_var:
            variance = list(true_var)

        for idx in tqdm(range(runup, len(X)), disable=self.disable):
            _est = np.zeros_like(est[idx])
            for sidx in range(runup):
                _est += (lamb ** sidx) * est[idx - sidx]
            _est *= ((1 - lamb) / (1 - (lamb ** runup))) / dt
            estimate.append(_est)
            fact_pvar.append(self.tau_scalar(tau))
            if allow_true_var:
                variance.append(self.tau_scalar(tau) * self.univ_var(true_var[idx]))
                true_pvar.append(self.univ_var(true_var[idx]))
            else:
                variance.append(self.tau_scalar(tau) * self.univ_var(_est))
                true_pvar.append(self.univ_var(_est))

        self.time_factpvar = fact_pvar
        self.true_est_var = true_pvar
        self.time_estimates = {**self.time_estimates,
                                **{'estimate': estimate, 'variance': variance}}

    def state_domain_smoother(self, dist: Union[bool, list, None] = None) -> None:
        """
        State-domain (kernel) smoother of the diffusion matrix.

        Must be called after :meth:`time_domain_smoother`.

        Parameters
        ----------
        dist : bool or list or None
            Joint density values.  ``True`` uses KDE; ``False`` uses ones;
            a list is used directly.
        """
        if not hasattr(self, 'true_est_var'):
            raise RuntimeError(
                'Call time_domain_smoother() before state_domain_smoother().'
            )
        if not isinstance(dist, list):
            dist = self.emp_dist(dist)
        self.kernel_dist = dist

        bandwidth, n_obs, T, _kernel = self.kernel_params.values()
        X = self.data
        DELTA = self.data.diff().fillna(0)
        dt = T / n_obs

        estimate: list = []
        variance: list = []
        fact_pvar: list = []

        for idx, x in enumerate(tqdm(X.values, disable=self.disable)):
            sliced = DELTA.iloc[X[(X - x).abs() <= bandwidth].dropna().index].to_numpy()
            norm: float = len(sliced)
            sub_estimate = sliced.T @ sliced if norm else np.zeros((2, 2))

            fact_pvar.append(bandwidth ** 2 / dist[idx])
            estimate.append(sub_estimate / (norm * dt))
            variance.append(self.true_est_var[idx] / dist[idx])

        self.state_factpvar = fact_pvar
        self.kernel_estimates = {**self.kernel_estimates,
                                  **{'estimate': estimate, 'variance': variance}}

    @staticmethod
    def integrate_kernel(bandwidth: float, p: int = 2) -> float:
        """Analytic integral of the box kernel (indicator kernel only)."""
        return (2 * bandwidth) / (bandwidth ** (2 * p))

    def gauss(self) -> None:
        """
        Compute the standardised Gaussian process from the two smoothers.

        Stores result in ``self.gaussian``.
        """
        b, n_obs, T, _kernel = self.kernel_params.values()
        bandwidth_state = self.kernel_params['bandwidth']
        dist = self.kernel_dist

        state_norm = self.state_factpvar
        time_norm = self.time_factpvar
        true_pvar = self.true_est_var

        L = self.VECH_VEC
        time_est = self.time_estimates['estimate']
        state_est = self.kernel_estimates['estimate']

        diff_vec, var, std_inv = [], [], []
        for time_pt, state_pt, v, sn, tn in zip(
            time_est, state_est, true_pvar, state_norm, time_norm
        ):
            diff_vec.append(L @ (time_pt - state_pt).reshape(4, 1))
            _v = (2 * sn + tn) * v
            var.append(_v)
            try:
                std_inv.append(np.linalg.inv(sqrtm(_v)))
            except Exception:
                std_inv.append(np.full((3, 3), np.nan))

        gaussian = []
        for i, dv in enumerate(diff_vec):
            try:
                vi = std_inv[i]
            except Exception:
                vi = np.full((3, 3), np.nan)
            gaussian.append(np.sqrt(T * b ** 2) * vi @ dv)
        self.gaussian = gaussian

    def transform_1D_gauss(self, alpha: float = 0.95) -> tuple:
        """
        Map the multivariate Gaussian process to a scalar running-maximum
        statistic and compute the Gumbel-type critical bound.

        Returns
        -------
        bound : list
        scalar_gauss : list
        """
        x = np.log(1 / np.log(1 / alpha))
        if not hasattr(self, 'gaussian'):
            self.gauss()
            print('Self@gauss() called automatically.')
        n = len(self.gaussian)
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [
            np.sqrt(2 * np.log(z))
            - (np.log(np.pi * np.log(z)) / (2 * np.sqrt(2 * np.log(z))))
            for z in range(1, n + 1)
        ]
        bound = [np.nan] + [(x / an[i]) + bn[i] for i in range(1, n)]
        scalar_gauss = [
            0 if np.isnan(g).any() else float(np.sum(g) / np.sqrt(3))
            for g in self.gaussian
        ]
        return bound, scalar_gauss


# ---------------------------------------------------------------------------
# ResultSummary  (was repl)
# ---------------------------------------------------------------------------

class ResultSummary:
    """
    Summarise rejection rates from pre-computed simulation results stored in
    a DataFrame (one column per run).

    Parameters
    ----------
    gauss : pd.DataFrame
        Columns ``run_0``, ``run_1``, ...  each containing the scalar
        Gaussian process for that run.
    name : str
        Label printed in the summary header.
    """

    def __init__(self, gauss: pd.DataFrame, name: str) -> None:
        self.gauss = gauss
        self.name = name
        self.number_of_runs = len(gauss.columns)
        self.rejections: dict = {}

    def _bound(self, alpha: float) -> list:
        """Gumbel-type critical bound for the running maximum at level ``alpha``."""
        return gumbel_running_max_bound(len(self.gauss['run_0']), alpha)

    def summary(self, alphas: Union[list, tuple] = (0.90, 0.95, 0.99)) -> None:
        print(f' {self.name} '.center(70, '-'))
        for alpha in alphas:
            bound = self._bound(alpha)
            rate = np.sum([
                running_maximum(self.gauss[f'run_{run}'])[-1] > bound[-1]
                for run in range(self.number_of_runs)
            ]) / self.number_of_runs
            self.rejections.setdefault('running maximum', {})[1 - alpha] = rate
            print('Number of runs: {}, rejection rate: {}% (α = {}%)'.format(
                self.number_of_runs,
                np.round(100 * rate, 2),
                np.round(100 * alpha, 0),
            ))
        print()


# ---------------------------------------------------------------------------
# Simulator  (was simulate)
# ---------------------------------------------------------------------------

class Simulator(_EstimatorBookkeepingMixin):
    """
    Monte Carlo simulation driver.

    Generates repeated realisations from a stochastic process model, applies
    the KernelTest to each realisation, and stores the scalar Gaussian process
    and the path data for post-processing.

    Parameters
    ----------
    number_of_runs : int
    config : dict
        Keyword arguments forwarded to ``ProcessGenerator``.
    est_config : dict or None
        Override the automatically generated estimator config.
    ProcessGenerator : class
        A process class with ``.simulate()`` and ``.config()`` methods
        (default: :class:`~mht.models.processes.BivariateOUProcess`).
    """

    def __init__(
        self,
        number_of_runs: int,
        config: Optional[dict] = None,
        est_config: Optional[dict] = None,
        ProcessGenerator: Optional[type] = None,
    ) -> None:
        self.number_of_runs = number_of_runs
        self.config = config or {}
        if est_config is not None:
            est_config.pop('data', None)
        self.est_config = est_config
        self.ProcessGenerator = (
            ProcessGenerator or _models.BivariateOUProcess
        )
        self.results: dict = {}
        self.info()

    def _generate_config(
        self, data: pd.DataFrame, T: float, n_obs: int,
        bandwidth_sequence: Optional[list] = None
    ) -> dict:
        """
        Build the estimator config dict.

        Uses Fan-Fan-Lv (arXiv:math/0701107) bandwidth correction if no
        custom ``est_config`` was supplied.
        """
        if bandwidth_sequence is None:
            bandwidth_sequence = [9, 6, 2]

        if not self.est_config:
            if T >= 200:
                bw_idx = 0
                time_bw_factor = 200
            elif T >= 100:
                bw_idx = 1
                time_bw_factor = 100
            else:
                bw_idx = 2
                time_bw_factor = 100

            return {
                'data': data,
                'kernel_params': {
                    'bandwidth': (
                        np.sqrt(3)
                        * bandwidth_sequence[bw_idx]
                        / ((n_obs ** (1 / 6)) * np.log(n_obs))
                    ),
                    'n': n_obs,
                    'T': T,
                    'kernel': Kernel.BaseKernel,
                },
                'time_params': {
                    'bandwidth': time_bw_factor * T / n_obs,
                    'n': n_obs,
                    'T': T,
                },
            }

        return {'data': data, **self.est_config}

    def run(
        self,
        seed: Optional[list] = None,
        state_kwargs: Optional[dict] = None,
        time_kwargs: Optional[dict] = None,
        **test_kwargs: Any,
    ) -> None:
        """
        Execute the simulation loop.

        Parameters
        ----------
        seed : list, optional
            Per-run random seeds.
        state_kwargs : dict
            Extra kwargs forwarded to ``state_domain_smoother``.
        time_kwargs : dict
            Extra kwargs forwarded to ``time_domain_smoother``.
        **test_kwargs
            Extra kwargs forwarded to the :class:`KernelTest` constructor.
        """
        if seed is None:
            seed = []
        if state_kwargs is None:
            state_kwargs = {}
        if time_kwargs is None:
            time_kwargs = {}

        if seed:
            print('Simulation started on custom seed.')

        process = self.ProcessGenerator(**self.config)
        for run in tqdm(range(self.number_of_runs)):
            try:
                np.random.seed(seed[run])
            except (IndexError, TypeError):
                print('No seed set', end='\r')

            process.simulate()
            X, T, n_obs = process.config()
            test = KernelTest(**{**self._generate_config(X, T, n_obs), **test_kwargs})

            test.time_domain_smoother(**time_kwargs)
            test.state_domain_smoother(**state_kwargs)
            test.gauss()

            bound, scalar_gauss = test.transform_1D_gauss()

            self.results[f'run_{run}'] = {
                'gauss': scalar_gauss,
                'process': X,
            }
        self.results['bound'] = bound

    def _bound(self, alpha: float) -> list:
        """Gumbel-type critical bound for the running maximum at level ``alpha``."""
        return gumbel_running_max_bound(len(self.results['run_0']['gauss']), alpha)

    def summary(self, alphas: Union[list, tuple] = (0.95,)) -> None:
        """Print rejection rates of the running-maximum test at each alpha."""
        for alpha in alphas:
            bound = self._bound(alpha)
            rate = np.sum([
                1 * (running_maximum(self.results[f'run_{run}']['gauss'])[-1] > bound[-1])
                for run in range(self.number_of_runs)
            ]) / self.number_of_runs
            print('Number of runs: {}, rejection rate: {}% (α = {}%)'.format(
                self.number_of_runs,
                np.round(100 * rate, 2),
                np.round(100 * alpha, 0),
            ))

    def plot_results(self, **kwargs: Any) -> None:
        """Plot per-run running maxima with theoretical Gumbel bounds, plus a histogram."""
        bound1 = self._bound(0.99)
        bound5 = self._bound(0.95)
        bound10 = self._bound(0.90)
        bound50 = self._bound(0.50)

        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        for run in range(self.number_of_runs):
            rr = self.results[f'run_{run}']
            label = 'Empirical 95% running maximum' if run == 0 else None
            axs[0].plot(running_maximum(rr['gauss']), c='C0', lw=1, alpha=0.5, label=label)

        axs[0].plot(bound1, color='grey', ls='-', lw=1, label='Theoretical running maximum (1%)')
        axs[0].plot(bound5, color='grey', ls='dashed', lw=1, label='Theoretical running maximum (5%)')
        axs[0].plot(bound10, color='grey', ls='dotted', lw=1, label='Theoretical running maximum (10%)')
        axs[0].plot(bound50, color='red', ls='dashed', lw=1, label='Theoretical running maximum (50%)')
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        axs[1].hist(
            [running_maximum(self.results[f'run_{run}']['gauss'])[-1]
             for run in range(self.number_of_runs)],
            **kwargs,
        )
        plt.show()


# ---------------------------------------------------------------------------
# TestPlotter  (was graph)
# ---------------------------------------------------------------------------

class TestPlotter(KernelTest):
    """
    Plotting utilities built on top of :class:`KernelTest`.

    Construct from a fully-estimated ``KernelTest`` instance::

        plotter = TestPlotter(test)
        plotter.plot_running_maximum()
    """

    def __init__(self, kernel_test: KernelTest) -> None:
        for key, val in kernel_test.__dict__.items():
            self.__setattr__(key, val)

    def _bound(self, alpha: float) -> list:
        """Gumbel-type critical bound for the running maximum at level ``alpha``."""
        if not hasattr(self, 'gaussian'):
            self.gauss()
            print('Gaussian process computed.')
        return gumbel_running_max_bound(len(self.gaussian), alpha)

    def plot_running_maximum(self) -> None:
        _, scalar_gauss = self.transform_1D_gauss()
        bound1 = self._bound(0.99)
        bound5 = self._bound(0.95)
        bound10 = self._bound(0.90)
        bound50 = self._bound(0.50)

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].plot(running_maximum(scalar_gauss), color='black', lw=1,
                    label='Theoretical 95% running maximum')
        axs[0].plot(bound1, color='grey', ls='-', lw=1, label='Empirical running maximum (1%)')
        axs[0].plot(bound5, color='grey', ls='dashed', lw=1, label='Empirical running maximum (5%)')
        axs[0].plot(bound10, color='grey', ls='dotted', lw=1, label='Empirical running maximum (10%)')
        axs[0].plot(bound50, color='red', ls='dashed', lw=1, label='Empirical running maximum (50%)')
        axs[0].plot(scalar_gauss, color='C0', lw=1, label='Standardised Gaussian Process')
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, ncol=3)
        axs[0].set_title('Running maximum')
        axs[0].grid(True)

        T = self.kernel_params.get('T')
        n = self.kernel_params.get('n')
        t = np.linspace(0, T, n)
        axs[1].plot(t, self.data['process 1'], label='Process 1')
        axs[1].plot(t, self.data['process 2'], label='Process 2')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Value')
        axs[1].set_title('Processes')
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, ncol=2)
        axs[1].grid(True)
        fig.tight_layout()
        plt.show()

    def plot_estimates(
        self,
        true_sigma1: Optional[npt.ArrayLike] = None,
        true_sigma2: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Plot state- and time-domain volatility estimates alongside
        optional true volatility paths.
        """
        est_time = self.time_estimates['estimate']
        est_state = self.kernel_estimates['estimate']

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        axs[0].plot([m.item(0) ** 0.5 for m in est_state], c='C1', lw=1, label='state estimate')
        axs[1].plot([m.item(3) ** 0.5 for m in est_state], c='C1', lw=1, label='state estimate')
        axs[0].plot([m.item(0) ** 0.5 for m in est_time], c='C0', lw=1, label='time estimate')
        axs[1].plot([m.item(3) ** 0.5 for m in est_time], c='C0', lw=1, label='time estimate')
        if true_sigma1 is not None:
            axs[0].plot(true_sigma1, c='C2', lw=1, label='true volatility')
        if true_sigma2 is not None:
            axs[1].plot(true_sigma2, c='C2', lw=1, label='true volatility')
        axs[0].set_title('Process 1')
        axs[1].set_title('Process 2')
        for ax in axs:
            ax.legend()
        fig.tight_layout()
        plt.show()
