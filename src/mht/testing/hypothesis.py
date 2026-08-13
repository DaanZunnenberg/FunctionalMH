"""
Multiple hypothesis testing utilities.

Classes
-------
MultipleHypTest
    Benjamini-Hochberg and Benjamini-Yekutieli procedures on z-score matrices.
UnitRootTest
    Batch KPSS and Leybourne-McCabe unit-root tests.
LaTeXTable
    Format rejection-rate results as a LaTeX table.
"""
from __future__ import annotations

import functools
import warnings
from functools import cache
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.tsa.stattools import kpss
from tqdm import tqdm

from mht.testing.leybourne_mccabe import Leybourne
from mht.utils.decorators import ignore_unhashable


def _ignore_warning(warning_type: type[Warning]) -> Callable:
    """Decorator: suppress a specific warning category inside the wrapped call."""
    def inner(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=warning_type)
                return func(*args, **kwargs)
        return wrapper
    return inner


class MultipleHypTest:
    """
    Apply Benjamini-Hochberg or Benjamini-Yekutieli FDR control to a
    matrix of z-scores (each column is one simulation run).

    Parameters
    ----------
    z_scores : pd.DataFrame
        Columns are runs, rows are time points.
    two_sides : bool
        Convert z-scores to two-sided p-values.
    remove_zero_rows : bool
        Drop rows that are exactly zero before testing.
    run_on_call : bool
        Execute ``run()`` immediately on construction.
    """

    def __init__(
        self,
        z_scores: pd.DataFrame,
        two_sides: bool = True,
        remove_zero_rows: bool = True,
        run_on_call: bool = False,
        **kwargs: Any,
    ) -> None:
        if remove_zero_rows:
            z_scores = z_scores[z_scores != 0].dropna()
        self.z_scores = z_scores
        self.two_sides = two_sides
        self.n_obs = len(z_scores)
        self.n_runs = len(z_scores.T)
        self.rejections: dict = {}

        if kwargs.get('name'):
            name = kwargs.pop('name')
            print(f' {name} '.center(70, '-'))
        if run_on_call:
            self.run(**kwargs)

    @ignore_unhashable
    @cache
    def p_values(self, z_scores: Any, **kwargs: Any) -> np.ndarray:
        """Convert z-scores to sorted p-values."""
        pvals = []
        if self.two_sides:
            for z in z_scores:
                pvals.append(1 - 2 * (scipy.stats.norm.cdf(np.abs(z), **kwargs) - 0.5))
        else:
            raise NotImplementedError('One-sided p-values not implemented.')
        self.pvals = np.sort(pvals)
        return np.sort(pvals)

    @ignore_unhashable
    @cache
    def benjamini_hochberg_yekutieli(
        self, p_vals: np.ndarray, method: str = 'hochberg', q: float = 0.05
    ) -> int:
        """Return 1 if H0 is rejected for at least one hypothesis, else 0."""
        if method == 'yekutieli':
            assert all(p_vals[i] <= p_vals[i + 1] for i in range(self.n_obs - 1))
            ns: float = np.sum([1 / i for i in range(1, self.n_obs + 1)])
            for i in range(1, self.n_obs + 1):
                if p_vals[i - 1] <= i * q / (self.n_obs * ns):
                    return 1
            return 0

        elif method == 'hochberg':
            assert all(p_vals[i] <= p_vals[i + 1] for i in range(self.n_obs - 1))
            for i in range(1, self.n_obs + 1):
                if p_vals[i - 1] <= i * q / self.n_obs:
                    return 1
            return 0

        else:
            raise ValueError("method must be 'hochberg' or 'yekutieli'")

    @ignore_unhashable
    @cache
    def BHY(
        self,
        method: str = 'hochberg',
        q: tuple[float, ...] = (0.1, 0.05, 0.01),
        ret: bool = False,
    ) -> Optional[dict]:
        """Run the BHY procedure for each significance level in ``q``."""
        rejections: dict = {}
        for sig_level in q:
            rejection_count = 0
            for _, z_score in tqdm(self.z_scores.T.iterrows()):
                pvals = self.p_values(z_score)
                rejection_count += self.benjamini_hochberg_yekutieli(
                    pvals, method=method, q=sig_level
                )
            rejections[sig_level] = rejection_count / self.n_runs
            self.rejections = {**self.rejections, **{method: rejections}}
        if ret:
            return rejections
        return None

    def run(self, q: Union[list, tuple] = (0.1, 0.05, 0.01)) -> None:
        for method in ('hochberg', 'yekutieli'):
            print(f'MultipleHypTest@Self.run: running [{method}]...')
            self.BHY(method=method, q=tuple(q))
            self._print_method(method)
        print('Finished'.center(70, '-'))

    def _print_method(self, method: str) -> None:
        print(f'\n{method.title()}:')
        for key, val in self.rejections.get(method, {}).items():
            print(
                'Rejection rate: {}% (α = {}%)'.format(
                    self._fmt(val), self._fmt(key)
                )
            )
        print()

    @staticmethod
    def bernoulli_variance(p: float, n: int) -> float:
        return p * (1 - p) / n

    @staticmethod
    def _fmt(flt: float) -> str:
        flt = float(flt)
        if flt < 0.1:
            return str(float(np.round(100 * flt))).rjust(4, ' ').ljust(5, '0')
        return str(float(np.round(100 * flt))).ljust(5, '0')

    def __repr__(self) -> str:
        return ''


class UnitRootTest:
    """
    Apply KPSS and Leybourne-McCabe stationarity tests to each column of a
    DataFrame and compute pairwise rejection rates.

    Parameters
    ----------
    processes : pd.DataFrame
        Columns alternate between the two processes of each simulation run.
    remove_zero_rows : bool
    run_on_call : bool
    """

    def __init__(
        self,
        processes: pd.DataFrame,
        remove_zero_rows: bool = True,
        run_on_call: bool = False,
        **kwargs: Any,
    ) -> None:
        if remove_zero_rows:
            processes = processes[processes != np.nan].dropna()
        self.processes = processes
        self.n_obs = len(processes)
        self.n_pairs = len(processes.T) / 2
        self.rejections: dict = {}
        self.rejections_kpss: dict = {}
        self.rejections_lm: dict = {}
        self.p_values_kpss: dict = {}
        self.p_values_lm: dict = {}

        if kwargs.get('name'):
            name = kwargs.pop('name')
            print(f' {name} '.center(70, '-'))
        if run_on_call:
            self.run(**kwargs)

    @ignore_unhashable
    @cache
    @_ignore_warning(UserWarning)
    def _run_kpss(self, x: pd.DataFrame, **kwargs: Any) -> None:
        name: str = x.columns[0]
        result = kpss(x=x, **kwargs)
        self.p_values_kpss[name] = result[1]

    @ignore_unhashable
    @cache
    @_ignore_warning(UserWarning)
    def _run_lm(self, x: pd.DataFrame, **kwargs: Any) -> None:
        name: str = x.columns[0]
        result = Leybourne().run(x=x, **kwargs)
        self.p_values_lm[name] = result[1]

    @_ignore_warning(UserWarning)
    def lm(self, **kwargs: Any) -> None:
        for col in tqdm(self.processes.columns):
            self._run_lm(x=self.processes[[col]], **kwargs)
        self.p_values_lm = {'lm': self.p_values_lm}

    @_ignore_warning(UserWarning)
    def kpss(self, **kwargs: Any) -> None:
        for col in tqdm(self.processes.columns):
            self._run_kpss(x=self.processes[[col]], **kwargs)
        self.p_values_kpss = {'kpss': self.p_values_kpss}

    @ignore_unhashable
    @cache
    @_ignore_warning(UserWarning)
    def run(self, q: Union[list, tuple] = (0.1, 0.05, 0.01)) -> None:
        for method in ('kpss', 'lm'):
            print(f'MultipleHypTest@Self.run: running [{method}]...')
            getattr(self, method)()
            for level in q:
                p_vals = getattr(self, f'p_values_{method}')[method]
                rate = self._rejection_rate(p_vals, level, self.n_pairs)
                key = f'{method}_{level}'
                self.rejections[key] = rate
                if method == 'kpss':
                    self.rejections_kpss[key] = rate
                else:
                    self.rejections_lm[key] = rate
            self._print_method(method)
        print('Finished'.center(70, '-'))

    @staticmethod
    def _rejection_rate(p_vals: dict[str, float], q: float, n_pairs: float) -> float:
        rejections = 0
        p_vals1 = list(p_vals.values())[::2]
        p_vals2 = list(p_vals.values())[1::2]
        for pv1, pv2 in zip(p_vals1, p_vals2, strict=True):
            if (pv1 <= q + 1e-7) and (pv2 <= q + 1e-7):
                rejections += 1
        return rejections / n_pairs

    def _print_method(self, method: str) -> None:
        print(f'\n{method.title()}:')
        for key, val in self.rejections.items():
            if method in key:
                print(
                    'Rejection rate: {}% (α = {}%)'.format(
                        self._fmt(val), self._fmt(key.split('_')[1])
                    )
                )
        print()

    @staticmethod
    def bernoulli_variance(p: float, n: int) -> float:
        return p * (1 - p) / n

    @staticmethod
    def _fmt(flt: float) -> str:
        flt = float(flt)
        if flt < 0.1:
            return str(float(np.round(100 * flt))).rjust(4, ' ').ljust(5, '0')
        return str(float(np.round(100 * flt))).ljust(5, '0')

    def __repr__(self) -> str:
        return ''


class LaTeXTable:
    """
    Format simulation rejection-rate results as a LaTeX table fragment.

    Parameters
    ----------
    methods : dict
        Nested dict: ``{process_name: {horizon: {test: results}}}``.
    n : int
        Number of simulation runs (used for Bernoulli SE calculation).
    """

    def __init__(self, methods: dict[str, Any], n: int = 200) -> None:
        self.methods = methods
        self.n = n

    def header(self, name: str) -> str:
        s = r"""
        & \multicolumn{5}{c}{"""
        s += "{}".format(name) + '}\\'
        s += r"""\\cmidrule(lr){2-6}
        & \multicolumn{1}{c}{Running Maximum} & \multicolumn{1}{c}{Benjamini-Hochberg} & \multicolumn{1}{c}{Benjamini-Yekutieli} & \multicolumn{1}{c}{KPSS} & \multicolumn{1}{c}{Leybourne-McCabe}\\
          \cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6}
        $k$ & 5\% & 5\% & 5\% & 5\%  & 5\% \\
        \midrule
        """
        return s

    def vals(self, *args: Any) -> str:
        return r"""
        50 & {} & {} & {} & {} & {} \\
           & ({}) & ({}) & ({}) & ({}) & ({}) \\
        [.2cm]
        150 & {} & {} & {} & {} & {} \\
           & ({}) & ({}) & ({}) & ({}) & ({}) \\
        [.2cm]
        365 & {} & {} & {} & {} & {} \\
           & ({}) & ({}) & ({}) & ({}) & ({}) \\
        """.format(*args)

    @staticmethod
    def bernoulli_variance(p: float, n: int) -> float:
        return float(np.round(p * (1 - p) / n, 4))

    def generate(self) -> None:
        for idx, (name, vals) in enumerate(self.methods.items()):
            print(self.header(name))
            config = []
            for _, y in vals.items():
                std = []
                pvals = []
                for z in y.values():
                    p = list(list(z.values())[0].values())[1]
                    std.append(str(self.bernoulli_variance(p, self.n)).ljust(5, '0'))
                    pvals.append(str(p).ljust(5, '0'))
                config += pvals + std
            print(self.vals(*config))
            if idx == len(self.methods) - 1:
                print(r'\bottomrule')
            else:
                print(r'\midrule')
