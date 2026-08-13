"""
Stochastic process models used in the diffusion matrix test simulations.

Classes
-------
BivariateOUProcess
    Bivariate correlated Ornstein-Uhlenbeck process.
BivariateCorrelatedBM
    Bivariate correlated Brownian motion (drifted).
BivariateNonHomogeneous
    Bivariate diffusion with a time-varying, non-homogeneous volatility.
BivariateCorrelatedDiffusion
    Bivariate diffusion with polynomial diffusion coefficient (Milstein scheme).
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

_COLUMNS: list[str] = ['process 1', 'process 2']


def Fourier4(x: float, gamma: float) -> float:
    """Fourier-4 spectral density weight: 1 / (1 + x^2)^(2*gamma)."""
    return 1 / (1 + x ** 2) ** (2 * gamma)


def _to_dataframe(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating],
                   name: str, **kwargs: Any) -> pd.DataFrame:
    """Assemble a two-column bivariate-path DataFrame with a ``.name`` label."""
    df = pd.DataFrame(np.column_stack([x, y]), columns=_COLUMNS)
    df.name = name  # type: ignore[attr-defined]
    for key, val in kwargs.items():
        setattr(df, key, val)
    return df


def _plot_paths(t: npt.NDArray[np.floating], x: npt.NDArray[np.floating],
                 y: npt.NDArray[np.floating], title: str,
                 label1: str, label2: str) -> None:
    """Shared two-line time-series plot used by every process's ``.plot()``."""
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, label=label1)
    plt.plot(t, y, label=label2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


class BivariateOUProcess:
    """
    Bivariate correlated Ornstein-Uhlenbeck (OU) process.

    dX_i(t) = theta_i (mu_i - X_i(t)) dt + sigma_i dW_i(t),  i = 1, 2

    where (W_1, W_2) are correlated Brownian motions with correlation rho.

    Parameters
    ----------
    theta1, theta2 : float
        Mean-reversion rates.
    mu1, mu2 : float
        Long-run means.
    sigma1, sigma2 : float
        Volatility parameters.
    rho : float
        Instantaneous correlation.
    T : float
        Total time horizon.
    dt : float
        Time step.
    """

    def __init__(
        self,
        theta1: float = 0.015,
        theta2: float = 0.006,
        mu1: float = 0.5,
        mu2: float = 0.5,
        sigma1: float = 1.5,
        sigma2: float = 3.0,
        rho: float = 0.5,
        T: float = 200,
        dt: float = 1 / 20,
    ) -> None:
        self.theta1 = theta1
        self.theta2 = theta2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.t = np.linspace(0, T, self.n_steps)

        self.corr_matrix = np.array([[1, rho], [rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        self.x = np.zeros(self.n_steps)
        self.y = np.zeros(self.n_steps)
        self.x[0] = mu1
        self.y[0] = mu2

    def simulate(self, seed: Union[int, bool] = False) -> None:
        """Simulate one path via Euler-Maruyama discretization (sequential by construction)."""
        if seed:
            np.random.seed(seed)
        for i in range(1, self.n_steps):
            z = np.random.normal(size=2)
            dz = self.L @ z * np.sqrt(self.dt)
            self.x[i] = self.x[i - 1] + self.theta1 * (self.mu1 - self.x[i - 1]) * self.dt + self.sigma1 * dz[0]
            self.y[i] = self.y[i - 1] + self.theta2 * (self.mu2 - self.y[i - 1]) * self.dt + self.sigma2 * dz[1]

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the simulated path as a labeled two-column DataFrame."""
        return _to_dataframe(self.x, self.y, 'Ornstein Uhlenbeck', **kwargs)

    def config(self) -> tuple[pd.DataFrame, float, int]:
        """Return ``(data, T, n_steps)`` for downstream estimators."""
        return self.dataframe(), self.T, self.n_steps

    def plot(self) -> None:
        """Plot both process paths against time."""
        _plot_paths(self.t, self.x, self.y,
                    'Bivariate Correlated Ornstein-Uhlenbeck Process',
                    'OU Process 1', 'OU Process 2')


class BivariateCorrelatedBM:
    """
    Bivariate correlated Brownian motion with drift.

    dX_i(t) = mu_i dt + sigma_i dW_i(t),  i = 1, 2

    Parameters
    ----------
    mu1, mu2 : float
        Drift coefficients.
    sigma1, sigma2 : float
        Diffusion coefficients (constant).
    rho : float
        Instantaneous correlation.
    T : float
        Total time horizon.
    dt : float
        Time step.
    """

    def __init__(
        self,
        mu1: float = 0.01,
        mu2: float = 0.01,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
        rho: float = 0.5,
        T: float = 1.0,
        dt: float = 0.01,
    ) -> None:
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.t = np.linspace(0, T, self.n_steps)

        self.corr_matrix = np.array([[1, rho], [rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        self.x = np.zeros(self.n_steps)
        self.y = np.zeros(self.n_steps)

    def simulate(self, seed: Union[int, bool] = False) -> None:
        """Simulate one path via Euler-Maruyama discretization (sequential by construction)."""
        if seed:
            np.random.seed(seed)
        for i in range(1, self.n_steps):
            z = np.random.normal(size=2)
            dz = self.L @ z * np.sqrt(self.dt)
            self.x[i] = self.x[i - 1] + self.mu1 * self.dt + self.sigma1 * dz[0]
            self.y[i] = self.y[i - 1] + self.mu2 * self.dt + self.sigma2 * dz[1]

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the simulated path as a labeled two-column DataFrame."""
        return _to_dataframe(self.x, self.y, 'Correlated Brownian motion', **kwargs)

    def config(self) -> tuple[pd.DataFrame, float, int]:
        """Return ``(data, T, n_steps)`` for downstream estimators."""
        return self.dataframe(), self.T, self.n_steps

    def plot(self) -> None:
        """Plot both process paths against time."""
        _plot_paths(self.t, self.x, self.y,
                    'Bivariate Correlated Brownian Motion',
                    'Correlated BM Process 1', 'Correlated BM Process 2')


class BivariateNonHomogeneous:
    """
    Bivariate diffusion with time-varying (non-homogeneous) volatility.

    dX_i(t) = b * mu(X_i(t), t) dt + sigma(t) dW_i(t),  i = 1, 2

    where sigma(t) = 2.01 + alpha * sin(16 * pi * t / T).

    Parameters
    ----------
    T : float
    dt : float
    rho : float
        Instantaneous correlation.
    alpha : float
        Amplitude of the sinusoidal volatility component.
    b : float
        Scaling of the drift term.
    """

    def __init__(
        self,
        T: float = 1.0,
        dt: float = 0.01,
        rho: float = 0.5,
        alpha: float = 1.0,
        b: float = 1.0,
    ) -> None:
        self.rho = rho
        self.alpha = alpha
        self.b = b
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.t = np.linspace(0, T, self.n_steps)

        self.corr_matrix = np.array([[1, rho], [rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        self.x = np.zeros(self.n_steps)
        self.y = np.zeros(self.n_steps)

    def mu(self, x: float, t: float) -> float:
        """Drift term: -2x (mean-reverting toward the origin)."""
        return -2 * x

    def sigma(self, t: float) -> float:
        """Time-varying (non-homogeneous) volatility: 2.01 + alpha * sin(16*pi*t/T)."""
        return 2.01 + self.alpha * np.sin(16 * np.pi * t / self.T)

    def simulate(self, seed: Union[int, bool] = False) -> None:
        """Simulate one path via Euler-Maruyama discretization (sequential by construction)."""
        if seed:
            np.random.seed(seed)
        for i in range(1, self.n_steps):
            z = np.random.normal(size=2)
            dz = self.L @ z * np.sqrt(self.dt)
            t = i * self.dt
            self.x[i] = self.x[i - 1] + self.b * self.mu(self.x[i - 1], t) * self.dt + self.sigma(t) * dz[0]
            self.y[i] = self.y[i - 1] + self.b * self.mu(self.y[i - 1], t) * self.dt + self.sigma(t) * dz[1]

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the simulated path as a labeled two-column DataFrame."""
        return _to_dataframe(self.x, self.y, 'Correlated Diffusion', **kwargs)

    def config(self) -> tuple[pd.DataFrame, float, int]:
        """Return ``(data, T, n_steps)`` for downstream estimators."""
        return self.dataframe(), self.T, self.n_steps

    def plot(self) -> None:
        """Plot both process paths against time."""
        _plot_paths(self.t, self.x, self.y,
                    'Bivariate Non-Homogeneous Diffusion',
                    'Correlated diffusion Process 1', 'Correlated diffusion Process 2')


class BivariateCorrelatedDiffusion:
    """
    Bivariate diffusion with polynomial diffusion coefficient, integrated
    via the Milstein scheme.

    dX_i(t) = mu_i dt + sigma * (1 + X_i(t)^2)^gamma dW_i(t),  i = 1, 2

    Parameters
    ----------
    mu1, mu2 : float
        Drift coefficients.
    rho : float
        Instantaneous correlation.
    T : float
    dt : float
    sigma1, sigma2 : float
        Scale of the diffusion coefficient (note: both use sigma1 internally
        as in the original code).
    gamma : float
        Polynomial exponent.
    """

    def __init__(
        self,
        mu1: float = 0.0,
        mu2: float = 0.0,
        rho: float = 0.5,
        T: float = 1.0,
        dt: float = 0.01,
        sigma1: float = 0.1,
        sigma2: float = 0.1,
        gamma: float = 1.0,
    ) -> None:
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gamma = gamma
        self.rho = rho
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.t = np.linspace(0, T, self.n_steps)

        self.corr_matrix = np.array([[1, rho], [rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        self.x = np.zeros(self.n_steps)
        self.y = np.zeros(self.n_steps)
        self.x[0] = np.random.normal()
        self.y[0] = np.random.normal()

    def _milstein_correction(self, x: float, dz: float, dt: float) -> float:
        """Milstein correction term for the polynomial diffusion coefficient."""
        bo = (1 + x ** 2) ** self.gamma
        bp = 2 * x * self.gamma * (1 + x ** 2) ** (self.gamma - 1)
        return 0.5 * bo * bp * (dz ** 2 - dt)

    def simulate(self, seed: Union[int, bool] = False) -> None:
        """Simulate one path via the Milstein scheme (sequential by construction)."""
        if seed:
            np.random.seed(seed)
        for i in range(1, self.n_steps):
            z = np.random.normal(size=2)
            dz = self.L @ z * np.sqrt(self.dt)
            self.x[i] = (
                self.x[i - 1]
                + self.mu1 * self.dt
                + self.sigma1 * (1 + self.x[i - 1] ** 2) ** self.gamma * dz[0]
                + self._milstein_correction(self.x[i - 1], dz[0], self.dt)
            )
            self.y[i] = (
                self.y[i - 1]
                + self.mu2 * self.dt
                + self.sigma1 * (1 + self.y[i - 1] ** 2) ** self.gamma * dz[1]
                + self._milstein_correction(self.y[i - 1], dz[1], self.dt)
            )

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the simulated path as a labeled two-column DataFrame."""
        return _to_dataframe(self.x, self.y, 'Correlated Diffusion', **kwargs)

    def config(self) -> tuple[pd.DataFrame, float, int]:
        """Return ``(data, T, n_steps)`` for downstream estimators."""
        return self.dataframe(), self.T, self.n_steps

    def plot(self) -> None:
        """Plot both process paths against time."""
        _plot_paths(self.t, self.x, self.y,
                    'Bivariate Correlated Diffusion',
                    'Correlated diffusion Process 1', 'Correlated diffusion Process 2')
