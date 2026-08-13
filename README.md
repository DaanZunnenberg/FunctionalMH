# mht -- Multivariate Nonparametric Stationarity Test for Diffusion Processes

A nonparametric test for stationarity of a multivariate Itô diffusion process,
based on comparing two estimators of the diffusion matrix whose convergence
rates diverge under the alternative of nonstationarity.

## Statistical background

### Setup

The process $X$ is assumed to be a $d$-dimensional time-homogeneous Itô diffusion

$$dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t, \qquad t \ge 0,$$

observed discretely on $[0,T]$ at $n$ equidistant points with step $\Delta_n = T/n$.
The object of interest is the diffusion matrix $c(x) = \sigma(x)\sigma(x)^\top$.

- $H_0$: $X$ is stationary (equivalently, positive Harris recurrent with $\alpha$-index $1$).
- $H_1$: $X$ is nonstationary ($\alpha$-index $< 1$, so the occupation measure $L_{n,T}(x) = o_p(T)$).

### Key identification result

The test relies on how much "occupation time" the process spends near any given point $x$ as more data arrives. A stationary process keeps revisiting every point in its state space at a rate proportional to the elapsed time $T$ (it has a well-defined long-run occupation density). A nonstationary process, by contrast, drifts away over time, so the time it spends near any fixed point grows more slowly than $T$. This difference in growth rate — proportional to $T$ under stationarity, sub-linear in $T$ under nonstationarity — is the statistical foothold the whole test is built on (Darling–Kac 1957; Meyn–Tweedie 2012).

### Two estimators of the diffusion matrix

The test follows a Durbin–Wu–Hausman-style logic: build two different estimators of the same quantity — the diffusion (instantaneous covariance) matrix $c(x)$ — that should agree under stationarity but drift apart under nonstationarity, then test whether they actually agree.

**Time-domain smoother** (Jacod–Protter 2011): estimates $c(x)$ by locally averaging the outer products of consecutive increments in a short window of time around a point. This estimator's accuracy depends only on how many observations fall in that time window, not on whether the process is stationary — it converges at the same rate regardless of $H_0$ or $H_1$, so it serves as a stable, stationarity-invariant baseline.

**State-domain smoother** (Bandi–Moloche 2018): estimates $c(x)$ by kernel-weighting squared increments according to how close the process's *value* is to $x$ (a Nadaraya–Watson-style regression in the state space, not the time domain), using a near-optimal bandwidth that shrinks as more data arrives. Under stationarity this estimator converges at a fast rate governed by how often the process revisits the neighborhood of $x$. Under nonstationarity, since the process no longer reliably revisits $x$, this estimator's error blows up at that same rate instead of shrinking.

### Test statistic

Under $H_0$, the two estimators can be shown to be asymptotically independent (via a $\beta$-mixing argument), so their standardized difference behaves like a standard normal random variable at every point in time. Under $H_1$, because the state-domain estimator's error diverges while the time-domain estimator's does not, this standardized difference grows without bound instead of staying bounded. The time axis is rescaled so that the resulting sequence of standardized differences has vanishing correlation between distant points, which is what the extreme-value theory in the next step requires.

### Critical bound (running maximum)

Because the test statistic is really a whole path over time rather than a single number, the test looks at the largest absolute value the standardized difference ever reaches along that path (its running maximum). For a stationary sequence whose correlations vanish quickly enough, classical extreme-value theory (Pickands 1969; Berman 1964) tells us how that running maximum behaves as more data arrives: after a known rescaling, it converges in distribution to a Gumbel (extreme-value) distribution. This gives an explicit, data-driven threshold for the running maximum. $H_0$ is rejected at level $\alpha$ whenever the observed running maximum exceeds that threshold.

### Multiple-testing supplements

For simulation studies, Benjamini–Hochberg (1995) and Benjamini–Yekutieli (2001) FDR procedures are applied to the $z$-score matrix to control false discoveries across replications and grid points.

The package also provides:

- Batch KPSS and Leybourne-McCabe stationarity tests on process paths for comparison.

## Installation

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10.

## Quick start

```python
import numpy as np
from mht.models.processes import BivariateOUProcess
from mht.testing.kernel_test import KernelTest, Kernel, TestPlotter

# Simulate a bivariate OU process
ou_config = {
    'T': 365, 'dt': 1/20,
    'sigma1': np.sqrt(2), 'sigma2': np.sqrt(2),
    'theta1': 0.2, 'theta2': 0.2,
    'rho': 0.75,
}
process = BivariateOUProcess(**ou_config)
process.simulate(seed=1)
X, T, n = process.config()

# Set up the test configuration
config = {
    'data': X,
    'kernel_params': {
        'bandwidth': np.sqrt(3) * 9 / ((n ** (1/6)) * np.log(n)),
        'n': n, 'T': T,
        'kernel': Kernel.BaseKernel,
    },
    'time_params': {'bandwidth': 200 * T / n, 'n': n, 'T': T},
}

# Estimate and test
test = KernelTest(**config)
test.time_domain_smoother(lamb=0.99)
test.state_domain_smoother(dist=True)   # True = use KDE for joint density
test.gauss()

bound, scalar_gauss = test.transform_1D_gauss()

# Plot
plotter = TestPlotter(test)
plotter.plot_running_maximum()
```

See `notebooks/example.ipynb` for a full worked example including
Monte Carlo simulations and comparison with KPSS / Leybourne-McCabe.

## Repository structure

```
src/mht/
    testing/
        kernel_test.py        # KernelTest, Simulator, TestPlotter
        hypothesis.py         # MultipleHypTest, UnitRootTest, LaTeXTable
        leybourne_mccabe.py   # Leybourne-McCabe test (single canonical copy)
    models/
        processes.py          # BivariateOUProcess, BivariateCorrelatedBM, ...
    io/
        reader.py             # Reader class for simulation CSV files
    viz/                      # TestPlotter re-exported here
    utils/
        decorators.py
simulations/                  # Pre-computed CSV simulation results
notebooks/
    example.ipynb
tests/
    test_processes.py
    test_kernel_test.py
```

## References

- Bandi, F.M., & Moloche, G. (2018). On the functional estimation of multivariate
  diffusion processes. *Econometric Theory*, 34(4): 896–946.
- Jacod, J., & Protter, P. (2011). *Discretization of Processes*. Springer.
- Darling, D.A., & Kac, M. (1957). On occupation times for Markoff processes.
  *Transactions of the American Mathematical Society*, 84(2): 444–458.
- Berman, S.M. (1964). Limit theorems for the maximum term in stationary sequences.
  *Annals of Mathematical Statistics*, 35(2): 502–516.
- Pickands, J. (1969). Asymptotic properties of the maximum in a stationary Gaussian
  process. *Transactions of the American Mathematical Society*, 145: 75–86.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  *Journal of the Royal Statistical Society B*, 57(1): 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *Annals of Statistics*, 29(4): 1165–1188.
- Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a unit root.
  *Journal of Business and Economic Statistics*, 12: 157–166.
- Meyn, S.P., & Tweedie, R.L. (2009). *Markov Chains and Stochastic Stability* (2nd ed.).
  Cambridge University Press.
