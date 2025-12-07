"""
Time series simulation utilities.

Provides functions for generating synthetic univariate time series processes
following standard autoregressive specifications. Useful for Monte Carlo
simulations, model validation, and testing forecast methodologies.
"""

from typing import Optional

import numpy as np
import pandas as pd


def generate_ar1(n: int = 500, phi: float = 0.7, sigma: float = 1.0,
                 x0: float = 0.0, seed: Optional[int] = None) -> pd.Series:
    """
    Simulate a first-order autoregressive (AR(1)) process.
    
    Generates a univariate AR(1) time series following the specification:
        x_t = φ * x_{t-1} + ε_t
        where ε_t ~ i.i.d. N(0, σ²)
    
    The process is stationary if |φ| < 1. For φ = 1, it becomes a random walk.
    
    Args:
        n: Sample size (number of observations to generate).
        phi: Autoregressive coefficient. For stationarity, require |φ| < 1.
            Default 0.7 produces a moderately persistent stationary process.
        sigma: Standard deviation of innovation shocks ε_t. Default 1.0.
        x0: Initial value x_0 for the process. Default 0.0 (unconditional mean
            for stationary process with zero mean innovations).
        seed: Random seed for reproducibility. If None, uses system entropy.
    
    Returns:
        pandas Series with integer index [0, 1, ..., n-1] named "t" and
        series name indicating AR(1) parameter value.
    
    Raises:
        ValueError: If n <= 0 or sigma < 0.
    
    Example:
        >>> # Generate stationary AR(1) with persistence 0.8
        >>> series = generate_ar1(n=1000, phi=0.8, sigma=0.5, seed=42)
        >>> print(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    if sigma < 0:
        raise ValueError(f"Noise standard deviation sigma must be non-negative, got {sigma}")
    
    # Initialize random number generator for reproducible simulations
    rng = np.random.default_rng(seed)
    
    # Generate innovation sequence: white noise shocks
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    
    # Initialize array and set initial condition
    x = np.zeros(n)
    x[0] = x0
    
    # Recursively generate AR(1) process: x_t = φ * x_{t-1} + ε_t
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    
    # Return as pandas Series with named index for time axis
    return pd.Series(x, index=pd.RangeIndex(n, name="t"), name=f"ar1_phi_{phi}")

def generate_trend_seasonal(
    n: int = 500,
    trend_slope: float = 0.01,
    period: int = 24,
    amplitude: float = 1.0,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Simulate a time series with linear trend and seasonal component.
    
    Generates a univariate time series following the specification:
        y_t = trend_slope * t + amplitude * sin(2πt / period) + ε_t
        where ε_t ~ i.i.d. N(0, σ²)
    
    This process combines a deterministic linear trend with a periodic seasonal
    pattern and additive Gaussian noise. Useful for testing forecasting methods
    on non-stationary series with known structure.
    
    Args:
        n: Sample size (number of observations to generate). Must be positive.
        trend_slope: Slope of the linear trend component. Default 0.01.
        period: Length of seasonal cycle (e.g., 24 for hourly daily pattern,
            12 for monthly annual pattern). Must be positive.
        amplitude: Amplitude of the seasonal sine wave. Default 1.0.
        sigma: Standard deviation of innovation shocks ε_t. Must be non-negative.
            Default 0.5.
        seed: Random seed for reproducibility. If None, uses system entropy.
    
    Returns:
        pandas Series with integer index [0, 1, ..., n-1] named "t" and
        series name "trend_seasonal".
    
    Raises:
        ValueError: If n <= 0, period <= 0, or sigma < 0.
    
    Example:
        >>> # Generate hourly data with daily seasonality (24-hour cycle)
        >>> series = generate_trend_seasonal(n=1000, period=24, seed=42)
        >>> # Generate monthly data with annual seasonality
        >>> series = generate_trend_seasonal(n=120, period=12, seed=42)
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    if period <= 0:
        raise ValueError(f"Seasonal period must be positive, got {period}")
    if sigma < 0:
        raise ValueError(f"Noise standard deviation sigma must be non-negative, got {sigma}")
    
    # Initialize random number generator for reproducible simulations
    rng = np.random.default_rng(seed)
    
    # Generate time index
    t = np.arange(n)
    
    # Linear trend component
    trend = trend_slope * t
    
    # Seasonal component: sinusoidal pattern
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    
    # Additive noise
    noise = rng.normal(loc=0.0, scale=sigma, size=n)
    
    # Combine components
    return pd.Series(
        trend + seasonal + noise,
        index=pd.RangeIndex(n, name="t"),
        name="trend_seasonal"
    )

def generate_regime_switching_noise(
    n: int = 800,
    sigma_low: float = 0.5,
    sigma_high: float = 2.0,
    p_up: float = 0.02,
    p_down: float = 0.10,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a zero-mean process with switching volatility regimes.

    There are two regimes:
      - 0: low volatility (sigma_low)
      - 1: high volatility (sigma_high)

    Transitions:
      - from 0 -> 1 with probability p_up
      - from 1 -> 0 with probability p_down

    Returns:
      DataFrame with columns:
        - 'value': the observed process
        - 'regime': 0 (low vol) or 1 (high vol)
    """
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n, dtype=int)
    values = np.zeros(n)

    for t in range(1, n):
        prev_regime = regimes[t - 1]
        if prev_regime == 0:
            # chance to jump from low to high
            regimes[t] = 1 if rng.random() < p_up else 0
        else:
            # chance to go back to low
            regimes[t] = 0 if rng.random() < p_down else 1

    for t in range(n):
        sigma = sigma_low if regimes[t] == 0 else sigma_high
        values[t] = rng.normal(loc=0.0, scale=sigma)

    index = pd.RangeIndex(n, name="t")
    return pd.DataFrame({"value": values, "regime": regimes}, index=index)

def generate_garch_like(
    n: int = 1000,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.6,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a GARCH(1,1)-like process:
    
        r_t = sigma_t * epsilon_t
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Returns a DataFrame with:
        - value: the return series
        - sigma: conditional volatility
        - var: conditional variance
    """

    rng = np.random.default_rng(seed)

    r = np.zeros(n)
    sigma2 = np.zeros(n)

    # Initialize variance
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = rng.normal(scale=np.sqrt(sigma2[0]))

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = rng.normal(scale=np.sqrt(sigma2[t]))

    df = pd.DataFrame(
        {
            "value": r,
            "sigma": np.sqrt(sigma2),
            "var": sigma2,
        }
    )
    df.index.name = "t"
    return df

def generate_structural_break_series(
    n: int = 800,
    break_point: int = 400,
    mu1: float = 0.0,
    mu2: float = 3.0,
    phi: float = 0.5,
    sigma: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate an AR(1)-type series with a single structural break in the mean.

    For t < break_point:
        x_t = mu1 + phi * (x_{t-1} - mu1) + eps_t
    For t >= break_point:
        x_t = mu2 + phi * (x_{t-1} - mu2) + eps_t

    Returns:
        DataFrame with columns:
            - value: the series
            - regime: 0 (pre-break) or 1 (post-break)
    """
    rng = np.random.default_rng(seed)

    if break_point <= 0 or break_point >= n:
        raise ValueError("break_point must be between 0 and n-1")

    x = np.zeros(n)
    regimes = np.zeros(n, dtype=int)

    # initialize at first mean
    x[0] = mu1 + rng.normal(scale=sigma)

    for t in range(1, n):
        if t < break_point:
            mu = mu1
            regimes[t] = 0
        else:
            mu = mu2
            regimes[t] = 1

        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(scale=sigma)

    df = pd.DataFrame({"value": x, "regime": regimes})
    df.index.name = "t"
    return df

import numpy as np
import pandas as pd

def generate_var1(
    n: int = 800,
    A: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a 2D VAR(1) process:

        x_t = mu + A x_{t-1} + eps_t

    where x_t is 2-dimensional, A is 2x2, eps_t ~ N(0, sigma^2 I).

    Args:
        n: length of the series
        A: 2x2 coefficient matrix. If None, a stable default is used.
        mu: 2D mean vector. If None, zeros are used.
        sigma: standard deviation of the noise innovations.
        seed: random seed.

    Returns:
        DataFrame with columns ['y1', 'y2'].
    """
    rng = np.random.default_rng(seed)

    if A is None:
        # A reasonably stable matrix with some cross-effects
        A = np.array([[0.5, 0.2],
                      [-0.3, 0.4]], dtype=float)

    if mu is None:
        mu = np.array([0.0, 0.0], dtype=float)

    x = np.zeros((n, 2), dtype=float)
    # start near the mean
    x[0] = mu + rng.normal(scale=sigma, size=2)

    for t in range(1, n):
        eps = rng.normal(scale=sigma, size=2)
        x[t] = mu + A @ x[t - 1] + eps

    df = pd.DataFrame(x, columns=["y1", "y2"])
    df.index.name = "t"
    return df