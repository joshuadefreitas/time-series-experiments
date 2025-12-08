"""
Time series simulation utilities.

Provides functions for generating synthetic univariate time series processes
following standard autoregressive specifications. Useful for Monte Carlo
simulations, model validation, and testing forecast methodologies.
"""

from typing import Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# AR(1)
# -------------------------------------------------------------------

def generate_ar1(n: int = 500, phi: float = 0.7, sigma: float = 1.0,
                 x0: float = 0.0, seed: Optional[int] = None) -> pd.Series:
    """
    Simulate a first-order autoregressive (AR(1)) process.
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    if sigma < 0:
        raise ValueError(f"Noise standard deviation sigma must be non-negative, got {sigma}")

    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)

    x = np.zeros(n)
    x[0] = x0

    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]

    return pd.Series(x, index=pd.RangeIndex(n, name="t"), name=f"ar1_phi_{phi}")


# -------------------------------------------------------------------
# Trend + seasonal
# -------------------------------------------------------------------

def generate_trend_seasonal(
    n: int = 500,
    trend_slope: float = 0.01,
    period: int = 24,
    amplitude: float = 1.0,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Simulate a series with trend + seasonal component.
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    if period <= 0:
        raise ValueError(f"Seasonal period must be positive, got {period}")
    if sigma < 0:
        raise ValueError(f"Noise standard deviation sigma must be non-negative, got {sigma}")

    rng = np.random.default_rng(seed)

    t = np.arange(n)
    trend = trend_slope * t
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(loc=0.0, scale=sigma, size=n)

    return pd.Series(
        trend + seasonal + noise,
        index=pd.RangeIndex(n, name="t"),
        name="trend_seasonal",
    )


# -------------------------------------------------------------------
# Regime-switching volatility
# -------------------------------------------------------------------

def generate_regime_switching_noise(
    n: int = 800,
    sigma_low: float = 0.5,
    sigma_high: float = 2.0,
    p_up: float = 0.02,
    p_down: float = 0.10,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Zero-mean process with switching volatility regimes.
    """
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n, dtype=int)
    values = np.zeros(n)

    for t in range(1, n):
        if regimes[t - 1] == 0:
            regimes[t] = 1 if rng.random() < p_up else 0
        else:
            regimes[t] = 0 if rng.random() < p_down else 1

    for t in range(n):
        sigma = sigma_low if regimes[t] == 0 else sigma_high
        values[t] = rng.normal(loc=0.0, scale=sigma)

    index = pd.RangeIndex(n, name="t")
    return pd.DataFrame({"value": values, "regime": regimes}, index=index)


# -------------------------------------------------------------------
# GARCH(1,1)-like generator
# -------------------------------------------------------------------

def generate_garch_like(
    n: int = 1000,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.6,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a GARCH(1,1)-like process.
    """
    if alpha + beta >= 1:
        raise ValueError(
            "GARCH(1,1) requires alpha + beta < 1 for stationarity. "
            f"Got alpha + beta = {alpha + beta:.3f}"
        )

    rng = np.random.default_rng(seed)

    r = np.zeros(n)
    sigma2 = np.zeros(n)

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


# -------------------------------------------------------------------
# Structural break
# -------------------------------------------------------------------

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
    AR(1)-type series with structural break in mean.
    """
    rng = np.random.default_rng(seed)

    if break_point <= 0 or break_point >= n:
        raise ValueError("break_point must be between 0 and n-1")

    x = np.zeros(n)
    regimes = np.zeros(n, dtype=int)

    x[0] = mu1 + rng.normal(scale=sigma)

    for t in range(1, n):
        mu = mu1 if t < break_point else mu2
        regimes[t] = 0 if t < break_point else 1
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(scale=sigma)

    df = pd.DataFrame({"value": x, "regime": regimes})
    df.index.name = "t"
    return df


# -------------------------------------------------------------------
# VAR(1)
# -------------------------------------------------------------------

def generate_var1(
    n: int = 800,
    A: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a 2D VAR(1) process.
    """
    rng = np.random.default_rng(seed)

    if A is None:
        A = np.array([[0.5, 0.2], [-0.3, 0.4]], dtype=float)

    if mu is None:
        mu = np.array([0.0, 0.0], dtype=float)

    x = np.zeros((n, 2), dtype=float)
    x[0] = mu + rng.normal(scale=sigma, size=2)

    for t in range(1, n):
        eps = rng.normal(scale=sigma, size=2)
        x[t] = mu + A @ x[t - 1] + eps

    df = pd.DataFrame(x, columns=["y1", "y2"])
    df.index.name = "t"
    return df


# -------------------------------------------------------------------
# Logistic map (chaos)
# -------------------------------------------------------------------

def generate_logistic_map(
    n: int = 800,
    r: float = 3.9,
    x0: float = 0.2,
) -> pd.Series:
    """
    Generate logistic map series.
    """
    x = np.zeros(n, dtype=float)
    x[0] = x0

    for t in range(1, n):
        x[t] = r * x[t - 1] * (1.0 - x[t - 1])

    s = pd.Series(x)
    s.index.name = "t"
    s.name = "value"
    return s


# -------------------------------------------------------------------
# Lorenz attractor
# -------------------------------------------------------------------

def generate_lorenz(
    n_steps: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: float = 1.0,
    y0: float = 1.0,
    z0: float = 1.0,
) -> pd.DataFrame:
    """
    Generate Lorenz system trajectory via Euler integration.
    """
    xs = np.zeros(n_steps, dtype=float)
    ys = np.zeros(n_steps, dtype=float)
    zs = np.zeros(n_steps, dtype=float)

    xs[0], ys[0], zs[0] = x0, y0, z0

    for t in range(1, n_steps):
        x, y, z = xs[t - 1], ys[t - 1], zs[t - 1]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        xs[t] = x + dx * dt
        ys[t] = y + dy * dt
        zs[t] = z + dz * dt

    df = pd.DataFrame({"x": xs, "y": ys, "z": zs})
    df.index.name = "step"
    return df
