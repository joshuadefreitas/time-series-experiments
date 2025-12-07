"""
Advanced forecasting models.

Currently:
- Exponential Smoothing / Holt-Winters (ETS)
- Lag-based linear regression baseline (scikit-learn)
"""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional import: we keep the project usable even without sklearn
try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def ets_forecast(
    series: pd.Series,
    horizon: int = 1,
    trend: Optional[str] = "add",          # "add", "mul", or None
    seasonal: Optional[str] = "add",       # "add", "mul", or None
    seasonal_periods: Optional[int] = None,
) -> float:
    """
    Holt–Winters exponential smoothing forecast.
    """
    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    if seasonal is not None and seasonal_periods is None:
        raise ValueError(
            "seasonal_periods must be provided when a seasonal component is used"
        )

    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(steps=horizon)

    return float(forecast.iloc[-1])


def lag_regression_forecast(
    series: pd.Series,
    horizon: int = 1,
    n_lags: int = 20,
) -> float:
    """
    Simple lag-based linear regression model using scikit-learn.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for lag_regression_forecast. "
            "Install it with `pip install scikit-learn`."
        )

    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    y = series.values.astype(float)
    T = len(y)

    if T <= n_lags:
        raise ValueError(
            f"Series too short for n_lags={n_lags}: length={T}"
        )

    # Build design matrix for one-step-ahead regression
    X = []
    target = []
    for t in range(n_lags, T):
        X.append(y[t - n_lags:t])  # last n_lags values as features
        target.append(y[t])

    X = np.asarray(X)
    target = np.asarray(target)

    model = LinearRegression()
    model.fit(X, target)

    # Iterative multi-step forecasting
    last_window = y[-n_lags:].copy()
    for _ in range(horizon):
        next_val = model.predict(last_window.reshape(1, -1))[0]
        # roll: drop oldest, append new
        last_window = np.roll(last_window, -1)
        last_window[-1] = next_val

    return float(last_window[-1])