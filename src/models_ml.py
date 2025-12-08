"""
Machine-learning forecasting models based on lag features.

Currently:
- RandomForestRegressor (scikit-learn)
- XGBRegressor (xgboost)
- LGBMRegressor (lightgbm)

All models follow the same pattern:
    - build lag features from a univariate time series
    - fit on (lags -> next value)
    - iterate forecasts up to the requested horizon
"""

from typing import Tuple

import numpy as np
import pandas as pd

# Optional imports: keep project usable if libs are missing
try:
    from sklearn.ensemble import RandomForestRegressor
    _HAS_SKLEARN_RF = True
except ImportError:
    _HAS_SKLEARN_RF = False

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


# ---------------------------------------------------------------------------
# Helper: build lag matrix
# ---------------------------------------------------------------------------

def _build_lag_matrix(
    series: pd.Series,
    n_lags: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build design matrix X and target y for lag-based forecasting.

    X_t = [y_{t-1}, y_{t-2}, ..., y_{t-n_lags}]
    y_t = y_t

    We start at t = n_lags, so we use only rows with full lag history.
    """
    y = series.values.astype(float)
    T = len(y)

    if T <= n_lags:
        raise ValueError(
            f"Series too short for n_lags={n_lags}: length={T}"
        )

    X = []
    target = []
    for t in range(n_lags, T):
        X.append(y[t - n_lags:t])
        target.append(y[t])

    X = np.asarray(X)
    target = np.asarray(target)
    return X, target


def _iterative_forecast(
    model,
    y: np.ndarray,
    horizon: int,
    n_lags: int,
) -> float:
    """
    Given a fitted model and the original series y, perform
    iterative multi-step forecasting using the last n_lags values.

    Returns:
        Forecast for y_{T + horizon} as float.
    """
    last_window = y[-n_lags:].copy()
    for _ in range(horizon):
        next_val = model.predict(last_window.reshape(1, -1))[0]
        # roll: drop oldest, append new
        last_window = np.roll(last_window, -1)
        last_window[-1] = next_val

    return float(last_window[-1])


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def rf_forecast(
    series: pd.Series,
    horizon: int = 1,
    n_lags: int = 20,
    n_estimators: int = 200,
    max_depth: int = 6,
    random_state: int = 42,
) -> float:
    """
    Random Forest-based time series forecast using lag features.

    Args:
        series: Univariate time series.
        horizon: Forecast horizon (steps ahead).
        n_lags: Number of lags to use as predictors.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        random_state: Random seed for reproducibility.

    Returns:
        Forecast for y_{T + horizon} as float.
    """
    if not _HAS_SKLEARN_RF:
        raise ImportError(
            "scikit-learn is required for rf_forecast. "
            "Install it with `pip install scikit-learn`."
        )

    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, target)

    y = series.values.astype(float)
    return _iterative_forecast(model, y, horizon=horizon, n_lags=n_lags)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def xgb_forecast(
    series: pd.Series,
    horizon: int = 1,
    n_lags: int = 20,
    max_depth: int = 4,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> float:
    """
    XGBoost-based time series forecast using lag features.

    Args:
        series: Univariate time series.
        horizon: Forecast horizon (steps ahead).
        n_lags: Number of lags to use as predictors.
        max_depth: Maximum tree depth.
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate (shrinkage).
        subsample: Row subsampling.
        colsample_bytree: Column subsampling per tree.
        random_state: Random seed.

    Returns:
        Forecast for y_{T + horizon} as float.
    """
    if not _HAS_XGBOOST:
        raise ImportError(
            "xgboost is required for xgb_forecast. "
            "Install it with `pip install xgboost`."
        )

    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X, target)

    y = series.values.astype(float)
    return _iterative_forecast(model, y, horizon=horizon, n_lags=n_lags)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def lgbm_forecast(
    series: pd.Series,
    horizon: int = 1,
    n_lags: int = 20,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 300,
    random_state: int = 42,
) -> float:
    """
    LightGBM-based time series forecast using lag features.
    """
    if not _HAS_LIGHTGBM:
        raise ImportError(
            "lightgbm is required for lgbm_forecast. "
            "Install it with `pip install lightgbm`."
        )

    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = LGBMRegressor(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        min_child_samples=10,   # helps avoid weird tiny leaves
        verbose=-1,             # silence training logs
    )
    model.fit(X, target)

    y = series.values.astype(float)
    return _iterative_forecast(model, y, horizon=horizon, n_lags=n_lags)