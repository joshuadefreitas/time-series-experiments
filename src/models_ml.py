"""
Machine-learning forecasting models based on lag features.

Models:
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
# Helper: build lag matrix with named columns
# ---------------------------------------------------------------------------

def _build_lag_matrix(
    series: pd.Series,
    n_lags: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build a lag feature matrix X (as DataFrame with column names)
    and a target vector y.
    """
    y = series.values.astype(float)
    T = len(y)

    if T <= n_lags:
        raise ValueError(
            f"Series too short for n_lags={n_lags}: length={T}"
        )

    rows = []
    target = []

    for t in range(n_lags, T):
        rows.append(y[t - n_lags:t])
        target.append(y[t])

    X = pd.DataFrame(rows, columns=[f"lag_{i+1}" for i in range(n_lags)])
    return X, np.asarray(target)


def _iterative_forecast(
    model,
    y: np.ndarray,
    horizon: int,
    n_lags: int,
) -> float:
    """
    Iterative multi-step forecasting with consistent feature naming.
    """
    last_window = y[-n_lags:].copy()

    for _ in range(horizon):
        X_next = pd.DataFrame([last_window], columns=[f"lag_{i+1}" for i in range(n_lags)])
        pred = model.predict(X_next)[0]

        # roll + append
        last_window = np.roll(last_window, -1)
        last_window[-1] = pred

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
    Random Forest regression on lag features.
    """
    if not _HAS_SKLEARN_RF:
        raise ImportError(
            "scikit-learn is required for rf_forecast. Install with `pip install scikit-learn`."
        )

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, target)

    return _iterative_forecast(
        model,
        y=series.values.astype(float),
        horizon=horizon,
        n_lags=n_lags,
    )


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def xgb_forecast(
    series: pd.Series,
    horizon: int = 1,
    n_lags: int = 20,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> float:
    """
    XGBoost regression on lag features.
    """
    if not _HAS_XGBOOST:
        raise ImportError(
            "xgboost is required for xgb_forecast. Install with `pip install xgboost`."
        )

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X, target)

    return _iterative_forecast(
        model,
        y=series.values.astype(float),
        horizon=horizon,
        n_lags=n_lags,
    )


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
    LightGBM regression on lag features.
    """
    if not _HAS_LIGHTGBM:
        raise ImportError(
            "lightgbm is required for lgbm_forecast. Install with `pip install lightgbm`."
        )

    X, target = _build_lag_matrix(series, n_lags=n_lags)

    model = LGBMRegressor(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        min_child_samples=10,
        verbose=-1,
    )
    model.fit(X, target)

    return _iterative_forecast(
        model,
        y=series.values.astype(float),
        horizon=horizon,
        n_lags=n_lags,
    )
