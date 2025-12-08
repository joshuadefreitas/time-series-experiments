"""
ARIMA and SARIMA model estimation and forecasting.

Implements ARIMA(p,d,q) and SARIMA(p,d,q)(P,D,Q)_s models using maximum
likelihood estimation via statsmodels following Box-Jenkins methodology.
"""

from typing import Tuple
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ARIMAResults:
    """
    Estimate ARIMA(p,d,q) or SARIMA(p,d,q)(P,D,Q)_s via MLE.
    """
    if series.empty:
        raise ValueError("Cannot fit ARIMA model: input series is empty")
    if len(series) < 2:
        raise ValueError(
            f"Insufficient observations: {len(series)} provided, ARIMA requires at least 2"
        )

    # Suppress only convergence warnings (not all warnings!)
    with warnings.catch_warnings():
    # suppress only known harmless SARIMA warnings
        warnings.filterwarnings(
            "ignore",
            message="Too few observations to estimate starting parameters for seasonal ARMA",
            category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
        message="Non-invertible starting seasonal moving average",
        category=UserWarning
    )
        
    # DO NOT suppress convergence warnings
    model = ARIMA(series, order=order, seasonal_order=seasonal_order)
    result = model.fit()


    # Surface non-convergence explicitly
    if hasattr(result, "mle_retvals"):
        if not result.mle_retvals.get("converged", True):
            warnings.warn(
                "ARIMA fit did not converge — forecasts may be unreliable.",
                RuntimeWarning
            )

    return result


def arima_forecast(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> float:
    """
    Generate out-of-sample forecast h steps ahead using ARIMA/SARIMA.
    """
    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"Forecast horizon must be positive, got {horizon}")
    if not isinstance(horizon, int):
        raise TypeError(f"Forecast horizon must be integer, got {type(horizon).__name__}")

    result = fit_arima(series, order=order, seasonal_order=seasonal_order)
    forecast = result.forecast(steps=horizon)
    return float(forecast.iloc[-1])


def sarima_forecast(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
) -> float:
    """
    Convenience wrapper for SARIMA with default seasonal parameters.
    """
    return arima_forecast(
        series=series,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
    )
