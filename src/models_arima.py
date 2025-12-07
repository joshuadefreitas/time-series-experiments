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


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ARIMAResults:
    """
    Estimate ARIMA(p,d,q) or SARIMA(p,d,q)(P,D,Q)_s model parameters via MLE.
    
    Model specification: φ(B)(1-B)^d y_t = θ(B)ε_t where ε_t ~ N(0, σ²).
    
    Args:
        series: Univariate time series (should be stationary after differencing).
        order: Tuple (p, d, q) for non-seasonal component.
            - p: AR order
            - d: Integration order (differences for stationarity)
            - q: MA order
        seasonal_order: Tuple (P, D, Q, s) for seasonal component.
            - P: Seasonal AR order
            - D: Seasonal differencing order
            - Q: Seasonal MA order
            - s: Seasonal period (e.g., 12 for monthly)
    
    Returns:
        ARIMAResults object with estimated parameters, log-likelihood, AIC/BIC,
        standard errors, and methods for forecasting and diagnostics.
    
    Raises:
        ValueError: If series is empty or has insufficient observations.
    """
    if series.empty:
        raise ValueError("Cannot fit ARIMA model: input series is empty")
    if len(series) < 2:
        raise ValueError(
            f"Insufficient observations: {len(series)} provided, "
            "ARIMA requires at least 2"
        )
    
    # Suppress convergence warnings during MLE optimization
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        model = ARIMA(series, order=order, seasonal_order=seasonal_order)
        result = model.fit()
    
    return result


def arima_forecast(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> float:
    """
    Generate out-of-sample point forecast h steps ahead using ARIMA/SARIMA.
    
    Estimates model on in-sample data and returns conditional expectation
    E[y_{T+h} | y_1, ..., y_T] where T = len(series) and h = horizon.
    
    Args:
        series: In-sample time series for model estimation.
        horizon: Forecast horizon (number of steps ahead). Must be positive.
        order: ARIMA order tuple (p, d, q). See fit_arima() documentation.
        seasonal_order: SARIMA seasonal order tuple (P, D, Q, s).
    
    Returns:
        Point forecast at horizon h as float.
    
    Raises:
        ValueError: If series is empty, insufficient length, or horizon <= 0.
    
    Note:
        For prediction intervals, use fitted_model.get_forecast().conf_int().
        Long-horizon forecasts converge to unconditional mean for stationary models.
    """
    if series.empty:
        raise ValueError("Cannot forecast: input series is empty")
    if horizon <= 0:
        raise ValueError(f"Forecast horizon must be positive, got {horizon}")
    if not isinstance(horizon, int):
        raise TypeError(f"Forecast horizon must be integer, got {type(horizon).__name__}")
    
    # Estimate model parameters on in-sample data
    result = fit_arima(series, order=order, seasonal_order=seasonal_order)
    
    # Generate recursive h-step ahead forecast path
    forecast = result.forecast(steps=horizon)
    
    # Return point forecast at final horizon step
    return float(forecast.iloc[-1])


def sarima_forecast(
    series: pd.Series,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
) -> float:
    """
    Convenience wrapper for seasonal ARIMA (SARIMA).

    This is identical to arima_forecast(), but with a more natural
    default seasonal_order and a name that makes intent explicit.

    Args:
        series: In-sample time series for model estimation.
        horizon: Forecast horizon (steps ahead).
        order: Non-seasonal ARIMA order (p, d, q).
        seasonal_order: Seasonal order (P, D, Q, s), where:
            - P: seasonal AR order
            - D: seasonal differencing order
            - Q: seasonal MA order
            - s: seasonal period (e.g. 12 for monthly, 24 for hourly, etc.)

    Returns:
        Point forecast at horizon h as float.
    """
    return arima_forecast(
        series=series,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
    )