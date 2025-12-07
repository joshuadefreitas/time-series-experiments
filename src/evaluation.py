"""
Forecast evaluation and cross-validation utilities.

Implements rolling-origin evaluation (also known as time series cross-validation
or walk-forward analysis) for assessing forecast accuracy. Provides standard
error metrics used in forecast evaluation and model comparison.

References:
    Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy:
    an analysis and review. International Journal of Forecasting, 16(4), 437-450.
"""

import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd


def rolling_forecast_origin(
    series: pd.Series,
    forecast_func: Callable[[pd.Series, int], float],
    horizon: int = 1,
    initial_window: int = 100,
) -> pd.DataFrame:
    """
    Perform rolling-origin (time series cross-validation) evaluation.
    
    Implements a walk-forward evaluation where the model is re-estimated at
    each time step using expanding windows. This approach respects temporal
    ordering and provides a realistic assessment of out-of-sample performance.
    
    Procedure:
        1. Reserve first `initial_window` observations as initial training set
        2. For each t ∈ [initial_window, T-h]:
           - Estimate model on series[0:t]
           - Generate h-step ahead forecast: ŷ_{t+h}
           - Compare with actual value y_{t+h}
    
    Args:
        series: Full time series (training + test data).
        forecast_func: Forecast function with signature f(series, horizon) -> float.
            Must accept training series and forecast horizon, return point forecast.
        horizon: Forecast horizon h (number of steps ahead to forecast).
        initial_window: Minimum training set size for initial forecast.
    
    Returns:
        DataFrame with columns:
            - t_train_end: Index of last observation used for training
            - t_forecast: Index of forecasted observation
            - y_true: Actual observed value
            - y_pred: Forecasted value
    
    Raises:
        ValueError: If series length is insufficient for specified parameters,
            or if horizon or initial_window are invalid.
    
    Example:
        >>> results = rolling_forecast_origin(
        ...     series=data,
        ...     forecast_func=arima_forecast,
        ...     horizon=1,
        ...     initial_window=100
        ... )
        >>> metrics = compute_metrics(results)
    
    Note:
        If forecast_func raises an exception for a particular time step, a warning
        is issued and that time step is skipped. The function continues with
        subsequent time steps. This allows evaluation to proceed even if some
        forecasts fail (e.g., due to model convergence issues).
    """
    if series.empty:
        raise ValueError("Cannot evaluate: input series is empty")
    if horizon <= 0:
        raise ValueError(f"Forecast horizon must be positive, got {horizon}")
    if not isinstance(horizon, int):
        raise TypeError(f"Forecast horizon must be integer, got {type(horizon).__name__}")
    if initial_window <= 0:
        raise ValueError(f"Initial window must be positive, got {initial_window}")
    if not isinstance(initial_window, int):
        raise TypeError(f"Initial window must be integer, got {type(initial_window).__name__}")
    if len(series) <= initial_window + horizon:
        raise ValueError(
            f"Series too short for given initial_window ({initial_window}) and "
            f"horizon ({horizon}). Need at least {initial_window + horizon + 1} "
            f"observations, got {len(series)}"
        )

    records = []
    index = series.index
    failed_forecasts = 0

    # Iterate through possible forecast origins
    for t in range(initial_window, len(series) - horizon):
        # Training set: all observations up to (but not including) time t
        train_series = series.iloc[:t]
        
        try:
            # Generate h-step ahead forecast from origin t
            y_pred = forecast_func(train_series, horizon)
            
            # Actual value at forecast target time
            t_forecast = index[t + horizon]
            y_true = series.iloc[t + horizon]

            records.append(
                {
                    "t_train_end": index[t - 1],
                    "t_forecast": t_forecast,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )
        except Exception as e:
            # Log warning and continue with next time step
            failed_forecasts += 1
            warnings.warn(
                f"Forecast failed at time step t={t} (training end={index[t-1]}): {e}. "
                f"Skipping this forecast.",
                UserWarning
            )
            continue

    if failed_forecasts > 0:
        warnings.warn(
            f"Total failed forecasts: {failed_forecasts} out of "
            f"{len(series) - initial_window - horizon} attempted. "
            f"Results contain {len(records)} successful forecasts.",
            UserWarning
        )
    
    if len(records) == 0:
        raise RuntimeError(
            "All forecasts failed. Check forecast_func implementation and "
            "ensure it can handle the provided series."
        )

    return pd.DataFrame(records)


def compute_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute standard forecast accuracy metrics from evaluation results.
    
    Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    and Mean Absolute Percentage Error (MAPE). These metrics measure different
    aspects of forecast accuracy: MAE is scale-dependent and robust to outliers,
    RMSE penalizes large errors more heavily, and MAPE provides scale-invariant
    relative error measures.
    
    Args:
        results: DataFrame with columns ["y_true", "y_pred"] containing
            actual and forecasted values from rolling forecast evaluation.
    
    Returns:
        Dictionary with keys:
            - MAE: Mean Absolute Error = (1/n)Σ|y_i - ŷ_i|
            - RMSE: Root Mean Squared Error = √[(1/n)Σ(y_i - ŷ_i)²]
            - MAPE: Mean Absolute Percentage Error = (100/n)Σ|(y_i - ŷ_i)/y_i|
              (computed only for non-zero actual values, NaN if all zeros)
    
    Raises:
        ValueError: If results DataFrame is empty or missing required columns.
    
    Note:
        MAPE is undefined for zero actual values. Implementation excludes
        such observations from MAPE calculation and returns NaN if no valid
        observations remain.
    """
    if results.empty:
        raise ValueError("Cannot compute metrics: results DataFrame is empty")
    
    required_columns = ["y_true", "y_pred"]
    missing_columns = [col for col in required_columns if col not in results.columns]
    if missing_columns:
        raise ValueError(
            f"Results DataFrame missing required columns: {missing_columns}. "
            f"Found columns: {list(results.columns)}"
        )
    
    y_true = results["y_true"].to_numpy()
    y_pred = results["y_pred"].to_numpy()
    errors = y_pred - y_true

    # Mean Absolute Error: average of absolute forecast errors
    mae = float(np.mean(np.abs(errors)))
    
    # Root Mean Squared Error: square root of mean squared errors
    # More sensitive to large forecast errors than MAE
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Mean Absolute Percentage Error: scale-invariant relative error metric
    # Only defined for non-zero actual values
    nonzero_idx = np.where(y_true != 0)[0]
    if len(nonzero_idx) > 0:
        mape = float(
            np.mean(np.abs(errors[nonzero_idx] / y_true[nonzero_idx])) * 100.0
        )
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}