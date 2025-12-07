from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def rolling_forecast_origin(
    series: pd.Series,
    forecast_func: Callable[[pd.Series, int], float],
    horizon: int = 1,
    initial_window: int = 100,
    max_forecasts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling-origin evaluation.

    - Use the first `initial_window` points as initial training.
    - For each t from initial_window to len(series)-horizon-1:
        - Train on series[:t]
        - Forecast t + horizon
        - Store y_true and y_pred.

    If `max_forecasts` is provided, only the last `max_forecasts` forecast
    points are evaluated (useful for expensive models like SARIMA).

    Returns DataFrame with:
        ["t_train_end", "t_forecast", "y_true", "y_pred"]
    """
    if len(series) <= initial_window + horizon:
        raise ValueError("Series too short for given initial_window and horizon.")

    index = series.index
    t_start = initial_window
    t_end_excl = len(series) - horizon

    all_t = list(range(t_start, t_end_excl))

    if max_forecasts is not None and max_forecasts < len(all_t):
        t_values: Iterable[int] = all_t[-max_forecasts:]
    else:
        t_values = all_t

    records = []

    for t in t_values:
        train_series = series.iloc[:t]
        y_pred = forecast_func(train_series, horizon)
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

    return pd.DataFrame(records)


def compute_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE from rolling forecast results.
    Expects columns ["y_true", "y_pred"].
    """
    y_true = results["y_true"].to_numpy()
    y_pred = results["y_pred"].to_numpy()
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE (ignore zero true values)
    nonzero_idx = np.where(y_true != 0)[0]
    if len(nonzero_idx) > 0:
        mape = float(
            np.mean(np.abs(errors[nonzero_idx] / y_true[nonzero_idx])) * 100.0
        )
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}