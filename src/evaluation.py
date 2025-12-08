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
    Rolling-origin evaluation using purely positional indexing (iloc),
    so this works for RangeIndex, DateTimeIndex, PeriodIndex, etc.

    Returns DataFrame with:
        ["t_train_end", "t_forecast", "y_true", "y_pred"]
    """
    n = len(series)
    if n <= initial_window + horizon:
        raise ValueError("Series too short for given initial_window and horizon.")

    index = series.index  # may be RangeIndex or DateTimeIndex
    t_start = initial_window
    t_end_excl = n - horizon  # last train index = t_end_excl - 1

    all_t = list(range(t_start, t_end_excl))

    if max_forecasts is not None and max_forecasts < len(all_t):
        t_values = all_t[-max_forecasts:]
    else:
        t_values = all_t

    records = []

    for t in t_values:
        # training window = [0 : t)
        train_series = series.iloc[:t]

        # forecast t + horizon
        y_pred = forecast_func(train_series, horizon)

        # ensure scalar
        if isinstance(y_pred, (np.ndarray, list)):
            if len(np.ravel(y_pred)) != 1:
                raise ValueError("forecast_func must return a scalar.")
            y_pred = float(np.ravel(y_pred)[0])

        # positional lookups (robust to any index type)
        t_train_end = index[t - 1]          # timestamp of last training obs
        t_forecast = index[t + horizon]     # timestamp being forecasted
        y_true = float(series.iloc[t + horizon])

        records.append(
            {
                "t_train_end": t_train_end,
                "t_forecast": t_forecast,
                "y_true": y_true,
                "y_pred": float(y_pred),
            }
        )

    return pd.DataFrame(records)


def compute_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE from rolling forecast results.
    Expects columns ["y_true", "y_pred"].
    """
    y_true = results["y_true"].to_numpy(dtype=float)
    y_pred = results["y_pred"].to_numpy(dtype=float)
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: ignore y_true == 0
    mask = (y_true != 0)
    if mask.any():
        mape = float(np.mean(np.abs(errors[mask] / y_true[mask])) * 100.0)
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
