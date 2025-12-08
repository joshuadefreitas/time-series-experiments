import pandas as pd

from src.evaluation import rolling_forecast_origin


def test_rolling_forecast_with_datetime_index():
    # simple increasing series with DatetimeIndex
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series(range(10), index=dates, dtype=float)

    # forecast function: always predict last observed + 1
    def plus_one_forecast(s, h):
        return float(s.iloc[-1] + 1)

    results = rolling_forecast_origin(
        series,
        forecast_func=plus_one_forecast,
        horizon=1,
        initial_window=3,
    )

    # All forecasts should be off by exactly +1
    assert results["y_pred"].equals(results["y_true"] + 1)

    # t_forecast should carry the datetime labels
    assert pd.api.types.is_datetime64_any_dtype(results["t_forecast"])

    # t_train_end should also be datetime and precede t_forecast
    assert (results["t_train_end"] < results["t_forecast"]).all()
