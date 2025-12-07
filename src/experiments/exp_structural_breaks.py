"""
Experiment: Structural break in the mean (single changepoint).

We generate an AR(1)-type series with a single mean shift and compare:
    - Global Mean forecast
    - Rolling Mean forecast
    - Global ARIMA(1,0,0)
    - Rolling ARIMA(1,0,0)

The goal is to see how long-memory models struggle after a break, and
how rolling / local models adapt.
"""

from pathlib import Path

import pandas as pd

from src.simulators import generate_structural_break_series
from src.models_basic import mean_forecast
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def rolling_mean_forecast(series: pd.Series, window: int, horizon: int) -> float:
    """Forecast = mean of the last `window` points."""
    if len(series) < window:
        window_series = series
    else:
        window_series = series.iloc[-window:]
    return float(window_series.mean())


def rolling_arima_forecast(
    series: pd.Series,
    window: int,
    horizon: int,
    order=(1, 0, 0),
) -> float:
    """Fit ARIMA on last `window` points only and forecast."""
    from src.models_arima import arima_forecast as _arima_forecast

    if len(series) < window:
        window_series = series
    else:
        window_series = series.iloc[-window:]
    return _arima_forecast(window_series, horizon, order=order)


def compute_metrics_post_break(
    results: pd.DataFrame,
    break_point: int,
) -> dict:
    """
    Compute metrics only for forecasts whose target time is after the break.
    Assumes t_forecast is an integer index (RangeIndex).
    """
    mask = results["t_forecast"] >= break_point
    if not mask.any():
        return {}
    subset = results.loc[mask]
    return compute_metrics(subset)


def main():
    # 1. Generate series with a clear structural break
    n = 800
    break_point = 400

    df = generate_structural_break_series(
        n=n,
        break_point=break_point,
        mu1=0.0,
        mu2=3.0,
        phi=0.6,
        sigma=1.0,
        seed=42,
    )

    series = df["value"]
    regimes = df["regime"]

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "structural_break_example.csv")

    print(f"Generated structural-break series of length {len(series)}")
    print("Regime counts:", regimes.value_counts().to_dict())

    # evaluation settings
    horizon = 1
    initial_window = 200
    rolling_window = 100  # for rolling models

    # --- Global Mean ---
    res_global_mean = rolling_forecast_origin(
        series,
        mean_forecast,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_global_mean = compute_metrics(res_global_mean)
    metrics_global_mean_post = compute_metrics_post_break(
        res_global_mean, break_point
    )

    plot_forecast_results(
        res_global_mean,
        title="Structural break – Global Mean vs actual",
        filename="structural_break_global_mean.png",
    )

    # --- Rolling Mean ---
    res_rolling_mean = rolling_forecast_origin(
        series,
        lambda s, h: rolling_mean_forecast(s, window=rolling_window, horizon=h),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_rolling_mean = compute_metrics(res_rolling_mean)
    metrics_rolling_mean_post = compute_metrics_post_break(
        res_rolling_mean, break_point
    )

    plot_forecast_results(
        res_rolling_mean,
        title="Structural break – Rolling Mean vs actual",
        filename="structural_break_rolling_mean.png",
    )

    # --- Global ARIMA(1,0,0) ---
    res_global_arima = rolling_forecast_origin(
        series,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=200,  # for speed
    )
    metrics_global_arima = compute_metrics(res_global_arima)
    metrics_global_arima_post = compute_metrics_post_break(
        res_global_arima, break_point
    )

    plot_forecast_results(
        res_global_arima,
        title="Structural break – Global ARIMA(1,0,0) vs actual",
        filename="structural_break_global_arima.png",
    )

    # --- Rolling ARIMA(1,0,0) ---
    res_rolling_arima = rolling_forecast_origin(
        series,
        lambda s, h: rolling_arima_forecast(
            s, window=rolling_window, horizon=h, order=(1, 0, 0)
        ),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=200,
    )
    metrics_rolling_arima = compute_metrics(res_rolling_arima)
    metrics_rolling_arima_post = compute_metrics_post_break(
        res_rolling_arima, break_point
    )

    plot_forecast_results(
        res_rolling_arima,
        title="Structural break – Rolling ARIMA(1,0,0) vs actual",
        filename="structural_break_rolling_arima.png",
    )

    # 2. Summary tables
    print("\n" + "=" * 60)
    print("STRUCTURAL BREAK – OVERALL METRICS")
    print("=" * 60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(
        f"{'Global Mean':<20} {metrics_global_mean['MAE']:<12.4f} "
        f"{metrics_global_mean['RMSE']:<12.4f} {metrics_global_mean['MAPE']:<12.4f}"
    )
    print(
        f"{'Rolling Mean':<20} {metrics_rolling_mean['MAE']:<12.4f} "
        f"{metrics_rolling_mean['RMSE']:<12.4f} {metrics_rolling_mean['MAPE']:<12.4f}"
    )
    print(
        f"{'Global ARIMA':<20} {metrics_global_arima['MAE']:<12.4f} "
        f"{metrics_global_arima['RMSE']:<12.4f} {metrics_global_arima['MAPE']:<12.4f}"
    )
    print(
        f"{'Rolling ARIMA':<20} {metrics_rolling_arima['MAE']:<12.4f} "
        f"{metrics_rolling_arima['RMSE']:<12.4f} {metrics_rolling_arima['MAPE']:<12.4f}"
    )

    print("\n" + "=" * 60)
    print("STRUCTURAL BREAK – POST-BREAK METRICS ONLY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)

    def fmt_post(label, m):
        if not m:
            print(f"{label:<20} {'n/a':<12} {'n/a':<12} {'n/a':<12}")
        else:
            print(
                f"{label:<20} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['MAPE']:<12.4f}"
            )

    fmt_post("Global Mean", metrics_global_mean_post)
    fmt_post("Rolling Mean", metrics_rolling_mean_post)
    fmt_post("Global ARIMA", metrics_global_arima_post)
    fmt_post("Rolling ARIMA", metrics_rolling_arima_post)

    # 3. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(
        f"""
We simulated an AR(1)-type series with a single structural break in the mean
at t = {break_point} (from mu1 to mu2).

Key ideas:
  • Global models (Global Mean, Global ARIMA) use all past data.
    After the break, they are pulled toward the old regime and adapt slowly.

  • Rolling models (Rolling Mean, Rolling ARIMA) rely more on recent data.
    They typically perform better post-break because they "forget" outdated information.

  • The post-break metrics are the most informative:
    they show how quickly each model can adapt once the underlying process changes.

In practice, when you suspect structural change, shorter windows or
adaptive models are often safer than one global model fitted on all history.
"""
    )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error running experiment: {e}")
        traceback.print_exc()
        raise