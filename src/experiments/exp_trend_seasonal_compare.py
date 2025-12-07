"""
Experiment: compare naive, mean, seasonal naive, ARIMA, SARIMA, and ETS
on a trend + seasonal synthetic time series.

Run from project root with:
    python -m src.experiments.exp_trend_seasonal_compare
"""

from pathlib import Path

from src.simulators import generate_trend_seasonal
from src.models_basic import (
    naive_forecast,
    mean_forecast,
    seasonal_naive_forecast,
)
from src.models_arima import arima_forecast, sarima_forecast
from src.models_advanced import ets_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate synthetic trend + seasonal data
    n = 500
    period = 24  # "season" length (e.g. daily pattern in hourly data, etc.)
    series = generate_trend_seasonal(
        n=n,
        trend_slope=0.02,
        period=period,
        amplitude=1.0,
        sigma=0.4,
        seed=123,
    )

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(data_dir / "trend_seasonal_example.csv", index_label="t")
    print(f"Generated trend+seasonal series of length {len(series)}")

    # evaluation settings
    horizon = 1
    # we want at least a few seasons in the initial training window
    initial_window = period * 4  # 4 full seasons

    # 2. Naive forecast (random walk)
    res_naive = rolling_forecast_origin(
        series,
        naive_forecast,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_naive = compute_metrics(res_naive)
    print("\nNaive forecast metrics:")
    for k, v in metrics_naive.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_naive,
        title="Trend+Seasonal – Naive forecast vs actual",
        filename="trend_seasonal_naive_vs_actual.png",
    )

    # 3. Mean forecast
    res_mean = rolling_forecast_origin(
        series,
        mean_forecast,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_mean = compute_metrics(res_mean)
    print("\nMean forecast metrics:")
    for k, v in metrics_mean.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_mean,
        title="Trend+Seasonal – Mean forecast vs actual",
        filename="trend_seasonal_mean_vs_actual.png",
    )

    # 4. Seasonal naive forecast
    seasonal_forecaster = lambda s, h: seasonal_naive_forecast(
        s,
        season_length=period,
        horizon=h,
    )
    res_seasonal = rolling_forecast_origin(
        series,
        seasonal_forecaster,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_seasonal = compute_metrics(res_seasonal)
    print("\nSeasonal naive forecast metrics:")
    for k, v in metrics_seasonal.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_seasonal,
        title="Trend+Seasonal – Seasonal naive forecast vs actual",
        filename="trend_seasonal_seasonal_naive_vs_actual.png",
    )

    # 5. ARIMA (non-seasonal) – intentionally ignoring seasonality
    res_arima = rolling_forecast_origin(
        series,
        lambda s, h: arima_forecast(s, h, order=(2, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_arima = compute_metrics(res_arima)
    print("\nARIMA(2,0,0) forecast metrics (no explicit seasonality):")
    for k, v in metrics_arima.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_arima,
        title="Trend+Seasonal – ARIMA(2,0,0) forecast vs actual",
        filename="trend_seasonal_arima_vs_actual.png",
    )

    # 6. SARIMA – explicitly modeling seasonality
    res_sarima = rolling_forecast_origin(
        series,
        lambda s, h: sarima_forecast(
            s,
            h,
            order=(1, 0, 0),
            seasonal_order=(0, 1, 1, period),
        ),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_sarima = compute_metrics(res_sarima)
    print(f"\nSARIMA(1,0,0)x(0,1,1,{period}) forecast metrics:")
    for k, v in metrics_sarima.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_sarima,
        title="Trend+Seasonal – SARIMA forecast vs actual",
        filename="trend_seasonal_sarima_vs_actual.png",
    )

    # 7. ETS / Holt–Winters exponential smoothing
    res_ets = rolling_forecast_origin(
        series,
        lambda s, h: ets_forecast(
            s,
            horizon=h,
            trend="add",
            seasonal="add",
            seasonal_periods=period,
        ),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_ets = compute_metrics(res_ets)
    print("\nETS (Holt-Winters) forecast metrics:")
    for k, v in metrics_ets.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_ets,
        title="Trend+Seasonal – ETS (Holt-Winters) forecast vs actual",
        filename="trend_seasonal_ets_vs_actual.png",
    )

    # 8. Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(
        f"{'Naive':<25} {metrics_naive['MAE']:<12.4f} "
        f"{metrics_naive['RMSE']:<12.4f} {metrics_naive['MAPE']:<12.4f}"
    )
    print(
        f"{'Mean':<25} {metrics_mean['MAE']:<12.4f} "
        f"{metrics_mean['RMSE']:<12.4f} {metrics_mean['MAPE']:<12.4f}"
    )
    print(
        f"{'Seasonal Naive':<25} {metrics_seasonal['MAE']:<12.4f} "
        f"{metrics_seasonal['RMSE']:<12.4f} {metrics_seasonal['MAPE']:<12.4f}"
    )
    print(
        f"{'ARIMA(2,0,0)':<25} {metrics_arima['MAE']:<12.4f} "
        f"{metrics_arima['RMSE']:<12.4f} {metrics_arima['MAPE']:<12.4f}"
    )
    print(
        f"{'SARIMA':<25} {metrics_sarima['MAE']:<12.4f} "
        f"{metrics_sarima['RMSE']:<12.4f} {metrics_sarima['MAPE']:<12.4f}"
    )
    print(
        f"{'ETS (HW)':<25} {metrics_ets['MAE']:<12.4f} "
        f"{metrics_ets['RMSE']:<12.4f} {metrics_ets['MAPE']:<12.4f}"
    )

    # Best model by metric
    print("\n" + "-" * 60)
    print("BEST MODEL BY METRIC:")
    print("-" * 60)

    models = {
        "Naive": metrics_naive,
        "Mean": metrics_mean,
        "Seasonal Naive": metrics_seasonal,
        "ARIMA(2,0,0)": metrics_arima,
        "SARIMA": metrics_sarima,
        "ETS (HW)": metrics_ets,
    }

    best_mae = min(models.items(), key=lambda x: x[1]["MAE"])
    best_rmse = min(models.items(), key=lambda x: x[1]["RMSE"])
    best_mape = min(models.items(), key=lambda x: x[1]["MAPE"])

    print(f"  MAE:  {best_mae[0]} ({best_mae[1]['MAE']:.4f})")
    print(f"  RMSE: {best_rmse[0]} ({best_rmse[1]['RMSE']:.4f})")
    print(f"  MAPE: {best_mape[0]} ({best_mape[1]['MAPE']:.4f})")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(
        """
  • Naive: follows the last observed value; struggles with trend/seasonality.
  • Mean: ignores time structure; best when the series has no trend/seasonality.
  • Seasonal Naive: strong baseline when seasonal pattern dominates.
  • ARIMA(2,0,0): can capture some dynamics, but no explicit seasonality.
  • SARIMA: explicitly models seasonal structure; often strong here.
  • ETS (Holt-Winters): another classical way to model level + trend + seasonality.

  For this synthetic trend+seasonal series, SARIMA and ETS should usually
  outperform naive and mean, with seasonal naive often quite competitive.
        """
    )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()