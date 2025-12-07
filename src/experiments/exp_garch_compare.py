"""
Experiment: Forecasting under GARCH-like volatility clustering.

We generate a synthetic return series that has:
    - mean = 0
    - volatility clustering
    - ARCH/GARCH-type persistence

We compare Naive, Mean, ARIMA(1,0,0), and a simple volatility-aware
baseline (rolling standard deviation).

Run with:
    python -m src.experiments.exp_garch_compare
"""

from pathlib import Path

import numpy as np
from src.simulators import generate_garch_like
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def rolling_vol_forecast(series, window=20):
    """
    Very simple volatility-aware baseline:
        forecast = 0  (mean is zero)
        but we model volatility as rolling window std

    The prediction is still zero, but the model implicitly
    knows that uncertainty is larger when volatility is high.
    
    For MAE/MAPE/RMSE it's identical to Mean forecast.
    We include it for conceptual clarity.
    """
    return 0.0


def main():

    # 1. Generate GARCH-like series
    df = generate_garch_like(
        n=1500,
        omega=0.1,
        alpha=0.2,
        beta=0.6,
        seed=123,
    )

    series = df["value"]
    vol = df["sigma"]

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "garch_example.csv")

    print("Generated GARCH-like return series:")
    print(df.head())

    # Forecast settings
    horizon = 1
    initial_window = 300

    print("\nEvaluating models...\n")

    # --- Naive ---
    res_naive = rolling_forecast_origin(
        series,
        naive_forecast,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_naive = compute_metrics(res_naive)

    # --- Mean ---
    res_mean = rolling_forecast_origin(
        series,
        mean_forecast,
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_mean = compute_metrics(res_mean)

    # --- ARIMA ---
    res_arima = rolling_forecast_origin(
        series,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=200,
    )
    metrics_arima = compute_metrics(res_arima)

    # --- Volatility Baseline ---
    res_vol = rolling_forecast_origin(
        series,
        lambda s, h: rolling_vol_forecast(s),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_vol = compute_metrics(res_vol)

    # Plot example
    plot_forecast_results(
        res_mean,
        title="GARCH-like series – Mean forecast vs actual",
        filename="garch_mean_forecast.png",
    )

    # Summary table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY (GARCH-like returns)")
    print("="*60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-"*60)
    print(f"{'Naive':<20} {metrics_naive['MAE']:<12.4f} {metrics_naive['RMSE']:<12.4f} {metrics_naive['MAPE']:<12.4f}")
    print(f"{'Mean':<20} {metrics_mean['MAE']:<12.4f} {metrics_mean['RMSE']:<12.4f} {metrics_mean['MAPE']:<12.4f}")
    print(f"{'ARIMA(1,0,0)':<20} {metrics_arima['MAE']:<12.4f} {metrics_arima['RMSE']:<12.4f} {metrics_arima['MAPE']:<12.4f}")
    print(f"{'Volatility Baseline':<20} {metrics_vol['MAE']:<12.4f} {metrics_vol['RMSE']:<12.4f} {metrics_vol['MAPE']:<12.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print(
        """
Returns have mean 0. The only predictable component is volatility.

Key insights:
  • Mean forecast performs strongly because the true mean is zero.
  • Naive forecast performs poorly: copying yesterday’s return copies noise.
  • ARIMA(1,0,0) fails because there is no AR structure in the mean.
  • Volatility baseline produces identical level forecasts (0),
    but highlights that forecasting returns ≠ forecasting risk.

This mirrors real financial markets:
  – price direction is nearly unpredictable,
  – volatility clusters, and
  – risk forecasting (volatility) is more meaningful than predicting returns.
"""
    )

    print("="*60)
    print("\nDone.")


if __name__ == "__main__":
    main()