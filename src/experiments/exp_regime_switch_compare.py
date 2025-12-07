"""
Experiment: Forecasting under regime-switching volatility.

We generate a zero-mean process with low/high volatility regimes and compare
Naive, Mean, and ARIMA(1,0,0) forecasts. The main interest is to see how
standard models behave when the variance structure is unstable.

Run from project root with:
    python -m src.experiments.exp_regime_switch_compare
"""

from pathlib import Path

import pandas as pd

from src.simulators import generate_regime_switching_noise
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate regime-switching series
    n = 800
    df = generate_regime_switching_noise(
        n=n,
        sigma_low=0.5,
        sigma_high=2.0,
        p_up=0.02,
        p_down=0.10,
        seed=123,
    )

    series = df["value"]
    regimes = df["regime"]

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "regime_switching_example.csv", index_label="t")
    print(f"Generated regime-switching series of length {len(series)}")

    # 2. Basic inspection: how many points in each regime?
    counts = regimes.value_counts().to_dict()
    print("\nRegime counts:", counts)

    # 3. Evaluation settings
    horizon = 1
    initial_window = 200  # need enough history to see both regimes

    # 4. Naive forecast
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
        title="Regime-switching – Naive forecast vs actual",
        filename="regime_switching_naive_vs_actual.png",
    )

    # 5. Mean forecast
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
        title="Regime-switching – Mean forecast vs actual",
        filename="regime_switching_mean_vs_actual.png",
    )

    # 6. ARIMA(1,0,0) forecast
    # This is intentionally mis-specified: it assumes constant variance.
    res_arima = rolling_forecast_origin(
        series,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=100,  # keep it computationally reasonable
    )
    metrics_arima = compute_metrics(res_arima)
    print("\nARIMA(1,0,0) forecast metrics (no volatility modeling):")
    for k, v in metrics_arima.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_arima,
        title="Regime-switching – ARIMA(1,0,0) forecast vs actual",
        filename="regime_switching_arima_vs_actual.png",
    )

    # 7. Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (REGIME-SWITCHING)")
    print("=" * 60)
    print(f"\n{'Model':<15} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(f"{'Naive':<15} {metrics_naive['MAE']:<12.4f} {metrics_naive['RMSE']:<12.4f} {metrics_naive['MAPE']:<12.4f}")
    print(f"{'Mean':<15} {metrics_mean['MAE']:<12.4f} {metrics_mean['RMSE']:<12.4f} {metrics_mean['MAPE']:<12.4f}")
    print(f"{'ARIMA(1,0,0)':<15} {metrics_arima['MAE']:<12.4f} {metrics_arima['RMSE']:<12.4f} {metrics_arima['MAPE']:<12.4f}")

    # 8. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(
        """
This process has zero mean but time-varying volatility: periods of calm
(low σ) and periods of turbulence (high σ). Standard linear models like
ARIMA(1,0,0) assume constant variance and a stable linear structure.

Because the conditional mean is always zero, and the main structure is
in the variance, we should not expect any model focused on forecasting
levels to perform dramatically better than the Mean forecast. In fact:

  • Mean forecast often does surprisingly well, since the true mean is 0.
  • Naive forecast can be fragile, especially when the process jumps
    between calm and turbulent states.
  • ARIMA cannot "explain" the volatility shifts using a linear mean
    structure and will typically not gain much over the Mean.

This experiment motivates volatility models (e.g. ARCH/GARCH) and regime
models (e.g. Markov-switching), which focus on forecasting the dynamics
of the variance or the regime, not just the level.
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