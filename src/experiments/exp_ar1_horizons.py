"""
Experiment: Horizon analysis on a simulated AR(1) time series.

We compare Naive, Mean, ARIMA(1,0,0), and a lag-based linear regression
model across different forecast horizons to understand how model
performance degrades as we look further into the future.

Run from project root with:
    python -m src.experiments.exp_ar1_horizons
"""

from pathlib import Path

from src.simulators import generate_ar1
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.models_advanced import lag_regression_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate synthetic AR(1) data
    n = 500
    phi = 0.7
    sigma = 1.0
    series = generate_ar1(n=n, phi=phi, sigma=sigma, seed=42)

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(data_dir / "ar1_horizons_example.csv", index_label="t")
    print(f"Generated AR(1) series of length {len(series)} with phi={phi}")

    # 2. Define horizons and evaluation settings
    horizons = [1, 5, 10, 20]
    initial_window = 100  # initial training window

    results_summary = []  # list of dicts: horizon, model, MAE, RMSE, MAPE

    for h in horizons:
        print("\n" + "=" * 60)
        print(f"HORIZON h = {h}")
        print("=" * 60)

        # --- Naive forecast ---
        res_naive = rolling_forecast_origin(
            series,
            naive_forecast,
            horizon=h,
            initial_window=initial_window,
        )
        metrics_naive = compute_metrics(res_naive)
        print("\nNaive forecast metrics:")
        for k, v in metrics_naive.items():
            print(f"  {k}: {v:.4f}")

        plot_forecast_results(
            res_naive,
            title=f"AR(1) – Naive forecast vs actual (h={h})",
            filename=f"ar1_naive_h{h}_vs_actual.png",
        )

        results_summary.append(
            {
                "horizon": h,
                "model": "Naive",
                **metrics_naive,
            }
        )

        # --- Mean forecast ---
        res_mean = rolling_forecast_origin(
            series,
            mean_forecast,
            horizon=h,
            initial_window=initial_window,
        )
        metrics_mean = compute_metrics(res_mean)
        print("\nMean forecast metrics:")
        for k, v in metrics_mean.items():
            print(f"  {k}: {v:.4f}")

        plot_forecast_results(
            res_mean,
            title=f"AR(1) – Mean forecast vs actual (h={h})",
            filename=f"ar1_mean_h{h}_vs_actual.png",
        )

        results_summary.append(
            {
                "horizon": h,
                "model": "Mean",
                **metrics_mean,
            }
        )

        # --- ARIMA(1,0,0) forecast ---
        res_arima = rolling_forecast_origin(
            series,
            lambda s, hh: arima_forecast(s, hh, order=(1, 0, 0)),
            horizon=h,
            initial_window=initial_window,
        )
        metrics_arima = compute_metrics(res_arima)
        print("\nARIMA(1,0,0) forecast metrics:")
        for k, v in metrics_arima.items():
            print(f"  {k}: {v:.4f}")

        plot_forecast_results(
            res_arima,
            title=f"AR(1) – ARIMA(1,0,0) forecast vs actual (h={h})",
            filename=f"ar1_arima_h{h}_vs_actual.png",
        )

        results_summary.append(
            {
                "horizon": h,
                "model": "ARIMA(1,0,0)",
                **metrics_arima,
            }
        )

        # --- Lag-based linear regression forecast ---
        res_lagreg = rolling_forecast_origin(
            series,
            lambda s, hh: lag_regression_forecast(s, hh, n_lags=20),
            horizon=h,
            initial_window=initial_window,
        )
        metrics_lagreg = compute_metrics(res_lagreg)
        print("\nLag-regression (20 lags) forecast metrics:")
        for k, v in metrics_lagreg.items():
            print(f"  {k}: {v:.4f}")

        plot_forecast_results(
            res_lagreg,
            title=f"AR(1) – Lag-regression (20 lags) forecast vs actual (h={h})",
            filename=f"ar1_lagreg_h{h}_vs_actual.png",
        )

        results_summary.append(
            {
                "horizon": h,
                "model": "LagReg (20 lags)",
                **metrics_lagreg,
            }
        )

    # 3. Print summary table across horizons
    print("\n" + "=" * 60)
    print("HORIZON ANALYSIS SUMMARY (AR(1))")
    print("=" * 60)

    print(f"\n{'Horizon':<8} {'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    for record in results_summary:
        print(
            f"{record['horizon']:<8} "
            f"{record['model']:<20} "
            f"{record['MAE']:<12.4f} "
            f"{record['RMSE']:<12.4f} "
            f"{record['MAPE']:<12.4f}"
        )

    # 4. Best model per horizon (by MAE)
    print("\n" + "-" * 60)
    print("BEST MODEL BY HORIZON (MAE)")
    print("-" * 60)

    horizons_unique = sorted({r["horizon"] for r in results_summary})
    for h in horizons_unique:
        subset = [r for r in results_summary if r["horizon"] == h]
        best = min(subset, key=lambda r: r["MAE"])
        print(
            f"h = {h:<2}  →  {best['model']:<18} "
            f"(MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f}, MAPE={best['MAPE']:.2f})"
        )

    # 5. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(
        """
For an AR(1) process, ARIMA(1,0,0) is well-specified and should perform
strongly at short horizons (h = 1). As the forecast horizon increases,
all models see their errors grow:

  • At h = 1:
      ARIMA(1,0,0) typically outperforms Naive and Mean, since it matches
      the underlying data-generating mechanism. A lag-based regression
      with enough lags can behave similarly.

  • At intermediate horizons (e.g. h = 5, 10):
      The advantage of ARIMA starts to shrink. Errors accumulate and the
      series gradually "forgets" the initial condition. Differences between
      ARIMA and lag-regression often become small.

  • At long horizons (e.g. h = 20):
      Forecasts converge towards the unconditional mean of the process.
      Structural details of the AR(1) (and of any linear model on lags)
      become less important, and simple models like the Mean forecast can
      be relatively competitive.

The key idea:
  The value of model sophistication depends not just on the structure
  of the time series, but also on how far ahead you are trying to see.
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