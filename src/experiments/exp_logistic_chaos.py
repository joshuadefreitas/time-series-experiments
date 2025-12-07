"""
Experiment: Forecasting a chaotic logistic map.

We generate a logistic map in the chaotic regime and see how a simple
ARIMA(1,0,0) model performs at horizons h = 1, 5, 10.

The system is deterministic but highly sensitive to initial conditions,
so even with a decent local fit, multi-step forecasts quickly diverge.
"""

from pathlib import Path

from src.simulators import generate_logistic_map
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate logistic map
    n = 1000
    series = generate_logistic_map(
        n=n,
        r=3.9,
        x0=0.2,
    )

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(data_dir / "logistic_map_r3_9.csv", index_label="t")

    print(f"Generated logistic map of length {len(series)} (r=3.9, chaotic regime)")

    # Forecast settings
    horizons = [1, 5, 10]
    initial_window = 300

    results = {}
    metrics = {}

    for h in horizons:
        print(f"\nRunning ARIMA(1,0,0) with horizon h = {h} ...")

        res = rolling_forecast_origin(
            series,
            lambda s, hh: arima_forecast(s, hh, order=(1, 0, 0)),
            horizon=h,
            initial_window=initial_window,
            max_forecasts=200,
        )
        results[h] = res
        metrics[h] = compute_metrics(res)

        plot_forecast_results(
            res,
            title=f"Logistic map (r=3.9) – ARIMA(1,0,0) forecast, h={h}",
            filename=f"logistic_arima_h{h}.png",
        )

        print("Metrics (h = {}):".format(h))
        for k, v in metrics[h].items():
            print(f"  {k}: {v:.4f}")

    # Summary across horizons
    print("\n" + "=" * 60)
    print("LOGISTIC MAP – HORIZON COMPARISON (ARIMA(1,0,0))")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    for h in horizons:
        m = metrics[h]
        print(
            f"{h:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['MAPE']:<12.4f}"
        )

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(
        """
The logistic map at r = 3.9 is deterministic but chaotic: very small
differences in the starting point lead to large differences later on.

Even if ARIMA(1,0,0) fits the local behaviour reasonably well:

  • At h = 1, forecasts can look acceptable.
  • At h = 5, errors grow significantly.
  • At h = 10, forecasts are often useless.

This is not just a model failure – it reflects a property of the system:
long-horizon prediction in chaotic dynamics is fundamentally limited.

The key lesson:
  Even with perfect data and a decent local model, some systems are
  only predictable at very short horizons. Beyond that, uncertainty
  explodes due to sensitivity to initial conditions.
"""
    )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error running logistic chaos experiment: {e}")
        traceback.print_exc()
        raise