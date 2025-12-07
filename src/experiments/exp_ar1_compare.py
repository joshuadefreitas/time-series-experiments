"""
Experiment: compare naive, mean, and ARIMA(1,0,0) on a simulated AR(1) series.

Run from project root with:
    python -m src.experiments.exp_ar1_compare
"""

import sys
from pathlib import Path

print("Starting experiment...")
sys.stdout.flush()  # Ensure output is visible immediately

from src.simulators import generate_ar1
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate synthetic AR(1) data
    series = generate_ar1(n=500, phi=0.7, sigma=1.0, seed=42)

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(data_dir / "ar1_example.csv", index_label="t")
    print(f"Generated AR(1) series of length {len(series)}")

    horizon = 1
    initial_window = 100

    # 2. Naive
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
        title="AR(1) - Naive forecast vs actual",
        filename="ar1_naive_vs_actual.png",
    )

    # 3. Mean
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
        title="AR(1) - Mean forecast vs actual",
        filename="ar1_mean_vs_actual.png",
    )

    # 4. ARIMA(1,0,0)
    res_arima = rolling_forecast_origin(
        series,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
    )
    metrics_arima = compute_metrics(res_arima)
    print("\nARIMA(1,0,0) forecast metrics:")
    for k, v in metrics_arima.items():
        print(f"  {k}: {v:.4f}")

    plot_forecast_results(
        res_arima,
        title="AR(1) - ARIMA(1,0,0) forecast vs actual",
        filename="ar1_arima_vs_actual.png",
    )

    # 5. Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(f"{'Naive':<20} {metrics_naive['MAE']:<12.4f} {metrics_naive['RMSE']:<12.4f} {metrics_naive['MAPE']:<12.4f}")
    print(f"{'Mean':<20} {metrics_mean['MAE']:<12.4f} {metrics_mean['RMSE']:<12.4f} {metrics_mean['MAPE']:<12.4f}")
    print(f"{'ARIMA(1,0,0)':<20} {metrics_arima['MAE']:<12.4f} {metrics_arima['RMSE']:<12.4f} {metrics_arima['MAPE']:<12.4f}")
    
    # Find best model for each metric
    print("\n" + "-"*60)
    print("BEST MODEL BY METRIC:")
    print("-"*60)
    
    models = {
        'Naive': metrics_naive,
        'Mean': metrics_mean,
        'ARIMA(1,0,0)': metrics_arima
    }
    
    best_mae = min(models.items(), key=lambda x: x[1]['MAE'])
    best_rmse = min(models.items(), key=lambda x: x[1]['RMSE'])
    best_mape = min(models.items(), key=lambda x: x[1]['MAPE'])
    
    print(f"  MAE:  {best_mae[0]} ({best_mae[1]['MAE']:.4f})")
    print(f"  RMSE: {best_rmse[0]} ({best_rmse[1]['RMSE']:.4f})")
    print(f"  MAPE: {best_mape[0]} ({best_mape[1]['MAPE']:.4f})")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("""
  • MAE (Mean Absolute Error): Average absolute forecast error.
    Lower is better. Units match the data (same scale).
    
  • RMSE (Root Mean Squared Error): Penalizes large errors more.
    Lower is better. Units match the data (same scale).
    
  • MAPE (Mean Absolute Percentage Error): Percentage error.
    Lower is better. Scale-invariant but unreliable when values
    are close to zero (which explains the high MAPE values here).
    
  Note: For this AR(1) series, ARIMA(1,0,0) should theoretically
  perform best since it matches the true data-generating process.
  Focus on MAE and RMSE for reliable comparison; MAPE may be
  inflated due to values near zero in the series.
    """)
    print("="*60)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error running experiment: {e}")
        traceback.print_exc()
        raise