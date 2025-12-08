"""
Experiment: Real-data daily FX returns (e.g., EURUSD).

Loads a CSV from data/real/eurusd.csv (schema: date, price) and evaluates
simple level-forecasting models on log returns:
    - Naive (random walk)
    - Mean (constant mean)
    - ARIMA(1,0,0)
    - Optional: lag regression / tree models if dependencies are installed

Run from project root:
    python -m src.experiments.exp_fx_returns
"""

from pathlib import Path

from src.data_utils import load_fx_returns, fetch_fx_prices_yahoo
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.models_advanced import lag_regression_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results
from src.io_utils import save_json

# Optional ML models
try:
    from src.models_ml import rf_forecast, xgb_forecast, lgbm_forecast

    _HAS_ML = True
except ImportError:
    _HAS_ML = False


def main():
    data_path = Path("data/real/eurusd.csv")
    if data_path.exists():
        series = load_fx_returns(data_path)
        print(f"Loaded {len(series)} daily log returns from {data_path}")
    else:
        print(f"File {data_path} not found. Attempting to fetch via yfinance (EURUSD=X)...")
        try:
            df_prices = fetch_fx_prices_yahoo("EURUSD=X", start="2015-01-01")
        except Exception as e:
            raise FileNotFoundError(
                f"Missing FX data at {data_path} and failed to fetch via yfinance. "
                f"Error: {e}"
            )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_prices.to_csv(data_path, index=False)
        print(f"Saved fetched prices to {data_path}")
        series = load_fx_returns(data_path)
        print(f"Loaded {len(series)} daily log returns from fetched data.")

    # ensure sorted by date (yfinance can include gaps/weekends)
    series = series.sort_index()

    horizon = 1
    initial_window = max(250, int(len(series) * 0.3))  # shorter window for speed
    n_lags = 12
    max_forecasts = 200  # limit expensive models to last N forecasts

    results_summary = []

    print("\nUsing rolling-origin evaluation with:")
    print(f"  horizon        = {horizon}")
    print(f"  initial_window = {initial_window}")
    print(f"  n_lags         = {n_lags} (for lag/ML models)")

    # helper to run and log a model
    def run_model(name: str, forecast_func, plot_name_suffix: str, limit: bool = False):
        res = rolling_forecast_origin(
            series,
            forecast_func,
            horizon=horizon,
            initial_window=initial_window,
            max_forecasts=max_forecasts if limit else None,
        )
        metrics = compute_metrics(res)
        results_summary.append({"model": name, **metrics})

        plot_forecast_results(
            res,
            title=f"FX returns – {name} (h={horizon})",
            filename=f"fx_returns_{plot_name_suffix}_h{horizon}.png",
        )

    # Classical models
    print("\nRunning Naive...")
    run_model("Naive", naive_forecast, "naive")

    print("Running Mean...")
    run_model("Mean", mean_forecast, "mean")

    print("Running ARIMA(1,0,0)...")
    run_model(
        "ARIMA(1,0,0)",
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        "arima100",
        limit=True,  # ARIMA is slower; evaluate on last N
    )

    # Lag regression
    try:
        print(f"Running LagReg (n_lags={n_lags})...")
        run_model(
            f"LagReg (n_lags={n_lags})",
            lambda s, h: lag_regression_forecast(s, h, n_lags=n_lags),
            "lagreg",
            limit=True,
        )
    except ImportError as e:
        print(f"[SKIP] LagReg missing dependency: {e}")

    # Optional ML models
    if _HAS_ML:
        print("Running RandomForest...")
        run_model(
            f"RandomForest (n_lags={n_lags})",
            lambda s, h: rf_forecast(s, h, n_lags=n_lags),
            "rf",
            limit=True,
        )

        print("Running XGBoost...")
        run_model(
            f"XGBoost (n_lags={n_lags})",
            lambda s, h: xgb_forecast(s, h, n_lags=n_lags),
            "xgb",
            limit=True,
        )

        print("Running LightGBM...")
        run_model(
            f"LightGBM (n_lags={n_lags})",
            lambda s, h: lgbm_forecast(s, h, n_lags=n_lags),
            "lgbm",
            limit=True,
        )
    else:
        print("[INFO] ML models skipped (xgboost/lightgbm not installed).")

    # Persist summary metrics + config
    metrics_out = {
        "data": str(data_path),
        "length": len(series),
        "horizon": horizon,
        "initial_window": initial_window,
        "n_lags": n_lags,
        "models": results_summary,
    }
    out_path = save_json(metrics_out, Path("outputs/metrics/fx_returns_metrics.json"))
    print(f"\nSaved metrics to {out_path}")

    # Print concise table
    print("\n" + "=" * 60)
    print(f"FX RETURNS – MODEL COMPARISON (h={horizon})")
    print("=" * 60)
    print(f"\n{'Model':<30} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    for rec in results_summary:
        print(
            f"{rec['model']:<30} "
            f"{rec['MAE']:<12.4f} "
            f"{rec['RMSE']:<12.4f} "
            f"{rec['MAPE']:<12.4f}"
        )

    # Interpretation (keep it concise, similar to other experiments)
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(
        """
Daily FX returns are close to white noise in levels. Key takeaways:

  • Mean vs Naive:
      Mean wins on MAE/RMSE because the true mean is near zero; copying
      the last return (Naive) just copies noise.

  • ARIMA(1,0,0):
      Performs like Mean/LagReg/RF. With little linear dependence, AR terms
      don't add much and may struggle to converge; expect AR(1) to behave
      similarly to a driftless random walk at h=1.

  • LagReg / RF / XGB:
      All land near the same error as ARIMA/Mean. On noise-like returns,
      nonlinear lag models have nothing to exploit; they mostly regularize
      toward the mean.

Practical implication: for one-step FX return forecasts, simple mean/AR(1)
baselines are as good as more complex lag models. If you want an edge, you
need richer features (e.g., realized volatility, cross-asset signals) or a
different target (volatility, not level).
        """
    )


if __name__ == "__main__":
    main()
