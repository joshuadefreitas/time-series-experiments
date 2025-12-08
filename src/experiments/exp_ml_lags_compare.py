"""
Experiment: Compare classical and ML lag-based models on an AR(1) time series.

Models:
  - Naive
  - Mean
  - ARIMA(1,0,0)
  - LagRegression (linear, scikit-learn)
  - RandomForestRegressor
  - XGBRegressor
  - LGBMRegressor

We use rolling-origin evaluation with horizon h = 1.

Run from project root with:
    python -m src.experiments.exp_ml_lags_compare
"""

from pathlib import Path

from src.simulators import generate_ar1
from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.models_advanced import lag_regression_forecast
from src.models_ml import rf_forecast, xgb_forecast, lgbm_forecast
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
    series.to_csv(data_dir / "ar1_ml_compare_example.csv", index_label="t")
    print(f"Generated AR(1) series of length {len(series)} with phi={phi}")

    # 2. Evaluation settings
    horizon = 1
    initial_window = 100  # initial training window
    n_lags = 20           # for lag-based models

    results_summary = []  # list of dicts: model, MAE, RMSE, MAPE

    print("\nUsing rolling-origin evaluation with:")
    print(f"  horizon       = {horizon}")
    print(f"  initial_window= {initial_window}")
    print(f"  n_lags        = {n_lags} (for ML/lag models)")

    # ------------------------------------------------------------------
    # Helper: run a model and record metrics + plot
    # ------------------------------------------------------------------
    def run_model(name, forecast_func, plot_name_suffix):
        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        res = rolling_forecast_origin(
            series,
            forecast_func,
            horizon=horizon,
            initial_window=initial_window,
        )
        metrics = compute_metrics(res)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        plot_forecast_results(
            res,
            title=f"AR(1) – {name} forecast vs actual (h={horizon})",
            filename=f"ar1_ml_{plot_name_suffix}_h{horizon}_vs_actual.png",
        )

        results_summary.append(
            {
                "model": name,
                **metrics,
            }
        )

    # ------------------------------------------------------------------
    # 3. Classical models
    # ------------------------------------------------------------------

    # Naive
    run_model(
        name="Naive",
        forecast_func=naive_forecast,
        plot_name_suffix="naive",
    )

    # Mean
    run_model(
        name="Mean",
        forecast_func=mean_forecast,
        plot_name_suffix="mean",
    )

    # ARIMA(1,0,0)
    run_model(
        name="ARIMA(1,0,0)",
        forecast_func=lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        plot_name_suffix="arima100",
    )

    # LagRegression (linear regression on lags)
    run_model(
        name=f"LagReg (n_lags={n_lags})",
        forecast_func=lambda s, h: lag_regression_forecast(
            s,
            horizon=h,
            n_lags=n_lags,
        ),
        plot_name_suffix="lagreg",
    )

    # ------------------------------------------------------------------
    # 4. ML models – RF / XGB / LGBM
    #    We wrap each in try/except so the experiment still runs even if
    #    a library is missing.
    # ------------------------------------------------------------------

    # Random Forest
    try:
        run_model(
            name=f"RandomForest (n_lags={n_lags})",
            forecast_func=lambda s, h: rf_forecast(
                s,
                horizon=h,
                n_lags=n_lags,
            ),
            plot_name_suffix="rf",
        )
    except ImportError as e:
        print("\n[SKIP] RandomForest: missing dependency:")
        print(f"  {e}")

    # XGBoost
    try:
        run_model(
            name=f"XGBoost (n_lags={n_lags})",
            forecast_func=lambda s, h: xgb_forecast(
                s,
                horizon=h,
                n_lags=n_lags,
            ),
            plot_name_suffix="xgb",
        )
    except ImportError as e:
        print("\n[SKIP] XGBoost: missing dependency:")
        print(f"  {e}")

    # LightGBM
    try:
        run_model(
            name=f"LightGBM (n_lags={n_lags})",
            forecast_func=lambda s, h: lgbm_forecast(
                s,
                horizon=h,
                n_lags=n_lags,
            ),
            plot_name_suffix="lgbm",
        )
    except ImportError as e:
        print("\n[SKIP] LightGBM: missing dependency:")
        print(f"  {e}")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON SUMMARY (AR(1), h={horizon})")
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

    # Best by MAE
    print("\n" + "-" * 60)
    print("BEST MODEL BY MAE")
    print("-" * 60)
    best = min(results_summary, key=lambda r: r["MAE"])
    print(
        f"{best['model']}  "
        f"(MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f}, MAPE={best['MAPE']:.2f})"
    )

    # ------------------------------------------------------------------
    # 6. Interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(
        f"""
This experiment compares classical time-series models and ML models
on a simple AR(1) process with phi={phi}.

Expected patterns you may see:

  • ARIMA(1,0,0):
      Well-specified for AR(1), so it should be very strong at h = 1.

  • Mean & Naive:
      Simple baselines; Naive can do okay when phi is close to 1 (random walk),
      Mean is competitive when the series fluctuates around a stable mean.

  • LagReg (linear):
      Similar spirit to AR models, but estimated as a regression on many lags.
      With enough data, it can approximate the AR structure.

  • RandomForest / XGBoost / LightGBM:
      Nonlinear function approximators on lag features. On a pure AR(1) with
      Gaussian noise, there is limited nonlinear structure to exploit, so
      they may not outperform a correctly specified ARIMA(1,0,0).

The key takeaway:
  Machine learning models shine when there is nonlinear structure or rich
  covariates. On a clean AR(1), classical models are hard to beat. This
  experiment gives a controlled benchmark for how ML behaves when the true
  process is simple and known.
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