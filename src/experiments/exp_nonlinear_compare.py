"""
Experiment: Compare classical and ML models on a nonlinear threshold AR process.

We simulate a nonlinear autoregressive process with a regime change depending
on the sign of y_{t-1}:

    if y_{t-1} > 0:
        y_t = c_pos + phi_pos * y_{t-1} + eps_t
    else:
        y_t = c_neg + phi_neg * y_{t-1} + eps_t

This creates a kinked, nonlinear mean function: the dynamics differ in
positive vs negative regions.

Models compared:
  - Naive
  - Mean
  - ARIMA(1,0,0)
  - LagRegression (linear, scikit-learn)
  - RandomForestRegressor
  - XGBRegressor
  - LGBMRegressor

We use rolling-origin evaluation with horizon h = 1.

Run from project root with:
    python -m src.experiments.exp_nonlinear_compare
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.models_basic import naive_forecast, mean_forecast
from src.models_arima import arima_forecast
from src.models_advanced import lag_regression_forecast
from src.models_ml import rf_forecast, xgb_forecast, lgbm_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


# ---------------------------------------------------------------------------
# Nonlinear threshold AR simulator
# ---------------------------------------------------------------------------

def generate_threshold_ar(
    n: int,
    phi_pos: float = 0.8,
    phi_neg: float = -0.2,
    c_pos: float = 0.5,
    c_neg: float = -0.5,
    sigma: float = 1.0,
    seed: int = 123,
) -> pd.Series:
    """
    Generate a nonlinear threshold AR(1)-type process.

    If y_{t-1} > 0:
        y_t = c_pos + phi_pos * y_{t-1} + eps_t
    else:
        y_t = c_neg + phi_neg * y_{t-1} + eps_t

    Args:
        n: length of the time series.
        phi_pos: AR coefficient in positive regime.
        phi_neg: AR coefficient in negative regime.
        c_pos: intercept in positive regime.
        c_neg: intercept in negative regime.
        sigma: std dev of Gaussian noise.
        seed: random seed.

    Returns:
        pd.Series of length n.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)

    y = np.zeros(n)
    for t in range(1, n):
        if y[t - 1] > 0:
            y[t] = c_pos + phi_pos * y[t - 1] + eps[t]
        else:
            y[t] = c_neg + phi_neg * y[t - 1] + eps[t]

    return pd.Series(y, index=pd.RangeIndex(n), name="y")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    # 1. Generate nonlinear threshold AR data
    n = 600
    series = generate_threshold_ar(
        n=n,
        phi_pos=0.8,
        phi_neg=-0.2,
        c_pos=0.5,
        c_neg=-0.5,
        sigma=1.0,
        seed=42,
    )

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(data_dir / "threshold_ar_example.csv", index_label="t")
    print(f"Generated nonlinear threshold AR series of length {len(series)}")

    # 2. Evaluation settings
    horizon = 1
    initial_window = 150  # initial training window
    n_lags = 20           # for lag-based models

    results_summary = []  # list of dicts: model, MAE, RMSE, MAPE

    print("\nUsing rolling-origin evaluation with:")
    print(f"  horizon        = {horizon}")
    print(f"  initial_window = {initial_window}")
    print(f"  n_lags         = {n_lags} (for ML/lag models)")

    # ------------------------------------------------------------------
    # Helper: run a model and record metrics + plot
    # ------------------------------------------------------------------
    def run_model(name: str, forecast_func, plot_name_suffix: str):
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
            title=f"Threshold AR – {name} forecast vs actual (h={horizon})",
            filename=f"threshold_ar_{plot_name_suffix}_h{horizon}_vs_actual.png",
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

    # ARIMA(1,0,0) – linear model, misspecified for the threshold nonlinearity
    run_model(
        name="ARIMA(1,0,0)",
        forecast_func=lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        plot_name_suffix="arima100",
    )

    # LagRegression (linear regression on many lags)
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
    print(f"MODEL COMPARISON SUMMARY (Threshold AR, h={horizon})")
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
        """
We simulated a nonlinear threshold autoregressive process where the
dynamics depend on the sign of y_{t-1}. This creates a kinked mean
function: one behavior when the series is positive, another when negative.

Key ideas to look for in the results:

  • ARIMA(1,0,0):
      A linear model. It tries to fit a single straight line through
      what is actually a piecewise relationship. It can capture some
      average behavior but misses the threshold.

  • LagReg (linear regression on many lags):
      Still linear, but with more lags as features. It may smooth over
      the threshold and approximate the average effect, but cannot
      represent the kink exactly.

  • RandomForest / XGBoost / LightGBM:
      Tree-based models can naturally represent piecewise functions and
      thresholds. On this kind of nonlinear process, they often have an
      advantage, especially if there is enough data to learn the split.

  • Naive & Mean:
      Simple baselines. They ignore the structure and are mainly here as
      reference points.

The main lesson:
  On a simple linear AR(1), classical ARIMA is hard to beat.
  Once the data-generating process becomes nonlinear (like this threshold
  model), tree-based ML models have a real chance to outperform linear
  models by capturing regime-dependent behavior that ARIMA cannot.
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