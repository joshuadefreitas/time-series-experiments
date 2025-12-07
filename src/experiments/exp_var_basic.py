"""
Experiment: Multivariate forecasting with VAR(1) vs univariate ARIMA.

We generate a 2D VAR(1) process and compare:
  - ARIMA(1,0,0) fitted separately on each series (univariate)
  - VAR(1) fitted jointly on both series

The goal is to see when using cross-series information (VAR) helps
compared to modeling each series in isolation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from src.simulators import generate_var1
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def main():
    # 1. Generate VAR(1) data
    n = 800
    df = generate_var1(n=n, sigma=1.0, seed=123)

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "var1_example.csv")

    print("Generated VAR(1) series of length", len(df))
    print(df.head())

    y1 = df["y1"]
    y2 = df["y2"]

    horizon = 1
    initial_window = 200

    # 2. Univariate ARIMA(1,0,0) for each series
    print("\nFitting univariate ARIMA(1,0,0) on each series...")

    res_y1_arima = rolling_forecast_origin(
        y1,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=200,
    )
    metrics_y1_arima = compute_metrics(res_y1_arima)

    res_y2_arima = rolling_forecast_origin(
        y2,
        lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=horizon,
        initial_window=initial_window,
        max_forecasts=200,
    )
    metrics_y2_arima = compute_metrics(res_y2_arima)

    plot_forecast_results(
        res_y1_arima,
        title="VAR(1) – Univariate ARIMA on y1",
        filename="var1_univariate_arima_y1.png",
    )

    plot_forecast_results(
        res_y2_arima,
        title="VAR(1) – Univariate ARIMA on y2",
        filename="var1_univariate_arima_y2.png",
    )

    # 3. VAR(1) forecasting (multivariate)
    print("\nFitting VAR(1) jointly on [y1, y2]...")

    results_var_y1 = []
    results_var_y2 = []

    # We'll do a manual walk-forward for VAR
    for t in range(initial_window, n - horizon):
        # training data up to time t-1
        train = df.iloc[:t]

        # fit VAR(1)
        model = VAR(train)
        res = model.fit(maxlags=1)
        k_ar = res.k_ar

        # prepare last k_ar observations as starting point
        last_obs = train.values[-k_ar:]
        fc = res.forecast(last_obs, steps=horizon)

        # horizon=1, so take first row
        y_pred_vec = fc[0]
        y_pred1 = float(y_pred_vec[0])
        y_pred2 = float(y_pred_vec[1])

        # actual at t (next point)
        y_true_vec = df.iloc[t]
        y_true1 = float(y_true_vec["y1"])
        y_true2 = float(y_true_vec["y2"])

        results_var_y1.append(
            {
                "t_forecast": t,
                "horizon": horizon,
                "y_true": y_true1,
                "y_pred": y_pred1,
            }
        )

        results_var_y2.append(
            {
                "t_forecast": t,
                "horizon": horizon,
                "y_true": y_true2,
                "y_pred": y_pred2,
            }
        )

    res_y1_var = pd.DataFrame(results_var_y1)
    res_y2_var = pd.DataFrame(results_var_y2)

    metrics_y1_var = compute_metrics(res_y1_var)
    metrics_y2_var = compute_metrics(res_y2_var)

    plot_forecast_results(
        res_y1_var,
        title="VAR(1) – Multivariate VAR forecast for y1",
        filename="var1_multivariate_y1.png",
    )

    plot_forecast_results(
        res_y2_var,
        title="VAR(1) – Multivariate VAR forecast for y2",
        filename="var1_multivariate_y2.png",
    )

    # 4. Summary comparison
    print("\n" + "=" * 60)
    print("VAR(1) – MODEL COMPARISON (y1)")
    print("=" * 60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(
        f"{'ARIMA(1,0,0)':<20} "
        f"{metrics_y1_arima['MAE']:<12.4f} "
        f"{metrics_y1_arima['RMSE']:<12.4f} "
        f"{metrics_y1_arima['MAPE']:<12.4f}"
    )
    print(
        f"{'VAR(1)':<20} "
        f"{metrics_y1_var['MAE']:<12.4f} "
        f"{metrics_y1_var['RMSE']:<12.4f} "
        f"{metrics_y1_var['MAPE']:<12.4f}"
    )

    print("\n" + "=" * 60)
    print("VAR(1) – MODEL COMPARISON (y2)")
    print("=" * 60)
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    print(
        f"{'ARIMA(1,0,0)':<20} "
        f"{metrics_y2_arima['MAE']:<12.4f} "
        f"{metrics_y2_arima['RMSE']:<12.4f} "
        f"{metrics_y2_arima['MAPE']:<12.4f}"
    )
    print(
        f"{'VAR(1)':<20} "
        f"{metrics_y2_var['MAE']:<12.4f} "
        f"{metrics_y2_var['RMSE']:<12.4f} "
        f"{metrics_y2_var['MAPE']:<12.4f}"
    )

    # 5. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(
        """
We simulated a 2D VAR(1) system where both series influence each other.

Univariate ARIMA models each series in isolation and ignores the other
dimension, while VAR(1) uses cross-series information.

Typical patterns you may see:
  • If one series helps predict the other, VAR(1) can outperform ARIMA
    on at least one of the series (lower MAE/RMSE).

  • If the cross-effects are weak, VAR and ARIMA may perform similarly.
    In that case, the benefit of multivariate modeling is limited.

The key idea:
  When series are linked (economically, physically, or statistically),
  modeling them jointly can improve forecasts compared to treating
  each one as an isolated univariate process.
"""
    )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error running VAR experiment: {e}")
        traceback.print_exc()
        raise