"""
Experiment: Forecasting the Lorenz system (chaotic dynamics).

We simulate the Lorenz attractor, project onto the x-coordinate,
and try to forecast x_t using a simple ARIMA(1,0,0) model for
different horizons (h = 1, 5, 10).

The Lorenz system is deterministic but chaotic, so long-horizon
prediction is fundamentally limited.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.simulators import generate_lorenz
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results


def save_lorenz_3d_plot(df, filename: str = "lorenz_attractor.png"):
    """
    Save a 3D plot of the Lorenz attractor.

    Args:
        df: DataFrame with columns ['x', 'y', 'z'].
        filename: output file name in outputs/plots/.
    """
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df["x"], df["y"], df["z"], linewidth=0.5)
    ax.set_title("Lorenz Attractor (x, y, z)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved Lorenz 3D plot to {filepath}")


def main():
    # 1. Simulate Lorenz system (shorter trajectory is enough)
    n_steps = 6000
    df = generate_lorenz(
        n_steps=n_steps,
        dt=0.01,
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        x0=1.0,
        y0=1.0,
        z0=1.0,
    )

    data_dir = Path("data/simulated")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "lorenz_example.csv")

    print(f"Generated Lorenz trajectory with {len(df)} steps")

    # Save a 3D visualization of the attractor
    save_lorenz_3d_plot(df)

    # Use x(t) as a univariate series
    series = df["x"]
    series.name = "x"

    # Drop initial transient
    burn_in = 1000
    series = series.iloc[burn_in:].reset_index(drop=True)
    print(f"Using series of length {len(series)} after burn-in")

    # 2. Forecast settings (keep it reasonable for runtime)
    horizons = [1, 5, 10]
    initial_window = 500
    max_forecasts = 200  # cap number of rolling forecasts for speed

    results = {}
    metrics = {}

    for h in horizons:
        print(f"\nRunning ARIMA(1,0,0) on Lorenz x_t with horizon h = {h} ...")

        res = rolling_forecast_origin(
            series,
            lambda s, hh: arima_forecast(s, hh, order=(1, 0, 0)),
            horizon=h,
            initial_window=initial_window,
            max_forecasts=max_forecasts,
        )
        results[h] = res
        metrics[h] = compute_metrics(res)

        plot_forecast_results(
            res,
            title=f"Lorenz x_t – ARIMA(1,0,0) forecast, h={h}",
            filename=f"lorenz_x_arima_h{h}.png",
        )

        print(f"Metrics (h = {h}):")
        for k, v in metrics[h].items():
            print(f"  {k}: {v:.4f}")

    # 3. Summary across horizons
    print("\n" + "=" * 60)
    print("LORENZ x_t – HORIZON COMPARISON (ARIMA(1,0,0))")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
    print("-" * 60)
    for h in horizons:
        m = metrics[h]
        print(
            f"{h:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['MAPE']:<12.4f}"
        )

    # 4. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(
        """
The Lorenz system is a classic example of deterministic chaos:
small differences in the trajectory grow rapidly over time.

We projected the 3D system onto x_t and tried to forecast it with
a simple ARIMA(1,0,0) model.

What you typically see:
  • For very short horizons (h = 1), forecasts sometimes follow
    the local pattern reasonably.

  • As the horizon grows (h = 5, 10), forecasts no longer track
    the detailed trajectory. The model effectively predicts
    'typical' values rather than the exact path.

The key lesson:
  In chaotic systems, even with perfect data and a decent local
  model, long-horizon prediction is fundamentally limited by
  sensitivity to initial conditions.
"""
    )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error running Lorenz chaos experiment: {e}")
        traceback.print_exc()
        raise