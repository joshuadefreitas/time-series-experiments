from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_forecast_results(results: pd.DataFrame, title: str, filename: str) -> None:
    """
    Plot actual vs forecast from a results DataFrame with
    columns ["t_forecast", "y_true", "y_pred"] and save to PNG.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(results["t_forecast"], results["y_true"], label="Actual")
    plt.plot(results["t_forecast"], results["y_pred"], label="Forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")