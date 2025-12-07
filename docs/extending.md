# Extending the Framework

This document explains how to extend the repository with new simulators, models, experiments, and (later) real data.

---

## 1. Adding a New Simulator

All synthetic data generators live in `src/simulators.py`.

A simulator is usually a function that returns a `pandas.Series` or `pandas.DataFrame`.

**Example: simple random walk**

```python
import numpy as np
import pandas as pd

def generate_random_walk(n: int = 500, sigma: float = 1.0, seed: int | None = None) -> pd.Series:
    if seed is not None:
        np.random.seed(seed)
    shocks = np.random.normal(loc=0.0, scale=sigma, size=n)
    x = np.cumsum(shocks)
    s = pd.Series(x)
    s.index.name = "t"
    s.name = "value"
    return s
```

You can then use this in a new experiment or directly with the evaluation utilities.

---

## 2. Adding a New Model

Baseline models live in `src/models_basic.py`.  
ARIMA and SARIMA models live in `src/models_arima.py`.

A forecasting model should be a function with signature:

```python
def my_forecast(series: pd.Series, horizon: int) -> float:
    ...
```

**Example: last-k moving average model**

```python
def last_k_mean(series: pd.Series, horizon: int, k: int = 20) -> float:
    window = series.iloc[-k:] if len(series) > k else series
    return float(window.mean())
```

To keep the interface simple, you can wrap parameters via a lambda when calling the evaluation function:

```python
from src.evaluation import rolling_forecast_origin

results = rolling_forecast_origin(
    series=series,
    forecast_func=lambda s, h: last_k_mean(s, h, k=20),
    horizon=1,
    initial_window=100,
)
```

---

## 3. Creating a New Experiment

Experiments live under `src/experiments/`.  
Each experiment follows roughly this pattern:

1. Generate or load a time series.  
2. Define one or more forecasting functions.  
3. Run `rolling_forecast_origin` for each model.  
4. Compute metrics with `compute_metrics`.  
5. Plot results with `plot_forecast_results`.  
6. Print a short interpretation.

**Skeleton template**

```python
from pathlib import Path

from src.simulators import generate_ar1  # or your own
from src.evaluation import rolling_forecast_origin, compute_metrics
from src.plots import plot_forecast_results
from src.models_arima import arima_forecast

def main():
    series = generate_ar1(n=500, phi=0.7, sigma=1.0, seed=42)

    res = rolling_forecast_origin(
        series=series,
        forecast_func=lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
        horizon=1,
        initial_window=100,
    )

    metrics = compute_metrics(res)
    print("Metrics:", metrics)

    plot_forecast_results(
        res,
        title="Example Experiment",
        filename="example_experiment.png",
    )

if __name__ == "__main__":
    main()
```

Name the file something like `src/experiments/exp_my_new_idea.py`.

---

## 4. Adding Real Data

Real datasets can be stored under `data/real/`.

For example:

```text
data/
├── simulated/
└── real/
    ├── airline_passengers.csv
    ├── electricity_load.csv
    └── fx_eurusd.csv
```

A real-data experiment would:

1. Load a CSV with `pandas.read_csv`.  
2. Select or transform a time series.  
3. Apply the same pattern as in synthetic experiments.  

This keeps the evaluation logic identical while only changing the source of the data.

---

## 5. Keeping Things Simple

The design aims to stay simple:

- A small number of core utilities (`simulators`, `evaluation`, `plots`).  
- Experiments as plain scripts under `src/experiments/`.  
- Models as simple functions.

This makes the repository easy to grow over time without turning into a full-blown framework.
