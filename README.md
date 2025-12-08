# Time-Series Forecasting Experiments

A compact set of experiments designed to understand how different forecasting models behave under controlled conditions.  
Not a showcase, not a “look at my project” portfolio piece — just clean experiments that explain model behaviour.

## Focus
ARIMA/SARIMA · Volatility · Structural Breaks · VAR · Nonlinear Dynamics · Chaos

## Overview

The project studies how forecasting models react when the underlying data-generating process changes.  
Everything starts from synthetic data so the ground truth is always known.

Covered systems include:

- **AR(1)** memory and horizon decay  
- **Trend + seasonality** with SARIMA  
- **Variance regimes** (switching volatility)  
- **GARCH‑like clustering**  
- **Structural breaks**  
- **VAR(1)** multivariate interaction  
- **Nonlinear/chaotic series** (logistic map, Lorenz attractor)

The goal is simple: understand what works *and why* — not chase leaderboard metrics.

---

## Current Experiments

### AR(1) Horizon Dependence  
How predictive power fades with longer horizons.

`src/experiments/exp_ar1_horizons.py`

### Trend + Seasonality  
SARIMA vs naïve and mean baselines.

`src/experiments/exp_trend_seasonal_compare.py`

### Regime Switching  
Why volatility—not levels—carries the structure.

`src/experiments/exp_regime_switch_compare.py`

### GARCH-like Volatility  
Volatility clustering and fat‑tailed behaviour.

`src/experiments/exp_garch_compare.py`

### Structural Breaks  
How forecasters behave when the mean jumps.

`src/experiments/exp_structural_breaks.py`

### VAR  
When multivariate dynamics actually help.

`src/experiments/exp_var_basic.py`

### Chaos  
Deterministic systems with vanishing predictability.

`src/experiments/exp_logistic_chaos.py`  
`src/experiments/exp_lorenz_chaos.py`

---

## Sample Plots

Plots live in `outputs/plots/`.  
A few highlights:

- SARIMA on trend + seasonality  
- AR(1) horizon comparison  
- Regime switching forecasts  
- GARCH volatility paths  
- Lorenz attractor (3D chaotic trajectory)

---

## Documentation

All extra notes are in `docs/`:

- `lessons.md` — distilled takeaways  
- `experiments.md` — experiment descriptions  
- `gallery.md` — selected visualisations  
- `extending.md` — how to add more models or simulators  

---

## Roadmap

- Additional nonlinear maps (Henon, Mackey–Glass)  
- Cointegration and error‑correction experiments  
- State‑space + Kalman filtering  
- Real‑world series (FX, macro, demand)  
- Optional lightweight live‑forecast runner  

---

## Project Structure

```
time-series-experiments/
├── src/
│   ├── simulators.py
│   ├── models_basic.py
│   ├── models_arima.py
│   ├── models_ml.py
│   ├── evaluation.py
│   └── experiments/
│       ├── exp_ar1_horizons.py
│       ├── exp_trend_seasonal_compare.py
│       ├── exp_regime_switch_compare.py
│       ├── exp_garch_compare.py
│       ├── exp_structural_breaks.py
│       ├── exp_var_basic.py
│       ├── exp_logistic_chaos.py
│       └── exp_lorenz_chaos.py
├── outputs/
│   └── plots/
├── data/
│   ├── simulated/
│   └── real/
├── docs/
└── requirements.txt
```

---

## Installation

```
git clone https://github.com/joshuadefreitas/time-series-experiments.git
cd time-series-experiments
pip install -r requirements.txt
```

---

## Running an Experiment

Example:

```
python -m src.experiments.exp_ar1_horizons
```

Each script:

- generates synthetic data  
- runs walk‑forward evaluation  
- saves plots  
- prints a short interpretation  

---

## Methodology Notes

### Rolling-Origin Evaluation  
Walk‑forward validation ensures no leakage and keeps conditions realistic.

### Metrics  
- MAE  
- RMSE  
- MAPE (treated carefully near zero)  

---

## Why Synthetic Data First?

Because it isolates what you want to learn:

- what memory looks like  
- what seasonality does  
- how breaks ruin global forecasts  
- why volatility matters  
- when multivariate structure helps  
- where predictability simply ends  

Only once that’s clear does it make sense to move to real data.

---

## License

MIT  
Free for research, learning, and experimentation.
