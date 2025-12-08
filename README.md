# Time-Series Forecasting Experiments

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Focus](https://img.shields.io/badge/focus-time--series%20%7C%20simulation%20%7C%20econometrics-orange)
![NumPy](https://img.shields.io/badge/numpy-1.x-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-green)

A practical collection of forecasting experiments designed to study how different models behave under controlled conditions.  
The focus is on *understanding behaviour*, not tuning for leaderboard performance.

The project covers a wide range of time-series dynamics:

- ARIMA & SARIMA (memory, trend, seasonality)
- Volatility and regime switching
- Structural breaks
- Multivariate VAR interactions
- Nonlinear & chaotic systems (logistic map, Lorenz)

---

## Key Features

- Modular synthetic simulators (AR, trend+seasonality, GARCH-like, regime shifts, VAR, chaos)
- Proper rolling-origin evaluation (walk-forward)
- Comparison of baselines, ARIMA/SARIMA, volatility-aware and multivariate models
- Clear documentation in `docs/`:
  - `lessons.md` — key concepts  
  - `experiments.md` — overview of all experiments  
  - `gallery.md` — visual examples  
  - `extending.md` — how to add models or experiments  

The structure is simple, extendable, and ready for future real-world datasets.

---

## Summary

This repository examines how forecasting models react to different underlying mechanisms:

- When simple models match or outperform complex ones  
- How forecast horizons affect accuracy  
- When seasonal structure dominates  
- Why volatility dynamics often matter more than the mean  
- How structural breaks disrupt global models  
- Limits of predictability in chaotic systems  
- When multivariate modelling helps (and when it doesn’t)

---

## Table of Contents

- [Purpose](#purpose)
- [Current Experiments](#current-experiments)
- [Sample Plots](#sample-plots)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Methodology](#methodology)
- [Why Synthetic Data](#why-synthetic-data)
- [Dependencies](#dependencies)
- [License](#license)

---

## Purpose

The project exists to answer practical questions like:

- How far ahead can different models see?  
- What breaks a model and why?  
- How does volatility structure change predictability?  
- What happens when the underlying system switches regimes?  
- Can chaotic systems be forecast at all beyond a step or two?  
- Do multivariate dependencies help forecast accuracy?

The framework provides:

- Controlled simulators  
- Walk-forward evaluation  
- Small model library  
- Reproducible experiments  

---

## Current Experiments

### **1. AR(1) memory & horizon dependence**  
`src/experiments/exp_ar1_horizons.py`

### **2. Trend + seasonality (SARIMA vs baselines)**  
`src/experiments/exp_trend_seasonal_compare.py`

### **3. Regime switching (variance shifts)**  
`src/experiments/exp_regime_switch_compare.py`

### **4. GARCH-like volatility clustering**  
`src/experiments/exp_garch_compare.py`

### **5. Structural breaks**  
`src/experiments/exp_structural_breaks.py`

### **6. Multivariate VAR forecasting**  
`src/experiments/exp_var_basic.py`

### **7. Chaos & nonlinear dynamics**  
`src/experiments/exp_logistic_chaos.py`  
`src/experiments/exp_lorenz_chaos.py`

---

## Sample Plots

Plots stored in `outputs/plots/`.

### SARIMA on trend + seasonality  
![Trend Seasonality](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/trend_seasonal_sarima_vs_actual.png)

### Horizon comparison (AR(1))  
![AR1 Horizon](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/ar1_h10_vs_actual.png)

### Regime switching  
![Regime](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/regime_switching_mean_vs_actual.png)

### GARCH-like volatility  
![GARCH](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/garch_mean_forecast.png)

### Lorenz attractor  
![Lorenz](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/lorenz_attractor.png)

---

## Documentation

`docs/` contains short, direct explanations:

- `lessons.md` — key ideas behind the experiments  
- `experiments.md` — how each script works  
- `gallery.md` — selected plots  
- `extending.md` — how to add models, simulators, or experiments  

---

## Roadmap

- Additional nonlinear systems (Henon map, Mackey–Glass)
- Cointegration & ECM models
- State-space models & Kalman filtering
- First real-world datasets (FX, electricity, macro)
- Optional small forecasting pipeline for scheduled runs

---

## Project Structure

```text
time-series-experiments/
├── src/
│   ├── simulators.py
│   ├── models_basic.py
│   ├── models_arima.py
│   ├── models_ml.py
│   ├── evaluation.py
│   ├── plots.py
│   └── experiments/
│       ├── exp_ar1_horizons.py
│       ├── exp_trend_seasonal_compare.py
│       ├── exp_regime_switch_compare.py
│       ├── exp_garch_compare.py
│       ├── exp_structural_breaks.py
│       ├── exp_var_basic.py
│       ├── exp_logistic_chaos.py
│       └── exp_lorenz_chaos.py
├── docs/
├── data/
│   ├── simulated/
│   └── real/
├── outputs/
│   └── plots/
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/joshuadefreitas/time-series-experiments.git
cd time-series-experiments
pip install -r requirements.txt
```

---

## Running Experiments

Example:

```bash
python -m src.experiments.exp_ar1_horizons
```

Each experiment:

- generates synthetic data  
- performs rolling-origin evaluation  
- saves plots in `outputs/plots/`  
- prints metrics and interpretation  

---

## Methodology

### Rolling-origin evaluation  
Walk-forward forecasting without leakage.

### Metrics  
- MAE  
- RMSE  
- MAPE  

---

## Why Synthetic Data

Synthetic data makes it possible to:

- control the structure  
- isolate behaviour  
- compare models fairly  
- diagnose why a model succeeds or fails  

Real-world datasets will later extend these controlled settings.

---

## Dependencies

- numpy  
- pandas  
- matplotlib  
- statsmodels  

Optional:
- scikit-learn  
- xgboost  
- lightgbm  

---

## License

MIT
