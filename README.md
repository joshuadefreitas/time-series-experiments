# Time Series Forecasting Experiments

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Focus](https://img.shields.io/badge/focus-time--series%20%7C%20econometrics%20%7C%20simulation-orange)
![NumPy](https://img.shields.io/badge/numpy-1.x-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-0.x-green)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-yellow)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

A practical forecasting framework for exploring how different models behave under controlled time-series conditions.  
The focus is on understanding model behaviour—not just comparing metrics.

---

## Summary

This project shows how different forecasting models behave under controlled time-series settings. It highlights:

- Working with synthetic data to isolate effects and understand model behaviour  
- Using proper rolling-origin (walk-forward) validation  
- Simulating different data-generating mechanisms  
- Comparing baseline, ARIMA/SARIMA, volatility-aware, multivariate, and chaotic settings  
- Keeping the codebase small, modular, and easy to extend  

The aim is clarity: how models react when the underlying system changes.

---

## Table of Contents

- [Purpose](#purpose-of-the-project)
- [Current Experiments](#current-capabilities)
- [Sample Plots](#sample-plots)
- [Documentation](#documentation)
- [Roadmap](#roadmap-upcoming-milestones)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Methodology](#methodology)
- [Why Start With Synthetic Data](#why-start-with-synthetic-data)
- [Dependencies](#dependencies)
- [License](#license)

---

## Purpose of the Project

The repository is built to answer practical forecasting questions:

- When do simple models outperform complex ones?  
- How much does forecasting horizon matter?  
- When does seasonality dominate?  
- Why are financial returns nearly unpredictable?  
- What happens when a structural break occurs?  
- Can chaotic systems be forecast beyond very short horizons?  
- When does multivariate modelling actually help?

The framework includes:

- Controlled synthetic simulators  
- Rolling-origin evaluation  
- A small library of models  
- Experiments with interpretable results and plots  

---

## Current Capabilities

### 1. AR(1) Memory + Horizon Dependence
How predictive power decays as horizons grow.

**Script:** `src/experiments/exp_ar1_horizons.py`

---

### 2. Trend & Seasonality (SARIMA vs Baselines)
Importance of modelling explicit seasonal structure.

**Script:** `src/experiments/exp_trend_seasonal_compare.py`

---

### 3. Regime Switching (Variance Shifts)
Why volatility, not levels, often contains the real structure.

**Script:** `src/experiments/exp_regime_switch_compare.py`

---

### 4. GARCH-like Volatility Clustering
Synthetic returns with volatility persistence.

**Script:** `src/experiments/exp_garch_compare.py`

---

### 5. Structural Breaks
Shows how global models struggle when the underlying mean shifts.

**Script:** `src/experiments/exp_structural_breaks.py`

---

### 6. Multivariate Forecasting (VAR)
Captures cross-series interactions missed by univariate models.

**Script:** `src/experiments/exp_var_basic.py`

---

### 7. Chaos & Nonlinear Dynamics
Logistic maps and the Lorenz system: limits of predictability.

**Scripts:**
- `src/experiments/exp_logistic_chaos.py`
- `src/experiments/exp_lorenz_chaos.py`

---

## Sample Plots

Plots are stored in `outputs/plots/`. A few examples:

### SARIMA on Trend + Seasonality  
![Trend Seasonality](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/trend_seasonal_sarima_vs_actual.png)

### Horizon Comparison (AR(1))  
![AR1 Horizon](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/ar1_h10_vs_actual.png)

### Regime Switching  
![Regime Switching](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/regime_switching_mean_vs_actual.png)

### GARCH-like Volatility  
![GARCH](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/garch_mean_forecast.png)

### Lorenz Attractor (Chaos)  
![Lorenz](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/lorenz_attractor.png)

---

## Documentation

Additional documentation lives in the `docs/` folder:

- **Key Lessons & Concepts:** `docs/lessons.md`  
- **Experiment Overview:** `docs/experiments.md`  
- **Plot Gallery:** `docs/gallery.md`  
- **Extending the Framework:** `docs/extending.md`  

These files summarise what each experiment teaches, show selected plots, and explain how to plug in new simulators or models.

---

## Roadmap (Upcoming Milestones)

- More nonlinear systems (e.g. Henon map, Mackey–Glass)  
- Cointegration and error-correction models  
- State-space modelling and Kalman filtering  
- Real-world datasets (electricity, FX, macro indicators)  
- Optional: light-weight live forecasting pipeline with scheduled runs  

---

## Project Structure

```text
time-series-experiments/
├── src/
│   ├── simulators.py
│   ├── models_basic.py
│   ├── models_arima.py
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
├── data/
│   ├── simulated/
│   └── real/
├── outputs/
│   └── plots/
├── docs/
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
- runs walk-forward evaluation  
- stores plots in `outputs/plots/`  
- prints metrics and a short interpretation  

---

## Methodology

### Rolling-Origin Evaluation  
Expanding or sliding windows ensure no leakage and realistic forecasting conditions.

### Metrics  

- **MAE** — average absolute error  
- **RMSE** — penalises larger errors  
- **MAPE** — percentage error (careful near zero and sign changes)  

---

## Why Start With Synthetic Data?

Synthetic datasets make it possible to:

- Control the data-generating mechanism  
- Know in advance which model “should” work best  
- Diagnose why models fail or succeed  
- Learn structure before dealing with real-world noise  

Real datasets can later mirror these controlled settings.

---

## Dependencies

- numpy  
- pandas  
- matplotlib  
- statsmodels  

---

## License

MIT  
Free for educational and analytical use.
