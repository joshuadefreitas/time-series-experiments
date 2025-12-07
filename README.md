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
The focus is on understanding **why** models behave as they do — not simply comparing metrics.

---

## Executive Summary (for Recruiters)

This project demonstrates practical skills in:

- Forecasting & time‑series modeling  
- Econometrics and simulation  
- Rolling‑origin (walk‑forward) model evaluation  
- Clean, modular Python development  
- Insightful interpretation of model behaviour  

It serves as a hands-on sandbox for understanding forecasting challenges in finance, econometrics, and analytics.

---

## Purpose of the Project

The goal is to explore and answer questions like:

- When do simple models beat complex ones?
- How does forecast horizon change difficulty?
- Why do trends and seasonality matter so much?
- Why are financial returns so hard to predict?
- How do structural breaks affect forecasts?
- Can chaotic nonlinear systems be forecast at all?
- When does multivariate modeling actually help?

To investigate these questions, the project includes:

- Synthetic simulators  
- Rolling-origin evaluation  
- A library of forecasting models  
- Experiments with explanations and visuals  

---

## Current Capabilities

### 1. AR(1) Memory & Horizon Dependence  
How predictive power decays as the horizon increases.

### 2. Trend & Seasonality (SARIMA vs Baselines)  
Shows how explicit seasonal modeling improves forecasts.

### 3. Regime Switching (Variance Shifts)  
Demonstrates that structure often lies in volatility, not in the mean.

### 4. GARCH-like Volatility Clustering  
Synthetic returns with volatility persistence.

### 5. Structural Breaks  
Changepoints and how rolling vs global models adapt.

### 6. Multivariate Forecasting (VAR)  
When joint modeling beats separate univariate models.

### 7. Chaos Experiments  
- Logistic Map (1D chaos)  
- Lorenz Attractor (3D chaos projected to xₜ)

---

## Sample Plots

Plots are available in `outputs/plots/` inside the repository.

---

## Roadmap

### Phase 2 — Real Data
Applying the same methodology to:

- Energy demand  
- FX returns  
- Retail sales  
- Macro indicators  

### Phase 3 — Optional Live Forecasting Pipeline  
Automated ingestion and reporting.

---

## Project Structure

```
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

```bash
python -m src.experiments.exp_ar1_horizons
```

All experiments:

- generate synthetic data  
- run rolling-origin evaluation  
- compute MAE / RMSE / MAPE  
- save plots  
- print interpretation  

---

## Methodology

### Rolling-Origin Evaluation  
A proper forecasting evaluation method that avoids data leakage.

### Metrics  
- **MAE** — absolute error  
- **RMSE** — penalizes large errors  
- **MAPE** — percentage error (careful near zero)

---

## Why Synthetic Data?

Synthetic datasets make it possible to isolate specific forecasting challenges:

- Memory  
- Seasonality  
- Volatility  
- Structural breaks  
- Multivariate dynamics  
- Chaos  

Later, real datasets will mirror these controlled settings.

---

## Dependencies

- numpy  
- pandas  
- matplotlib  
- statsmodels  

---

## License

MIT License.
