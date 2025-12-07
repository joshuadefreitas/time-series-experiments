# Time Series Forecasting Experiments

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Focus](https://img.shields.io/badge/focus-time--series%20%7C%20econometrics%20%7C%20simulation-orange)

*A research‑style forecasting framework exploring how models behave under different time-series structures.*

A structured Python framework for exploring, testing, and understanding forecasting models across different types of time-series behavior. The goal is not only to compare forecasting models, but to understand **how** and **why** they succeed or fail under different data‑generating mechanisms.

The repository is organized like a miniature research pipeline: each experiment highlights a key forecasting concept — from memory and seasonality to volatility, structural breaks, multivariate interactions, and nonlinear dynamics.

---

## Table of Contents

* [Purpose](#purpose-of-the-project)
* [Current Experiments](#current-capabilities)
* [Sample Plots](#sample-plots)
* [Roadmap](#roadmap-upcoming-milestones)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running Experiments](#running-experiments)
* [Methodology](#methodology)
* [Why Synthetic First](#why-start-with-synthetic-data)
* [Dependencies](#dependencies)
* [License](#license)

---

## Purpose of the Project

This project helps answer foundational forecasting questions:

* When do simple models outperform complex ones?
* How does forecast accuracy change with the horizon?
* How do trend and seasonality affect model behavior?
* Why are financial returns so difficult to predict?
* How do regime shifts and structural breaks impact models?
* Can chaotic systems be forecast at all?

To explore these ideas, the framework includes:

* Controlled **synthetic simulators**
* Clean **rolling-origin (walk-forward) evaluation**
* Interpretable **metrics and visualizations**
* Modular experiments ready to extend to real datasets

---

## Current Capabilities

Here are the core experiments implemented so far — each demonstrating a fundamental forecasting concept.

### Completed Experiments

#### **1. AR(1) Memory + Horizon Dependence**

Illustrates how short-term memory affects forecasting across different horizons.

#### **2. Trend & Seasonality (SARIMA vs Baselines)**

Shows why explicit seasonal modeling matters, and when simple baselines perform well.

#### **3. Volatility Regimes (Mean-Stable, Variance-Unstable)**

Demonstrates that structure often lives in volatility rather than levels.

#### **4. GARCH-like Volatility Clustering**

Synthetic financial returns exhibiting persistent volatility. Highlights why volatility forecasting is often more meaningful than predicting direction.

Each experiment includes:

* Synthetic data generation
* Model comparison
* Rolling-origin evaluation
* Clear, interpretable explanations

---

## Sample Plots

Below are a few examples of plots generated directly from the experiments:

### **Trend + Seasonality (SARIMA vs Baselines)**

![Trend Seasonality](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/trend_seasonal_sarima_vs_actual.png)

### **AR(1) Horizon Differences**

![AR1 Horizons](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/ar1_h10_vs_actual.png)

### **Regime Switching Process**

![Regime Switching](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/regime_switching_mean_vs_actual.png)

### **GARCH-like Volatility Clustering**

![GARCH Example](https://raw.githubusercontent.com/joshuadefreitas/time-series-experiments/main/outputs/plots/garch_mean_forecast.png)

These visuals help communicate *how* models deviate from reality under different assumptions — a central lesson in applied forecasting.

---

## Roadmap: Upcoming Milestones

### **Milestone 5 — Structural Breaks**

Study how models behave when the underlying mean shifts. Compare long-window vs short-window approaches.

### **Milestone 6 — Multivariate Forecasting (VAR)**

Explore cross‑series dependence and forecast vector-valued processes.

### **Milestone 7 — Nonlinear Dynamics & Chaos**

Use logistic maps and Lorenz-type systems to show limits of predictability.

### **Phase 2 — Real Data Experiments**

Once synthetic foundations are complete, mirror the experiments using real datasets (e.g., FX returns, electricity demand, macro-economic variables).

### **Phase 3 — Optional Live Forecasting Pipeline**

Add a lightweight engineering layer for automatic data ingestion and scheduled forecasting.

---

## Project Structure

```text
time-series-forecasting/
├── src/
│   ├── simulators.py          # Synthetic data generators
│   ├── models_basic.py        # Baseline forecasting models
│   ├── models_arima.py        # ARIMA & SARIMA forecasting
│   ├── evaluation.py          # Rolling-origin evaluation & metrics
│   ├── plots.py               # Visualization utilities
│   └── experiments/
│       ├── exp_ar1_horizons.py
│       ├── exp_trend_seasonal_compare.py
│       ├── exp_regime_switch_compare.py
│       └── exp_garch_compare.py
├── data/
│   ├── simulated/             # Generated synthetic datasets
│   └── real/                  # Real-world datasets (future)
├── outputs/
│   └── plots/                 # Forecast comparison visuals
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

All experiments follow the same pattern:

* Generate synthetic data
* Evaluate models using walk‑forward validation
* Compute MAE, RMSE, MAPE
* Produce forecast‑vs‑actual plots
* Print a structured interpretation

---

## Methodology

### **Rolling-Origin Evaluation (Walk-Forward Validation)**

Ensures a realistic and unbiased evaluation by:

1. Training on an initial window
2. Forecasting ahead
3. Expanding or sliding the window
4. Repeating across the full series

This prevents look-ahead bias and mirrors real-world deployment.

### **Evaluation Metrics**

* **MAE** — Average absolute error
* **RMSE** — Penalizes larger errors
* **MAPE** — Percentage error (with caution near zero values)

---

## Why Start with Synthetic Data?

Synthetic experiments provide:

* Full control over the data-generating mechanism
* Clear illustrations of concepts
* Ability to replicate, stress-test, and isolate model behavior

Real datasets will be added to mirror these controlled scenarios.

---

## Dependencies

* numpy
* pandas
* matplotlib
* statsmodels

---

## License

Open-source for research and educational use.
