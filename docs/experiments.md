# Experiments Overview

This document gives a short description for each experiment script in `src/experiments/`.

---

## `exp_ar1_horizons.py` – AR(1) Memory and Horizon

**What it does**

- Simulates an AR(1) process.
- Compares naive, mean, and ARIMA(1,0,0) models.
- Evaluates performance across different forecast horizons.

**What to look for**

- ARIMA(1,0,0) performs best at short horizons because it matches the true data-generating process.
- As horizon increases, all models deteriorate.
- The gap between a well-specified model and simple baselines narrows with horizon.

---

## `exp_trend_seasonal_compare.py` – Trend + Seasonality

**What it does**

- Generates a time series with both trend and seasonal components plus noise.
- Compares:
  - naive forecast
  - mean forecast
  - seasonal naive forecast
  - ARIMA (no seasonal terms)
  - SARIMA (with seasonal terms)

**What to look for**

- Seasonal naive often performs well due to the strong seasonal signal.
- SARIMA generally captures trend + seasonality better than standard ARIMA.
- Baseline models struggle to match seasonal turning points.

---

## `exp_regime_switch_compare.py` – Regime-Switching Volatility

**What it does**

- Simulates a process with constant mean but switching volatility regimes (e.g. low and high variance states).
- Compares naive, mean, and ARIMA(1,0,0) forecasts.

**What to look for**

- Mean forecast can be quite competitive because the true mean is constant.
- Naive forecast struggles when volatility jumps between regimes.
- ARIMA does not gain much, because the main structure lies in the variance, not the mean.

---

## `exp_garch_compare.py` – GARCH-like Volatility

**What it does**

- Generates synthetic returns with volatility clustering (GARCH-like behaviour).
- Compares:
  - naive forecast (copy last value)
  - mean forecast (near zero)
  - ARIMA(1,0,0)
  - a simple volatility-aware baseline for illustration

**What to look for**

- Level forecasts are difficult to improve beyond “mean zero”.
- Naive forecast tends to copy noise and performs poorly.
- The interesting structure is in volatility, not in the mean level.

---

## `exp_structural_breaks.py` – Structural Break in Mean

**What it does**

- Simulates a series with an AR(1)-type structure and a single break in the mean at a known time.
- Compares:
  - Global mean and global ARIMA (fitted on all available data)
  - Rolling mean and rolling ARIMA (windowed or local models)

**What to look for**

- Global models adapt slowly after the break because old data dominates.
- Rolling models adapt more quickly after the break by dropping older observations.
- Post-break metrics show the difference most clearly.

---

## `exp_var_basic.py` – Multivariate VAR(1)

**What it does**

- Simulates a 2D VAR(1) system (two series that influence each other).
- Fits:
  - Univariate ARIMA(1,0,0) to each series separately.
  - VAR(1) model to both series jointly.

**What to look for**

- When cross-dependence is strong, VAR(1) can outperform the separate ARIMA models.
- If cross-effects are weak, VAR and ARIMA may perform similarly.
- This mirrors many macro and financial settings where variables move together.

---

## `exp_logistic_chaos.py` – Logistic Map

**What it does**

- Simulates the logistic map in the chaotic regime.
- Uses ARIMA(1,0,0) to forecast the series at different horizons.

**What to look for**

- At short horizons, the model can sometimes track the local pattern.
- At longer horizons, forecasts no longer match the exact trajectory.
- Errors stabilise at a level that reflects predicting “typical” values, not exact paths.

---

## `exp_lorenz_chaos.py` – Lorenz Attractor

**What it does**

- Simulates the Lorenz system (x, y, z).
- Saves a 3D plot of the attractor.
- Uses the x-coordinate as a univariate time series and fits ARIMA(1,0,0) with multiple horizons.

**What to look for**

- Horizon 1: forecasts can occasionally follow local behaviour.
- Horizons 5 and 10: errors grow rapidly.
- The experiment shows fundamental limits of prediction in chaotic systems.
