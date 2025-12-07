# Key Lessons from the Time Series Experiments

This document summarises the main ideas behind the experiments in the repository and gives small examples of how to use the core API.

---

## 1. Memory and Horizon – AR(1) Processes

**Idea**  
A simple AR(1) process has short memory. A correctly specified AR(1) / ARIMA(1,0,0) model performs well at short horizons, but forecast quality naturally decays as the horizon grows.

**Key points**

- Forecast horizon directly affects difficulty.
- Even a correctly specified model performs worse the further you look ahead.
- For long horizons, all reasonable models tend to drift towards the unconditional mean.

**Related script**

- `src/experiments/exp_ar1_horizons.py`

**Example (using the library)**

```python
from src.simulators import generate_ar1
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics

series = generate_ar1(n=500, phi=0.7, sigma=1.0, seed=42)

results = rolling_forecast_origin(
    series=series,
    forecast_func=lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
    horizon=1,
    initial_window=100,
)

metrics = compute_metrics(results)
print(metrics)
```

---

## 2. Trend and Seasonality – Why SARIMA Matters

**Idea**  
When a series has trend and seasonality, models that ignore these patterns will struggle. Seasonal naive and SARIMA models are often strong baselines.

**Key points**

- Seasonal naive can work surprisingly well when the pattern is stable.
- Plain ARIMA without seasonal terms can miss structure.
- SARIMA (seasonal ARIMA) explicitly captures seasonal differences and often improves accuracy.

**Related script**

- `src/experiments/exp_trend_seasonal_compare.py`

---

## 3. Regime Switching – Structure in the Variance

**Idea**  
Some processes have constant mean but changing variance (volatility). In this case, there is very little to forecast in the level, but a lot to understand in the volatility.

**Key points**

- The best mean forecast can still be “just the mean”.
- Volatility regimes (calm vs turbulent periods) matter more than the exact level.
- This motivates volatility and regime models such as GARCH and Markov switching.

**Related script**

- `src/experiments/exp_regime_switch_compare.py`

---

## 4. GARCH-like Volatility – Returns vs Risk

**Idea**  
Financial returns are often close to unforecastable in direction, but their volatility is not. Volatility tends to cluster: high-volatility periods follow each other.

**Key points**

- Returns may look like noise centred around zero.
- Mean forecasts (including ARIMA) may not beat a simple zero-mean benchmark.
- Volatility, however, shows persistence and structure.

**Related script**

- `src/experiments/exp_garch_compare.py`

---

## 5. Structural Breaks – When History Misleads You

**Idea**  
If the process has a structural break (e.g. a change in mean level), models fitted on the full history adapt slowly. Local or rolling models can adapt faster by “forgetting” old regimes.

**Key points**

- Global models use all past data and are pulled towards old regimes.
- Rolling or windowed models adapt faster after a break.
- Post-break metrics are more informative than overall metrics.

**Related script**

- `src/experiments/exp_structural_breaks.py`

---

## 6. Multivariate Dynamics – When VAR Helps

**Idea**  
When multiple series influence each other, modelling them jointly can improve forecasts compared to separate univariate models.

**Key points**

- Univariate ARIMA models ignore cross-series interactions.
- VAR (vector autoregression) uses lagged values of all series.
- If one variable helps predict another, VAR can outperform separate univariate models.

**Related script**

- `src/experiments/exp_var_basic.py`

---

## 7. Chaos and Nonlinear Dynamics – Limits of Prediction

### 7.1 Logistic Map

**Idea**  
The logistic map is a simple nonlinear recurrence that can be chaotic for certain parameter values. Even though the system is deterministic, long-horizon prediction is very limited.

**Key points**

- Short-horizon forecasts can follow the local shape reasonably well.
- For longer horizons, forecasts stop tracking the exact path and gravitate towards “typical” values.
- Models effectively approximate the invariant distribution rather than the path itself.

**Related script**

- `src/experiments/exp_logistic_chaos.py`

---

### 7.2 Lorenz System

**Idea**  
The Lorenz system is a classic continuous-time chaotic system. Projecting the 3D trajectory onto a single coordinate (xₜ) yields a complicated time series.

**Key points**

- At very short horizons, a simple model may occasionally follow the local path.
- As horizon increases, MAE and RMSE grow sharply.
- This is not just model failure; it reflects a fundamental limit of predictability driven by sensitivity to initial conditions.

**Related script**

- `src/experiments/exp_lorenz_chaos.py`

---

## 8. Plugging in Your Own Model

Any forecasting function with the signature:

```python
def my_forecast(series, horizon) -> float:
    ...
```

can be used with `rolling_forecast_origin`.

**Example: simple moving-average forecaster**

```python
def last_k_mean(series, horizon, k=20):
    window = series.iloc[-k:] if len(series) > k else series
    return float(window.mean())
```

Use it as:

```python
from src.evaluation import rolling_forecast_origin, compute_metrics

res = rolling_forecast_origin(
    series=series,
    forecast_func=lambda s, h: last_k_mean(s, h, k=20),
    horizon=1,
    initial_window=100,
)

print(compute_metrics(res))
```
