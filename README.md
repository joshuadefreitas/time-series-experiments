# Time Series Forecasting Experiments

A comprehensive Python framework for evaluating and comparing time series forecasting models using rigorous statistical methodologies. This repository implements rolling-origin evaluation (time series cross-validation) to assess forecast accuracy across multiple models, from simple baselines to sophisticated ARIMA specifications.

## Overview

This project provides a modular, research-oriented toolkit for time series forecasting experiments. It implements standard forecast evaluation procedures following best practices in the forecasting literature, ensuring fair and reproducible model comparisons through proper temporal validation techniques.

## Features

- **Rolling-Origin Evaluation**: Implements walk-forward analysis (time series cross-validation) that respects temporal ordering and provides realistic out-of-sample performance assessment
- **Multiple Forecasting Models**:
  - Baseline methods: Naive (random walk) and Mean forecasts
  - ARIMA models: Flexible ARIMA(p,d,q) and SARIMA specifications via maximum likelihood estimation
- **Comprehensive Evaluation Metrics**: 
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- **Synthetic Data Generation**: Utilities for generating AR(1) processes for model validation and Monte Carlo experiments
- **Visualization**: Automated plotting of forecast vs. actual comparisons
- **Modular Architecture**: Clean separation of concerns for easy extension and experimentation

## Project Structure

```
time-series-forecasting/
├── src/
│   ├── simulators.py          # Time series data generation (AR processes)
│   ├── models_basic.py        # Baseline forecasting methods
│   ├── models_arima.py        # ARIMA/SARIMA model estimation and forecasting
│   ├── evaluation.py          # Rolling-origin evaluation and metrics
│   ├── plots.py               # Visualization utilities
│   └── experiments/
│       └── exp_ar1_compare.py # Example experiment comparing models
├── data/
│   ├── simulated/            # Generated synthetic time series
│   └── real/                 # Real-world datasets (placeholder)
├── outputs/
│   └── plots/                # Generated forecast comparison plots
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshuadefreitas/time-series-experiments.git
cd time-series-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Example Experiment

The included experiment compares naive, mean, and ARIMA(1,0,0) forecasts on a simulated AR(1) process:

```bash
python -m src.experiments.exp_ar1_compare
```

This will:
- Generate a synthetic AR(1) time series (φ=0.7, n=500)
- Evaluate each model using rolling-origin cross-validation
- Compute MAE, RMSE, and MAPE metrics
- Generate comparison plots
- Display a summary table ranking models by performance

### Example Output

```
MODEL COMPARISON SUMMARY
============================================================
Model                MAE          RMSE         MAPE        
------------------------------------------------------------
Naive                1.1820       1.4916       279.2519    
Mean                 1.1323       1.4438       105.4288    
ARIMA(1,0,0)         1.0541       1.3297       208.4083    

BEST MODEL BY METRIC:
------------------------------------------------------------
  MAE:  ARIMA(1,0,0) (1.0541)
  RMSE: ARIMA(1,0,0) (1.3297)
  MAPE: Mean (105.4288)
```

### Using the Framework

```python
from src.simulators import generate_ar1
from src.models_arima import arima_forecast
from src.evaluation import rolling_forecast_origin, compute_metrics

# Generate synthetic data
series = generate_ar1(n=500, phi=0.7, sigma=1.0, seed=42)

# Evaluate a model
results = rolling_forecast_origin(
    series=series,
    forecast_func=lambda s, h: arima_forecast(s, h, order=(1, 0, 0)),
    horizon=1,
    initial_window=100
)

# Compute metrics
metrics = compute_metrics(results)
print(metrics)  # {'MAE': 1.0541, 'RMSE': 1.3297, 'MAPE': 208.4083}
```

## Methodology

### Rolling-Origin Evaluation

The framework implements rolling-origin evaluation (also known as time series cross-validation or walk-forward analysis), which:

1. **Reserves initial training window**: Uses the first `initial_window` observations as the initial training set
2. **Expanding window approach**: For each time step `t` from `initial_window` to `T-h`:
   - Trains the model on all observations up to time `t`
   - Generates an `h`-step ahead forecast
   - Compares the forecast with the actual value at `t+h`
3. **Temporal integrity**: Ensures no future information leaks into past forecasts, providing realistic out-of-sample performance estimates

This approach is the gold standard for time series evaluation and is recommended in forecasting literature (Tashman, 2000; Hyndman & Athanasopoulos, 2021).

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute forecast error. Scale-dependent, robust to outliers.
- **RMSE (Root Mean Squared Error)**: Square root of mean squared errors. Penalizes large errors more heavily than MAE.
- **MAPE (Mean Absolute Percentage Error)**: Scale-invariant percentage error. Useful for comparing across series with different scales, but unreliable when values are close to zero.

## Results

The example experiment demonstrates that ARIMA(1,0,0) achieves the best performance on simulated AR(1) data, which is expected since it matches the true data-generating process. The framework enables systematic comparison of model performance across different scenarios and datasets.

## Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation and time series handling
- `scipy` - Statistical functions
- `statsmodels` - ARIMA model estimation
- `matplotlib` - Plotting and visualization

## References

- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts.

## License

This project is open source and available for educational and research purposes.

