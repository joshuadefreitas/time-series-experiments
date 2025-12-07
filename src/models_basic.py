"""
Baseline forecasting methods.

Implements simple benchmark forecasting methods commonly used as baselines
in forecast evaluation. These methods serve as null models to assess whether
more sophisticated approaches provide meaningful improvements.

References:
    Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and
    practice (3rd ed.). OTexts.
"""

import pandas as pd


def naive_forecast(series: pd.Series, horizon: int = 1) -> float:
    """
    Naive (random walk) forecast: ŷ_{T+h} = y_T.
    
    Uses the last observed value as the forecast for all horizons. This is
    the optimal forecast for a random walk process and serves as a standard
    benchmark in forecast competitions.
    
    Args:
        series: Historical time series observations.
        horizon: Forecast horizon (unused; included for API consistency).
    
    Returns:
        Point forecast (last observed value) as float.
    
    Raises:
        ValueError: If series is empty.
    
    Note:
        Equivalent to assuming y_t follows a random walk: y_t = y_{t-1} + ε_t.
    """
    if series.empty:
        raise ValueError("Series is empty")
    return float(series.iloc[-1])


def mean_forecast(series: pd.Series, horizon: int = 1) -> float:
    """
    Mean (constant) forecast: ŷ_{T+h} = ȳ = (1/T)Σ_{t=1}^T y_t.
    
    Uses the sample mean of historical observations as the forecast. Optimal
    for a constant mean process with white noise errors.
    
    Args:
        series: Historical time series observations.
        horizon: Forecast horizon (unused; included for API consistency).
    
    Returns:
        Point forecast (historical mean) as float.
    
    Raises:
        ValueError: If series is empty.
    
    Note:
        Assumes process has constant mean with no trend or structural breaks.
    """
    if series.empty:
        raise ValueError("Series is empty")
    return float(series.mean())


def seasonal_naive_forecast(series: pd.Series, season_length: int,
                            horizon: int = 1) -> float:
    """
    Seasonal naive forecast: ŷ_{T+h} = y_{T+h-s}.
    
    Uses the value from one full seasonal cycle ago. Optimal forecast for
    seasonal random walk processes. Standard benchmark for seasonal data.
    
    Args:
        series: Historical time series observations with seasonal patterns.
        season_length: Length of seasonal cycle (e.g., 12 for monthly data
            with annual seasonality, 4 for quarterly data, 7 for weekly data
            with daily seasonality).
        horizon: Forecast horizon (unused; included for API consistency).
    
    Returns:
        Point forecast (value from s periods ago) as float.
    
    Raises:
        ValueError: If series length is less than one seasonal cycle.
    
    Example:
        >>> # Monthly data with annual seasonality
        >>> forecast = seasonal_naive_forecast(series, season_length=12)
    
    Note:
        For h > s, this repeats the value from T+h-s rather than from
        the most recent occurrence of that seasonal position.
    """
    if len(series) < season_length:
        raise ValueError("Series length smaller than one season.")
    # Extract value from one full seasonal cycle ago
    return float(series.iloc[-season_length])