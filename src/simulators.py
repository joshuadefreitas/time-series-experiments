"""
Time series simulation utilities.

Provides functions for generating synthetic univariate time series processes
following standard autoregressive specifications. Useful for Monte Carlo
simulations, model validation, and testing forecast methodologies.
"""

from typing import Optional

import numpy as np
import pandas as pd


def generate_ar1(n: int = 500, phi: float = 0.7, sigma: float = 1.0,
                 x0: float = 0.0, seed: Optional[int] = None) -> pd.Series:
    """
    Simulate a first-order autoregressive (AR(1)) process.
    
    Generates a univariate AR(1) time series following the specification:
        x_t = φ * x_{t-1} + ε_t
        where ε_t ~ i.i.d. N(0, σ²)
    
    The process is stationary if |φ| < 1. For φ = 1, it becomes a random walk.
    
    Args:
        n: Sample size (number of observations to generate).
        phi: Autoregressive coefficient. For stationarity, require |φ| < 1.
            Default 0.7 produces a moderately persistent stationary process.
        sigma: Standard deviation of innovation shocks ε_t. Default 1.0.
        x0: Initial value x_0 for the process. Default 0.0 (unconditional mean
            for stationary process with zero mean innovations).
        seed: Random seed for reproducibility. If None, uses system entropy.
    
    Returns:
        pandas Series with integer index [0, 1, ..., n-1] named "t" and
        series name indicating AR(1) parameter value.
    
    Example:
        >>> # Generate stationary AR(1) with persistence 0.8
        >>> series = generate_ar1(n=1000, phi=0.8, sigma=0.5, seed=42)
        >>> print(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
    """
    # Initialize random number generator for reproducible simulations
    rng = np.random.default_rng(seed)
    
    # Generate innovation sequence: white noise shocks
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    
    # Initialize array and set initial condition
    x = np.zeros(n)
    x[0] = x0
    
    # Recursively generate AR(1) process: x_t = φ * x_{t-1} + ε_t
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    
    # Return as pandas Series with named index for time axis
    return pd.Series(x, index=pd.RangeIndex(n, name="t"), name=f"ar1_phi_{phi}")