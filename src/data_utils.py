from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Optional: live fetch via yfinance
try:
    import yfinance as yf

    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


def load_fx_returns(
    path: str | Path,
    price_col: str = "price",
    date_col: str = "date",
    tz: Optional[str] = None,
) -> pd.Series:
    """
    Load daily FX prices and return log returns as a DatetimeIndex series.

    Expected CSV schema:
        date, price
    where price is the close/mid. Returns are log(price_t) - log(price_{t-1}).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected FX price file at {path}. "
            "Place a CSV with columns [date, price] (or set price_col/date_col)."
        )

    df = pd.read_csv(path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{date_col}' and '{price_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.dropna(subset=[date_col])
    if tz:
        df[date_col] = df[date_col].dt.tz_convert(tz)

    # Coerce price to numeric and drop invalid rows
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df.sort_values(date_col)

    prices = df[price_col].astype(float)
    log_prices = np.log(prices)
    log_returns = log_prices.diff().dropna()

    returns_series = pd.Series(
        log_returns.values,
        index=df.loc[log_returns.index, date_col],
        name="log_return",
    )
    returns_series.index.name = "date"

    return returns_series


def fetch_fx_prices_yahoo(
    ticker: str = "EURUSD=X",
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch FX prices from Yahoo Finance using yfinance (optional dependency).
    Returns a DataFrame with columns [date, price].
    """
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is required to fetch live FX data. Install with `pip install yfinance`."
        )

    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    # Prefer Adj Close if available, otherwise Close
    price_col = None
    for col in ["Adj Close", "Close"]:
        if col in data.columns:
            price_col = col
            break
    if price_col is None:
        raise ValueError(f"Price column not found in downloaded data for {ticker}.")

    df = data.reset_index()[["Date", price_col]].rename(
        columns={"Date": "date", price_col: "price"}
    )
    return df
