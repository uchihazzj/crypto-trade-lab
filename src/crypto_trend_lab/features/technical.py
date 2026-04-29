"""Technical indicators for crypto OHLCV data.

All indicators use only current and past observations. No future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TECHNICAL_FEATURE_COLUMNS = [
    "log_return_1",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "rolling_vol_24",
    "rolling_vol_72",
    "rolling_vol_168",
    "volume_change_24",
    "ma_7",
    "ma_25",
    "ma_99",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "bollinger_width",
    "atr_14",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a new DataFrame.

    Does not mutate *df*. Returns a DataFrame with only the technical
    feature columns. Columns with insufficient history are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Technical feature columns keyed to the same index.
    """
    result = pd.DataFrame(index=df.index)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns for technical features: {missing}"
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Returns ---
    result["log_return_1"] = np.log(close / close.shift(1))
    result["return_1"] = close.pct_change(1)
    result["return_3"] = close.pct_change(3)
    result["return_6"] = close.pct_change(6)
    result["return_12"] = close.pct_change(12)
    result["return_24"] = close.pct_change(24)

    # --- Rolling volatility (on log_return_1) ---
    log_ret = result["log_return_1"]
    result["rolling_vol_24"] = log_ret.rolling(24).std()
    result["rolling_vol_72"] = log_ret.rolling(72).std()
    result["rolling_vol_168"] = log_ret.rolling(168).std()

    # --- Volume change ---
    result["volume_change_24"] = volume.pct_change(24)

    # --- Moving averages ---
    result["ma_7"] = close.rolling(7).mean()
    result["ma_25"] = close.rolling(25).mean()
    result["ma_99"] = close.rolling(99).mean()
    result["ema_12"] = close.ewm(span=12, adjust=False).mean()
    result["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- RSI (Wilder smoothing, alpha = 1/14) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    result["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # --- MACD ---
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20-period, 2 std) ---
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper = ma_20 + 2.0 * std_20
    lower = ma_20 - 2.0 * std_20
    result["bollinger_width"] = (upper - lower) / ma_20

    # --- ATR (Wilder smoothing, alpha = 1/14) ---
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    return result
