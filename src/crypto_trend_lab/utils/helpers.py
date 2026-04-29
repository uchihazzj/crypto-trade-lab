"""Shared helper functions for crypto-trend-lab."""

from __future__ import annotations

import pandas as pd

TIMEFRAME_TO_FREQ: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def symbol_to_safe(symbol: str) -> str:
    """Convert a trading symbol like BTC/USDT to a safe directory name BTC_USDT."""
    return symbol.replace("/", "_")


def timeframe_to_freq(timeframe: str) -> str:
    """Convert a CCXT timeframe like '1h' to a pandas frequency string like '1h'.

    Raises KeyError if the timeframe is not in the known mapping.
    """
    try:
        return TIMEFRAME_TO_FREQ[timeframe]
    except KeyError:
        raise KeyError(
            f"Unsupported timeframe: {timeframe!r}. "
            f"Supported: {list(TIMEFRAME_TO_FREQ.keys())}"
        )


OHLCV_COLUMNS = [
    "exchange",
    "symbol",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* does not match the stable OHLCV schema."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing}")


def normalize_ohlcv(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Normalize raw CCXT OHLCV data into the stable schema.

    Drops duplicate timestamps, sorts by timestamp, converts to UTC, and
    ensures the standard column set is present.
    """
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.copy()
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    df.rename(
        columns={
            "datetime": "timestamp",
            "timestamp_open": "timestamp",
        },
        inplace=True,
    )

    if "timestamp" not in df.columns:
        raise ValueError("Cannot find timestamp column in OHLCV data")

    # CCXT returns Unix milliseconds (e.g. 1704067200000).
    # If timestamps are already datetime, just convert to UTC.
    if df["timestamp"].dtype.kind in ("i", "f"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} missing after normalization")

    return df[OHLCV_COLUMNS]
