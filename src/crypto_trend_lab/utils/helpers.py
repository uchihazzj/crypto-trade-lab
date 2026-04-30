"""Shared helper functions for crypto-trend-lab."""

from __future__ import annotations

import pandas as pd

TIMEFRAME_TO_FREQ: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}

TIMEFRAME_TO_DURATION: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert a CCXT timeframe string to a pandas Timedelta.

    Parameters
    ----------
    timeframe : str
        CCXT timeframe string (e.g. '1h', '4h', '1d').

    Returns
    -------
    pd.Timedelta

    Raises
    ------
    KeyError
        If the timeframe is not in the known mapping.
    """
    if timeframe in TIMEFRAME_TO_DURATION:
        return TIMEFRAME_TO_DURATION[timeframe]
    raise KeyError(
        f"Unsupported timeframe: {timeframe!r}. "
        f"Supported: {sorted(TIMEFRAME_TO_DURATION.keys())}"
    )


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


def estimate_coverage(limit: int, timeframe: str) -> str:
    """Return a human-readable estimate of the duration covered by *limit* bars.

    Parameters
    ----------
    limit : int
        Number of OHLCV bars.
    timeframe : str
        CCXT timeframe string (e.g. '1h', '1d').

    Returns
    -------
    str
        Estimated coverage description, e.g. ``"~8 days, 8 hours"``.
    """
    try:
        td = timeframe_to_timedelta(timeframe)
    except KeyError:
        return f"{limit} bars (unknown timeframe {timeframe!r})"

    delta = limit * td
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60

    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        parts.append("< 1 minute")

    return f"~{', '.join(parts)}"


def dataset_sizing_warning(row_count: int, timeframe: str) -> str | None:
    """Return a warning if the dataset is too small for reliable modeling.

    Parameters
    ----------
    row_count : int
        Number of OHLCV rows after fetching.
    timeframe : str
        CCXT timeframe string.

    Returns
    -------
    str or None
        Warning message, or None if the dataset size is adequate.
    """
    if row_count < 1000:
        return (
            f"Only {row_count} rows fetched. "
            "This is sufficient for UI testing but too small for "
            "reliable model evaluation."
        )

    if timeframe in ("1h",) and row_count < 3000:
        return (
            f"Only {row_count} rows for {timeframe}. "
            "Weak for model evaluation; consider fetching more data."
        )

    if timeframe in ("4h", "1d") and row_count < 1000:
        return (
            f"Only {row_count} rows for {timeframe}. "
            "Weak for model evaluation; consider fetching more data."
        )

    return None
