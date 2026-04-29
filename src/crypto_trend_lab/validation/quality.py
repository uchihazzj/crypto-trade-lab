"""Data quality checks for OHLCV DataFrames."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.crypto_trend_lab.utils.helpers import (
    OHLCV_COLUMNS,
    timeframe_to_freq,
    validate_ohlcv_schema,
)


def check_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the OHLCV schema and return schema info."""
    try:
        validate_ohlcv_schema(df)
        valid = True
        missing: list[str] = []
        extra = [c for c in df.columns if c not in OHLCV_COLUMNS]
    except ValueError:
        valid = False
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in OHLCV_COLUMNS]

    return {
        "schema_valid": valid,
        "expected_columns": OHLCV_COLUMNS,
        "actual_columns": list(df.columns),
        "missing_columns": missing,
        "extra_columns": extra,
    }


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Count duplicate timestamps."""
    if df.empty or "timestamp" not in df.columns:
        return {"total_rows": len(df), "duplicate_timestamps": 0, "unique_timestamps": 0}
    dup_count = int(df["timestamp"].duplicated().sum())
    return {
        "total_rows": len(df),
        "duplicate_timestamps": dup_count,
        "unique_timestamps": len(df) - dup_count,
    }


def check_missing_bars(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Detect missing bars based on timeframe continuity.

    Returns the count of missing bars and a high-level summary.
    """
    if df.empty or "timestamp" not in df.columns:
        return {
            "missing_bar_count": 0,
            "expected_bar_count": 0,
            "freq": None,
            "min_timestamp": None,
            "max_timestamp": None,
        }

    freq = timeframe_to_freq(timeframe)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    expected_range = pd.date_range(start=ts_min, end=ts_max, freq=freq)
    expected_count = len(expected_range)
    actual_count = len(df["timestamp"].drop_duplicates())
    missing = max(0, expected_count - actual_count)

    return {
        "missing_bar_count": missing,
        "expected_bar_count": expected_count,
        "actual_bar_count": actual_count,
        "freq": freq,
        "min_timestamp": ts_min.isoformat() if ts_min is not pd.NaT else None,
        "max_timestamp": ts_max.isoformat() if ts_max is not pd.NaT else None,
    }


def check_nulls(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize null values per column."""
    if df.empty:
        return {"null_counts": {}, "total_rows": 0}
    null_series = df.isnull().sum()
    return {
        "null_counts": {str(k): int(v) for k, v in null_series.items()},
        "total_rows": len(df),
    }


def check_ohlcv_quality(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Run all data quality checks on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data to validate.
    timeframe : str
        CCXT timeframe string (e.g. '1h').

    Returns
    -------
    dict
        Combined quality report with keys: schema, duplicates, missing_bars, nulls,
        row_count, min_timestamp, max_timestamp.
    """
    schema = check_schema(df)
    duplicates = check_duplicates(df)
    missing_bars = check_missing_bars(df, timeframe)
    nulls = check_nulls(df)

    return {
        "row_count": len(df),
        "min_timestamp": (
            df["timestamp"].min().isoformat()
            if not df.empty and "timestamp" in df.columns and pd.notna(df["timestamp"].min())
            else None
        ),
        "max_timestamp": (
            df["timestamp"].max().isoformat()
            if not df.empty and "timestamp" in df.columns and pd.notna(df["timestamp"].max())
            else None
        ),
        "schema": schema,
        "duplicates": duplicates,
        "missing_bars": missing_bars,
        "nulls": nulls,
    }
