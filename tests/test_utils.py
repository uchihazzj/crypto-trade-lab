"""Tests for crypto-trend-lab utility functions and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.crypto_trend_lab.utils.helpers import (
    OHLCV_COLUMNS,
    symbol_to_safe,
    timeframe_to_freq,
    validate_ohlcv_schema,
    normalize_ohlcv,
)
from src.crypto_trend_lab.validation.quality import (
    check_duplicates,
    check_missing_bars,
    check_schema,
    check_nulls,
    check_ohlcv_quality,
)


# ---------------------------------------------------------------------------
# symbol_to_safe
# ---------------------------------------------------------------------------


def test_symbol_to_safe_btc_usdt():
    assert symbol_to_safe("BTC/USDT") == "BTC_USDT"


def test_symbol_to_safe_eth_usdt():
    assert symbol_to_safe("ETH/USDT") == "ETH_USDT"


def test_symbol_to_safe_no_slash():
    assert symbol_to_safe("BTCUSDT") == "BTCUSDT"


# ---------------------------------------------------------------------------
# timeframe_to_freq
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tf, expected",
    [
        ("1m", "1min"),
        ("5m", "5min"),
        ("15m", "15min"),
        ("30m", "30min"),
        ("1h", "1h"),
        ("4h", "4h"),
        ("1d", "1D"),
        ("1w", "1W"),
    ],
)
def test_timeframe_to_freq_valid(tf, expected):
    assert timeframe_to_freq(tf) == expected


def test_timeframe_to_freq_invalid():
    with pytest.raises(KeyError, match="Unsupported timeframe"):
        timeframe_to_freq("2x")


# ---------------------------------------------------------------------------
# validate_ohlcv_schema
# ---------------------------------------------------------------------------


def test_validate_ohlcv_schema_valid():
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    validate_ohlcv_schema(df)  # should not raise


def test_validate_ohlcv_schema_missing():
    df = pd.DataFrame(columns=["timestamp", "open", "close"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_ohlcv_schema(df)


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


def test_check_schema_valid():
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = check_schema(df)
    assert result["schema_valid"] is True
    assert result["missing_columns"] == []


def test_check_schema_invalid():
    df = pd.DataFrame(columns=["timestamp", "close"])
    result = check_schema(df)
    assert result["schema_valid"] is False
    assert len(result["missing_columns"]) > 0


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------


def test_check_duplicates_none():
    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="utc")
    df = _make_ohlcv(ts)
    result = check_duplicates(df)
    assert result["duplicate_timestamps"] == 0
    assert result["unique_timestamps"] == 5


def test_check_duplicates_present():
    ts = pd.to_datetime(
        ["2024-01-01 00:00+00:00", "2024-01-01 00:00+00:00", "2024-01-01 01:00+00:00"]
    )
    df = _make_ohlcv(ts)
    result = check_duplicates(df)
    assert result["duplicate_timestamps"] == 1
    assert result["unique_timestamps"] == 2


def test_check_duplicates_empty():
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = check_duplicates(df)
    assert result["duplicate_timestamps"] == 0


# ---------------------------------------------------------------------------
# check_missing_bars
# ---------------------------------------------------------------------------


def test_check_missing_bars_none():
    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="utc")
    df = _make_ohlcv(ts)
    result = check_missing_bars(df, "1h")
    assert result["missing_bar_count"] == 0


def test_check_missing_bars_present():
    ts = pd.to_datetime(
        ["2024-01-01 00:00+00:00", "2024-01-01 02:00+00:00"]  # 01:00 missing
    )
    df = _make_ohlcv(ts)
    result = check_missing_bars(df, "1h")
    assert result["missing_bar_count"] == 1


def test_check_missing_bars_empty():
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = check_missing_bars(df, "1h")
    assert result["missing_bar_count"] == 0


# ---------------------------------------------------------------------------
# check_nulls
# ---------------------------------------------------------------------------


def test_check_nulls_none():
    ts = pd.date_range("2024-01-01", periods=3, freq="1h", tz="utc")
    df = _make_ohlcv(ts)
    result = check_nulls(df)
    for v in result["null_counts"].values():
        assert v == 0


def test_check_nulls_present():
    ts = pd.date_range("2024-01-01", periods=3, freq="1h", tz="utc")
    df = _make_ohlcv(ts)
    df.loc[1, "volume"] = None
    result = check_nulls(df)
    assert result["null_counts"]["volume"] == 1


# ---------------------------------------------------------------------------
# check_ohlcv_quality (integration)
# ---------------------------------------------------------------------------


def test_check_ohlcv_quality_integration():
    ts = pd.date_range("2024-01-01", periods=10, freq="1h", tz="utc")
    df = _make_ohlcv(ts)
    result = check_ohlcv_quality(df, "1h")
    assert result["row_count"] == 10
    assert result["min_timestamp"] is not None
    assert result["max_timestamp"] is not None
    assert result["schema"]["schema_valid"] is True
    assert result["duplicates"]["duplicate_timestamps"] == 0
    assert result["missing_bars"]["missing_bar_count"] == 0


# ---------------------------------------------------------------------------
# normalize_ohlcv
# ---------------------------------------------------------------------------


def test_normalize_ohlcv_drops_duplicates():
    ts = pd.to_datetime(
        ["2024-01-01 00:00+00:00", "2024-01-01 00:00+00:00", "2024-01-01 01:00+00:00"]
    )
    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, 1001, 1002],
        }
    )
    result = normalize_ohlcv(raw, exchange="binance", symbol="BTC/USDT", timeframe="1h")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# normalize_ohlcv — CCXT millisecond timestamp conversion
# ---------------------------------------------------------------------------


def test_normalize_ohlcv_converts_ms_timestamps():
    """1704067200000 ms must map to 2024-01-01 00:00:00+00:00, not 1970."""
    raw = pd.DataFrame(
        {
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
            "open": [42000.0, 42100.0, 42200.0],
            "high": [42500.0, 42600.0, 42700.0],
            "low": [41800.0, 41900.0, 42000.0],
            "close": [42300.0, 42400.0, 42500.0],
            "volume": [100.0, 110.0, 105.0],
        }
    )
    result = normalize_ohlcv(
        raw, exchange="binance", symbol="BTC/USDT", timeframe="1h"
    )

    expected_ts = pd.Timestamp("2024-01-01 00:00:00", tz="utc")
    assert result["timestamp"].iloc[0] == expected_ts
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-01-01 01:00:00", tz="utc")
    assert result["timestamp"].iloc[2] == pd.Timestamp("2024-01-01 02:00:00", tz="utc")

    # Ensure year is 2024, not 1970
    assert result["timestamp"].dt.year.iloc[0] == 2024

    # Schema must be intact
    assert list(result.columns) == OHLCV_COLUMNS


def test_normalize_ohlcv_timestamp_dtype_is_utc_datetime64():
    """Timestamp column must be timezone-aware UTC datetime64."""
    raw = pd.DataFrame(
        {
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41800.0],
            "close": [42300.0],
            "volume": [100.0],
        }
    )
    result = normalize_ohlcv(
        raw, exchange="binance", symbol="BTC/USDT", timeframe="1h"
    )

    dtype = result["timestamp"].dtype
    assert dtype.kind == "M"  # datetime64
    assert getattr(dtype, "tz", None) is not None  # timezone-aware


def test_normalize_ohlcv_handles_already_datetime():
    """normalize_ohlcv must be idempotent when timestamps are already datetime."""
    ts = pd.to_datetime(
        ["2024-01-01 00:00+00:00", "2024-01-01 01:00+00:00"]
    )
    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41800.0, 41900.0],
            "close": [42300.0, 42400.0],
            "volume": [100.0, 110.0],
        }
    )
    result = normalize_ohlcv(
        raw, exchange="binance", symbol="BTC/USDT", timeframe="1h"
    )

    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="utc")
    assert result["timestamp"].dtype.kind == "M"
    assert getattr(result["timestamp"].dtype, "tz", None) is not None


def test_fetcher_style_ohlcv_rows_with_ms_timestamps():
    """Simulate the exact row format that CCXT fetch_ohlcv returns."""
    ms_samples = [
        1704067200000,  # 2024-01-01 00:00:00 UTC
        1704153600000,  # 2024-01-02 00:00:00 UTC
        1704240000000,  # 2024-01-03 00:00:00 UTC
        1704326400000,  # 2024-01-04 00:00:00 UTC
        1704412800000,  # 2024-01-05 00:00:00 UTC
    ]

    records = []
    for ms in ms_samples:
        records.append([ms, 40000.0, 41000.0, 39000.0, 40500.0, 500.0])

    df = pd.DataFrame(
        records,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    result = normalize_ohlcv(
        df, exchange="binance", symbol="BTC/USDT", timeframe="1d"
    )

    assert len(result) == 5
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz="utc")
    assert result["timestamp"].iloc[4] == pd.Timestamp("2024-01-05", tz="utc")
    assert list(result.columns) == OHLCV_COLUMNS
    assert result["timestamp"].dtype.kind == "M"
    assert getattr(result["timestamp"].dtype, "tz", None) is not None
    assert result["exchange"].iloc[0] == "binance"
    assert result["symbol"].iloc[0] == "BTC/USDT"
    assert result["timeframe"].iloc[0] == "1d"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(timestamps: pd.DatetimeIndex | list) -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame for testing."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": pd.DatetimeIndex(timestamps),
            "open": [100.0 + i for i in range(n)],
            "high": [105.0 + i for i in range(n)],
            "low": [95.0 + i for i in range(n)],
            "close": [103.0 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )
