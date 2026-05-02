"""Tests for feature engineering and prediction targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    add_technical_features,
)
from src.crypto_trend_lab.features.target import (
    TARGET_COLUMNS,
    add_prediction_targets,
)
from src.crypto_trend_lab.features.pipeline import (
    build_features,
    get_model_input_columns,
    get_ohlcv_columns,
    get_target_columns,
    get_technical_feature_columns,
)
from src.crypto_trend_lab.storage.parquet import (
    _build_features_path,
    load_features_parquet,
    save_features_parquet,
)
from src.crypto_trend_lab.utils.helpers import OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 100) -> pd.DataFrame:
    """Build a deterministic valid OHLCV DataFrame with *n* 1h rows."""
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": ts,
            "open": [40000.0 + i for i in range(n)],
            "high": [40100.0 + i for i in range(n)],
            "low": [39900.0 + i for i in range(n)],
            "close": [40050.0 + i for i in range(n)],
            "volume": [100.0 + i for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# add_technical_features — immutability
# ---------------------------------------------------------------------------


def test_add_technical_features_does_not_mutate_input():
    df = _make_ohlcv_df(50)
    original = df.copy()
    _ = add_technical_features(df)
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# add_technical_features — column existence
# ---------------------------------------------------------------------------


def test_add_technical_features_produces_all_columns():
    df = _make_ohlcv_df(120)
    result = add_technical_features(df)
    for col in TECHNICAL_FEATURE_COLUMNS:
        assert col in result.columns, f"Missing technical column: {col}"
    # No extra columns beyond TECHNICAL_FEATURE_COLUMNS
    assert list(result.columns) == TECHNICAL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# add_prediction_targets — column existence
# ---------------------------------------------------------------------------


def test_add_prediction_targets_produces_all_columns():
    df = _make_ohlcv_df(50)
    result = add_prediction_targets(df)
    for col in TARGET_COLUMNS:
        assert col in result.columns, f"Missing target column: {col}"
    assert list(result.columns) == TARGET_COLUMNS


# ---------------------------------------------------------------------------
# target_return_h — correct future shift
# ---------------------------------------------------------------------------


def test_target_return_1_uses_1_step_ahead():
    """target_return_1[t] = ln(close[t+1] / close[t])"""
    df = _make_ohlcv_df(10)
    targets = add_prediction_targets(df)

    close = df["close"].values
    for t in range(9):
        expected = np.log(close[t + 1] / close[t])
        actual = targets["target_return_1"].iloc[t]
        assert np.isclose(actual, expected), f"Mismatch at t={t}"

    # Last row has no t+1
    assert pd.isna(targets["target_return_1"].iloc[9])


def test_target_return_4_uses_4_steps_ahead():
    """target_return_4[t] = ln(close[t+4] / close[t])"""
    df = _make_ohlcv_df(10)
    targets = add_prediction_targets(df)

    close = df["close"].values
    for t in range(6):
        expected = np.log(close[t + 4] / close[t])
        actual = targets["target_return_4"].iloc[t]
        assert np.isclose(actual, expected), f"Mismatch at t={t}"

    # Last 4 rows have no t+4
    for t in range(6, 10):
        assert pd.isna(targets["target_return_4"].iloc[t])


def test_target_return_24_uses_24_steps_ahead():
    """target_return_24[t] = ln(close[t+24] / close[t])"""
    df = _make_ohlcv_df(50)
    targets = add_prediction_targets(df)

    close = df["close"].values
    for t in range(25):
        expected = np.log(close[t + 24] / close[t])
        actual = targets["target_return_24"].iloc[t]
        assert np.isclose(actual, expected), f"Mismatch at t={t}"

    # Last 24 rows NaN
    for t in range(26, 50):
        assert pd.isna(targets["target_return_24"].iloc[t])


# ---------------------------------------------------------------------------
# target_direction_h — consistent with target_return_h
# ---------------------------------------------------------------------------


def test_target_direction_matches_return_sign():
    """target_direction_h must be 1 iff target_return_h > 0, else 0."""
    df = _make_ohlcv_df(50)
    targets = add_prediction_targets(df)

    for h in (1, 4, 24):
        ret_col = f"target_return_{h}"
        dir_col = f"target_direction_{h}"

        for idx in range(50 - h):
            ret_val = targets[ret_col].iloc[idx]
            dir_val = targets[dir_col].iloc[idx]
            if ret_val > 0:
                assert dir_val == 1.0, f"Direction wrong at idx={idx}, h={h}"
            else:
                assert dir_val == 0.0, f"Direction wrong at idx={idx}, h={h}"

        # NaN rows: direction should also be NaN (not 0 or 1)
        for idx in range(50 - h, 50):
            assert pd.isna(targets[dir_col].iloc[idx])


# ---------------------------------------------------------------------------
# Rolling features — current and past values only
# ---------------------------------------------------------------------------


def test_rolling_features_use_only_past_values():
    """Modifying future rows must not change features at earlier indices."""
    df = _make_ohlcv_df(60)

    features_original = add_technical_features(df)

    # Modify rows 50+ drastically
    df_mod = df.copy()
    df_mod.loc[50:, "close"] = 1.0
    df_mod.loc[50:, "open"] = 1.0
    df_mod.loc[50:, "high"] = 1.0
    df_mod.loc[50:, "low"] = 1.0
    df_mod.loc[50:, "volume"] = 1.0

    features_mod = add_technical_features(df_mod)

    # Features at index 40 (well before the modification) should be identical
    for col in TECHNICAL_FEATURE_COLUMNS:
        val_orig = features_original[col].iloc[40]
        val_mod = features_mod[col].iloc[40]
        if pd.isna(val_orig) and pd.isna(val_mod):
            continue
        assert val_orig == val_mod, (
            f"{col} at index 40 differs after future modification"
        )


def test_rolling_vol_uses_only_past_returns():
    """rolling_vol_24[t] must be computed from log_return_1[t-23:t+1] only."""
    df = _make_ohlcv_df(50)
    features = add_technical_features(df)

    # rolling_vol_24 at index 24 depends on indices 1..24 of log_return_1
    # (log_return_1[0] is NaN since close.shift(1) at 0 is NaN)
    # Verify it doesn't depend on index 25+
    expected = features["log_return_1"].iloc[1:25].std(ddof=0)
    actual = features["rolling_vol_24"].iloc[24]
    # Use ddof=0 because pandas .std() defaults to ddof=1
    expected_ddof1 = features["log_return_1"].iloc[1:25].std()
    assert np.isclose(actual, expected_ddof1), (
        f"rolling_vol_24[24] = {actual}, expected {expected_ddof1}"
    )


# ---------------------------------------------------------------------------
# build_features — preserves OHLCV columns
# ---------------------------------------------------------------------------


def test_build_features_preserves_ohlcv_columns():
    df = _make_ohlcv_df(100)
    result = build_features(df)

    for col in OHLCV_COLUMNS:
        assert col in result.columns, f"Missing OHLCV column: {col}"

    # OHLCV values unchanged
    pd.testing.assert_series_equal(
        result["close"], df["close"], check_names=False
    )
    pd.testing.assert_series_equal(
        result["timestamp"], df["timestamp"], check_names=False
    )


def test_build_features_does_not_mutate_input():
    df = _make_ohlcv_df(100)
    original = df.copy()
    _ = build_features(df)
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# build_features — NaN handling
# ---------------------------------------------------------------------------


def test_target_nan_for_unavailable_future():
    """The final h rows of target_return_h must be NaN."""
    df = _make_ohlcv_df(50)
    result = build_features(df)

    # target_return_1: last row NaN
    assert pd.isna(result["target_return_1"].iloc[49])
    assert not pd.isna(result["target_return_1"].iloc[48])

    # target_return_4: last 4 rows NaN
    for i in range(46, 50):
        assert pd.isna(result["target_return_4"].iloc[i])
    assert not pd.isna(result["target_return_4"].iloc[45])

    # target_return_24: last 24 rows NaN
    for i in range(26, 50):
        assert pd.isna(result["target_return_24"].iloc[i])
    assert not pd.isna(result["target_return_24"].iloc[25])


# ---------------------------------------------------------------------------
# build_features — insufficient rows
# ---------------------------------------------------------------------------


def test_build_features_handles_few_rows():
    """Pipeline must not crash on small DataFrames."""
    df = _make_ohlcv_df(5)
    result = build_features(df)

    assert len(result) == 5
    # All technical and target columns should exist
    for col in TECHNICAL_FEATURE_COLUMNS:
        assert col in result.columns
    for col in TARGET_COLUMNS:
        assert col in result.columns
    # OHLCV columns preserved
    for col in OHLCV_COLUMNS:
        assert col in result.columns


def test_build_features_handles_empty_df():
    """Pipeline must return an empty DataFrame with all columns."""
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = build_features(df)

    assert result.empty
    for col in OHLCV_COLUMNS:
        assert col in result.columns
    for col in TECHNICAL_FEATURE_COLUMNS:
        assert col in result.columns
    for col in TARGET_COLUMNS:
        assert col in result.columns


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def test_get_ohlcv_columns():
    assert get_ohlcv_columns() == OHLCV_COLUMNS


def test_get_technical_feature_columns():
    assert get_technical_feature_columns() == TECHNICAL_FEATURE_COLUMNS


def test_get_target_columns():
    assert get_target_columns() == TARGET_COLUMNS


def test_get_model_input_columns_excludes_targets():
    inputs = get_model_input_columns()
    for t in TARGET_COLUMNS:
        assert t not in inputs, f"{t} must not be a model input"


def test_get_model_input_columns_excludes_metadata():
    inputs = get_model_input_columns()
    for c in ("exchange", "symbol", "timeframe", "timestamp"):
        assert c not in inputs, f"Metadata column {c} must not be a model input"


def test_get_model_input_columns_includes_numeric_ohlcv():
    inputs = get_model_input_columns()
    for c in ("open", "high", "low", "close", "volume"):
        assert c in inputs, f"Numeric OHLCV column {c} must be a model input"


# ---------------------------------------------------------------------------
# Feature storage — path convention
# ---------------------------------------------------------------------------


def test_build_features_path_convention():
    path = _build_features_path("binance", "BTC/USDT", "1h")
    path_str = str(path).replace("\\", "/")

    assert "data/features" in path_str
    assert "exchange=binance" in path_str
    assert "symbol=BTC_USDT" in path_str
    assert "timeframe=1h" in path_str
    assert path_str.endswith("features.parquet")


def test_save_and_load_features_roundtrip(tmp_path, monkeypatch):
    """Save a feature DataFrame and load it back."""
    import src.crypto_trend_lab.storage.parquet as pmod

    monkeypatch.setattr(pmod, "DATA_DIR", tmp_path)

    df = _make_ohlcv_df(100)
    features = build_features(df)

    path = save_features_parquet(features, "binance", "BTC/USDT", "1h")
    assert path.exists()

    loaded = load_features_parquet("binance", "BTC/USDT", "1h")
    assert len(loaded) == 100

    # Round-trip: core OHLCV values preserved
    pd.testing.assert_series_equal(
        loaded["close"], features["close"], check_names=False
    )
    # Technical features preserved
    pd.testing.assert_series_equal(
        loaded["rsi_14"], features["rsi_14"], check_names=False, check_index=False
    )
    # Targets preserved
    for col in TARGET_COLUMNS:
        pd.testing.assert_series_equal(
            loaded[col], features[col], check_names=False, check_index=False
        )


def test_load_features_missing_file_returns_empty():
    df = load_features_parquet("nonexistent", "BTC/USDT", "1h")
    assert df.empty
