"""Tests for milestone 3: evaluation, models, dataset, and storage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.crypto_trend_lab.features.pipeline import build_features
from src.crypto_trend_lab.evaluation.split import (
    chronological_train_test_split,
    walk_forward_split,
)
from src.crypto_trend_lab.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from src.crypto_trend_lab.evaluation.report import (
    compare_baselines_and_models,
    evaluate_model,
)
from src.crypto_trend_lab.models.baseline import (
    LastReturnBaseline,
    MajorityClassBaseline,
    MomentumDirectionBaseline,
    MovingAverageReturnBaseline,
    ZeroReturnBaseline,
)
from src.crypto_trend_lab.models.dataset import (
    get_default_feature_columns,
    get_target_column,
    prepare_modeling_data,
)
from src.crypto_trend_lab.models.tabular import (
    LogisticRegressionModel,
    RidgeRegressionModel,
)
from src.crypto_trend_lab.storage.parquet import (
    _build_predictions_path,
    load_predictions_parquet,
    save_predictions_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 200) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame."""
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    rng = np.random.default_rng(42)
    prices = 40000 + np.cumsum(rng.normal(0, 50, n))
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": ts,
            "open": prices - 10,
            "high": prices + 20,
            "low": prices - 20,
            "close": prices,
            "volume": rng.random(n) * 100 + 100,
        }
    )


def _make_features_df(n: int = 200) -> pd.DataFrame:
    """Build a feature DataFrame from deterministic OHLCV data."""
    df = _make_ohlcv_df(n)
    return build_features(df)


# ---------------------------------------------------------------------------
# chronological_train_test_split
# ---------------------------------------------------------------------------


def test_chronological_split_preserves_order():
    df = _make_features_df(100)
    train, test = chronological_train_test_split(df, test_size=20)

    assert len(train) == 80
    assert len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()


def test_chronological_split_no_overlap():
    df = _make_features_df(100)
    train, test = chronological_train_test_split(df, test_size=0.2)

    train_ts = set(train["timestamp"])
    test_ts = set(test["timestamp"])
    assert train_ts.isdisjoint(test_ts)


def test_chronological_split_float_test_size():
    df = _make_features_df(200)
    train, test = chronological_train_test_split(df, test_size=0.25)

    assert len(test) == 50
    assert len(train) == 150


def test_chronological_split_int_test_size():
    df = _make_features_df(200)
    train, test = chronological_train_test_split(df, test_size=30)

    assert len(test) == 30
    assert len(train) == 170


def test_chronological_split_invalid_fraction():
    df = _make_features_df(50)
    with pytest.raises(ValueError, match="test_size fraction"):
        chronological_train_test_split(df, test_size=1.5)
    with pytest.raises(ValueError, match="test_size fraction"):
        chronological_train_test_split(df, test_size=0.0)


def test_chronological_split_too_small():
    df = _make_features_df(3)
    with pytest.raises(ValueError, match="Need at least 1 train row"):
        chronological_train_test_split(df, test_size=3)


# ---------------------------------------------------------------------------
# walk_forward_split
# ---------------------------------------------------------------------------


def test_walk_forward_split_preserves_order():
    df = _make_features_df(100)
    folds = list(walk_forward_split(df, train_size=50, test_size=10, step_size=10))

    assert len(folds) == 5
    for train, test in folds:
        assert train["timestamp"].max() < test["timestamp"].min()


def test_walk_forward_split_no_future_leakage():
    df = _make_features_df(60)
    folds = list(walk_forward_split(df, train_size=30, test_size=10, step_size=10))

    # Each test window should be strictly after its train window
    for i, (train, test) in enumerate(folds):
        train_max = train["timestamp"].max()
        test_min = test["timestamp"].min()
        assert train_max < test_min, f"Fold {i}: train max {train_max} >= test min {test_min}"

        # No test row should appear in any train set
        test_indices = set(test.index)
        train_indices = set(train.index)
        assert test_indices.isdisjoint(train_indices), f"Fold {i}: overlap detected"


def test_walk_forward_split_invalid_sizes():
    df = _make_features_df(50)
    with pytest.raises(ValueError, match="train_size must be >= 1"):
        list(walk_forward_split(df, train_size=0, test_size=10))
    with pytest.raises(ValueError, match="test_size must be >= 1"):
        list(walk_forward_split(df, train_size=10, test_size=0))


def test_walk_forward_split_insufficient_data():
    df = _make_features_df(20)
    with pytest.raises(ValueError, match="train_size \\+ test_size"):
        list(walk_forward_split(df, train_size=15, test_size=10))


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------


def test_zero_return_baseline():
    model = ZeroReturnBaseline()
    model.fit(np.ones((10, 3)), np.array([0.01, -0.02, 0.0, 0.05, -0.01] * 2))
    pred = model.predict(np.ones((5, 3)))
    assert pred.shape == (5,)
    assert np.all(pred == 0.0)


def test_last_return_baseline():
    y_train = np.array([0.01, -0.02, 0.0, 0.05, -0.01])
    model = LastReturnBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((3, 3)))
    assert pred.shape == (3,)
    assert np.allclose(pred, -0.01)


def test_moving_average_return_baseline():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = MovingAverageReturnBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((2, 3)))
    assert np.allclose(pred, 3.0)


def test_momentum_direction_baseline_positive():
    y_train = np.array([-0.01, -0.02, 0.0, -0.01, 0.05])
    model = MomentumDirectionBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((3, 3)))
    assert pred.shape == (3,)
    assert np.all(pred == 1.0)


def test_momentum_direction_baseline_negative():
    y_train = np.array([0.01, 0.02, 0.0, 0.01, -0.05])
    model = MomentumDirectionBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((3, 3)))
    assert np.all(pred == 0.0)


def test_majority_class_baseline_uses_only_training():
    y_train = np.array([0, 1, 0, 1, 0])  # majority = 0
    model = MajorityClassBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((3, 3)))
    assert np.all(pred == 0.0)


def test_majority_class_baseline_tie():
    y_train = np.array([0, 1, 0, 1])
    model = MajorityClassBaseline()
    model.fit(np.ones((4, 3)), y_train)
    pred = model.predict(np.ones((2, 3)))
    # First unique value wins in tie
    assert pred[0] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def test_get_target_column_regression():
    assert get_target_column("regression", 1) == "target_return_1"
    assert get_target_column("regression", 4) == "target_return_4"
    assert get_target_column("regression", 24) == "target_return_24"


def test_get_target_column_classification():
    assert get_target_column("classification", 1) == "target_direction_1"
    assert get_target_column("classification", 4) == "target_direction_4"
    assert get_target_column("classification", 24) == "target_direction_24"


def test_get_target_column_invalid():
    with pytest.raises(ValueError):
        get_target_column("invalid", 1)


def test_get_default_feature_columns_excludes_targets():
    df = _make_features_df(50)
    features = get_default_feature_columns(df)
    targets = {"target_return_1", "target_return_4", "target_return_24",
               "target_direction_1", "target_direction_4", "target_direction_24"}
    assert targets.isdisjoint(set(features))


def test_get_default_feature_columns_excludes_metadata():
    df = _make_features_df(50)
    features = get_default_feature_columns(df)
    metadata = {"exchange", "symbol", "timeframe", "timestamp"}
    assert metadata.isdisjoint(set(features))


def test_get_default_feature_columns_only_numeric():
    df = _make_features_df(50)
    features = get_default_feature_columns(df)
    for col in features:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_prepare_modeling_data_drops_nan_targets():
    df = _make_features_df(50)
    # Last row of target_return_24 is NaN
    X, y, timestamps, feats = prepare_modeling_data(
        df, target_column="target_return_24"
    )
    assert len(X) < 50
    assert not np.any(np.isnan(y))


def test_prepare_modeling_data_shape():
    df = _make_features_df(100)
    X, y, timestamps, feats = prepare_modeling_data(
        df, target_column="target_return_1"
    )
    assert X.shape[0] == len(y)
    assert len(timestamps) == len(y)
    assert X.shape[1] == len(feats)


def test_prepare_modeling_data_missing_target():
    df = _make_features_df(50)
    # Remove target column
    df_no_target = df.drop(columns=["target_return_1"])
    with pytest.raises(ValueError, match="not in DataFrame"):
        prepare_modeling_data(df_no_target, target_column="target_return_1")


def test_prepare_modeling_data_no_features():
    df = _make_features_df(50)
    with pytest.raises(ValueError, match="No feature columns"):
        prepare_modeling_data(df, target_column="target_return_1", feature_columns=[])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["directional_accuracy"] == 1.0


def test_regression_metrics_with_errors():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0


def test_regression_metrics_handles_nan():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y_true, y_pred)
    assert len(metrics) > 0
    assert "mae" in metrics


def test_regression_metrics_all_nan():
    metrics = regression_metrics(
        np.array([np.nan, np.nan]), np.array([1.0, 2.0])
    )
    assert metrics == {}


def test_classification_metrics_known_values():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_with_errors():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5


def test_classification_metrics_with_probability():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = classification_metrics(y_true, y_pred, y_prob)
    assert "auc" in metrics


def test_classification_metrics_handles_nan():
    y_true = np.array([0, 1, np.nan, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert len(metrics) > 0


def test_classification_metrics_all_nan():
    metrics = classification_metrics(
        np.array([np.nan, np.nan]), np.array([1.0, 0.0])
    )
    assert metrics == {}


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------


def test_ridge_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = RidgeRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    assert pred.shape == (10,)


def test_logistic_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = (X[:, 0] > 0).astype(float)
    model = LogisticRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    prob = model.predict_proba(X[40:])
    assert pred.shape == (10,)
    assert prob.shape == (10,)
    assert np.all((prob >= 0) & (prob <= 1))


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_regression():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = ZeroReturnBaseline()
    result = evaluate_model(
        model, "Zero", X[:40], y[:40], X[40:], y[40:], "regression"
    )
    assert result["model_name"] == "Zero"
    assert "mae" in result["metrics"]
    assert len(result["y_true"]) == 10
    assert len(result["y_pred"]) == 10


def test_evaluate_model_classification():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = (X[:, 0] > 0).astype(float)
    model = LogisticRegressionModel()
    result = evaluate_model(
        model, "LogReg", X[:40], y[:40], X[40:], y[40:], "classification"
    )
    assert result["model_name"] == "LogReg"
    assert "accuracy" in result["metrics"]
    assert "y_prob" in result


# ---------------------------------------------------------------------------
# compare_baselines_and_models
# ---------------------------------------------------------------------------


def test_compare_baselines_regression():
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2
    )
    assert result["task_type"] == "regression"
    assert result["horizon"] == 1
    assert "train_dates" in result
    assert "test_dates" in result
    assert len(result["metrics_table"]) >= 3  # at least 3 baselines
    assert len(result["predictions"]) > 0
    assert result["train_dates"]["end"] < result["test_dates"]["start"]


def test_compare_baselines_classification():
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="classification", horizon=1, test_size=0.2
    )
    assert result["task_type"] == "classification"
    assert len(result["metrics_table"]) >= 2  # at least 2 baselines
    assert "predictions" in result


def test_compare_baselines_no_tabular():
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2, include_tabular=False
    )
    model_names = result["metrics_table"]["model_name"].tolist()
    assert "Ridge" not in model_names
    assert "LightGBM" not in model_names


def test_compare_baselines_insufficient_data():
    df = _make_features_df(5)
    with pytest.raises(ValueError, match="Insufficient data"):
        compare_baselines_and_models(
            df, task_type="regression", horizon=24, test_size=0.3
        )


def test_compare_baselines_missing_target():
    df = _make_ohlcv_df(50)
    with pytest.raises(ValueError, match="Run build_features"):
        compare_baselines_and_models(
            df, task_type="regression", horizon=1, test_size=0.2
        )


# ---------------------------------------------------------------------------
# LightGBM graceful handling
# ---------------------------------------------------------------------------


def test_lightgbm_import_check():
    from src.crypto_trend_lab.models import tabular as tmod
    # _HAS_LIGHTGBM should be a boolean
    assert isinstance(tmod._HAS_LIGHTGBM, bool)
    # If not installed, constructing LightGBM models should raise ImportError
    if not tmod._HAS_LIGHTGBM:
        with pytest.raises(ImportError, match="LightGBM is not installed"):
            tmod.LightGBMRegressor()
        with pytest.raises(ImportError, match="LightGBM is not installed"):
            tmod.LightGBMClassifier()


# ---------------------------------------------------------------------------
# Prediction storage
# ---------------------------------------------------------------------------


def test_build_predictions_path_convention():
    path = _build_predictions_path(
        "binance", "BTC/USDT", "1h", "Ridge", "target_return_1"
    )
    path_str = str(path).replace("\\", "/")

    assert "data/predictions" in path_str
    assert "exchange=binance" in path_str
    assert "symbol=BTC_USDT" in path_str
    assert "timeframe=1h" in path_str
    assert "model=ridge" in path_str
    assert "target=target_return_1" in path_str
    assert path_str.endswith("predictions.parquet")


def test_save_and_load_predictions_roundtrip(tmp_path, monkeypatch):
    import src.crypto_trend_lab.storage.parquet as pmod

    monkeypatch.setattr(pmod, "DATA_DIR", tmp_path)

    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="utc")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "y_true": [0.01, -0.02, 0.0, 0.05, -0.01],
            "y_pred": [0.0, 0.0, 0.0, 0.0, 0.0],
            "model_name": "Zero Return",
            "target_column": "target_return_1",
        }
    )

    path = save_predictions_parquet(
        df, "binance", "BTC/USDT", "1h", "Zero Return", "target_return_1"
    )
    assert path.exists()

    loaded = load_predictions_parquet(
        "binance", "BTC/USDT", "1h", "Zero Return", "target_return_1"
    )
    assert len(loaded) == 5
    pd.testing.assert_series_equal(
        loaded["y_true"], df["y_true"], check_names=False
    )


def test_load_predictions_missing_file_returns_empty():
    df = load_predictions_parquet(
        "nonexistent", "BTC/USDT", "1h", "NoModel", "target_return_1"
    )
    assert df.empty
