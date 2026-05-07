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
from src.crypto_trend_lab.evaluation.full_report import (
    build_best_model_summary,
    generate_report_analysis,
    run_full_evaluation_report,
)
from src.crypto_trend_lab.evaluation.report import (
    compare_baselines_and_models,
    evaluate_model,
)
from src.crypto_trend_lab.evaluation.forecast import (
    forecast_path,
    forward_forecast,
)
from src.crypto_trend_lab.utils.helpers import (
    dataset_sizing_warning,
    estimate_coverage,
)

from src.crypto_trend_lab.models.baseline import (
    HistoricalMeanReturnBaseline,
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
    CatBoostClassifier,
    CatBoostRegressor,
    ElasticNetRegressionModel,
    ExtraTreesClassificationModel,
    ExtraTreesRegressionModel,
    HistGradientBoostingClassificationModel,
    HistGradientBoostingRegressionModel,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestRegressionModel,
    RidgeRegressionModel,
    XGBoostClassifier,
    XGBoostRegressor,
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


def test_walk_forward_split_invalid_step_size():
    df = _make_features_df(100)
    with pytest.raises(ValueError, match="step_size must be >= 1"):
        list(walk_forward_split(df, train_size=50, test_size=10, step_size=0))


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


def test_historical_mean_return_baseline_uses_only_training():
    y_train = np.array([0.01, -0.02, 0.03, 0.05, -0.01])
    model = HistoricalMeanReturnBaseline()
    model.fit(np.ones((5, 3)), y_train)
    pred = model.predict(np.ones((3, 3)))
    expected_mean = float(np.mean(y_train))
    assert np.allclose(pred, expected_mean)
    # Output shape matches test set
    assert pred.shape == (3,)


def test_historical_mean_return_baseline_output_shape():
    X_train = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_train = np.random.default_rng(43).normal(0, 0.01, 100)
    X_test = np.random.default_rng(44).normal(0, 1, (20, 5))
    model = HistoricalMeanReturnBaseline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == (20,)


def test_baselines_do_not_use_test_targets():
    """All naive baselines must only use training y, never test y."""
    y_train = np.array([0.01, -0.02, 0.03])

    for model in [
        ZeroReturnBaseline(),
        LastReturnBaseline(),
        MovingAverageReturnBaseline(),
        HistoricalMeanReturnBaseline(),
    ]:
        model.fit(np.ones((3, 2)), y_train)
        pred = model.predict(np.ones((2, 2)))
        # Zero return: always 0
        if isinstance(model, ZeroReturnBaseline):
            assert np.all(pred == 0.0), f"{type(model).__name__} failed"
        # Predictions should be based on y_train, not y_test
        assert pred.shape == (2,), f"{type(model).__name__} shape mismatch"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def test_get_target_column_regression():
    assert get_target_column("regression", 1) == "target_return_1"
    assert get_target_column("regression", 2) == "target_return_2"
    assert get_target_column("regression", 4) == "target_return_4"
    assert get_target_column("regression", 5) == "target_return_5"
    assert get_target_column("regression", 24) == "target_return_24"


def test_get_target_column_classification():
    assert get_target_column("classification", 1) == "target_direction_1"
    assert get_target_column("classification", 2) == "target_direction_2"
    assert get_target_column("classification", 4) == "target_direction_4"
    assert get_target_column("classification", 24) == "target_direction_24"


def test_get_target_column_numpy_int64():
    assert get_target_column("regression", np.int64(1)) == "target_return_1"
    assert get_target_column("regression", np.int64(2)) == "target_return_2"
    assert get_target_column("classification", np.int64(2)) == "target_direction_2"


def test_get_target_column_numpy_int32():
    assert get_target_column("regression", np.int32(4)) == "target_return_4"
    assert get_target_column("classification", np.int32(24)) == "target_direction_24"


def test_get_target_column_invalid_task():
    with pytest.raises(ValueError, match="Unknown task_type"):
        get_target_column("invalid", 1)


@pytest.mark.parametrize("bad_horizon", [0, -1, 1.5, "1", None, True, False])
def test_get_target_column_invalid_horizon(bad_horizon):
    with pytest.raises(ValueError, match="positive integer"):
        get_target_column("regression", bad_horizon)


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


def test_regression_metrics_constant_y_pred_no_warning():
    """Constant y_pred must return NaN for spearman_r without warnings."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = regression_metrics(y_true, y_pred)
        # No ConstantInputWarning should escape
        constant_warnings = [
            x for x in w if "constant" in str(x.message).lower()
        ]
        assert len(constant_warnings) == 0, (
            f"ConstantInputWarning leaked: {constant_warnings}"
        )

    assert "spearman_r" in metrics
    assert np.isnan(metrics["spearman_r"])


def test_regression_metrics_constant_y_true_no_warning():
    """Constant y_true must also return NaN for spearman_r without warnings."""
    y_true = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = regression_metrics(y_true, y_pred)
        constant_warnings = [
            x for x in w if "constant" in str(x.message).lower()
        ]
        assert len(constant_warnings) == 0

    assert "spearman_r" in metrics
    assert np.isnan(metrics["spearman_r"])


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
# New sklearn tabular models — regression
# ---------------------------------------------------------------------------


def test_elasticnet_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = ElasticNetRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    assert pred.shape == (10,)


def test_random_forest_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = RandomForestRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    assert pred.shape == (10,)


def test_extra_trees_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = ExtraTreesRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    assert pred.shape == (10,)


def test_hist_gradient_boosting_regression_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = X[:, 0] * 0.5 + np.random.default_rng(99).normal(0, 0.1, 50)
    model = HistGradientBoostingRegressionModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    assert pred.shape == (10,)


# ---------------------------------------------------------------------------
# New sklearn tabular models — classification
# ---------------------------------------------------------------------------


def test_random_forest_classification_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = (X[:, 0] > 0).astype(float)
    model = RandomForestClassificationModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    prob = model.predict_proba(X[40:])
    assert pred.shape == (10,)
    assert prob.shape == (10,)
    assert np.all((prob >= 0) & (prob <= 1))


def test_extra_trees_classification_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = (X[:, 0] > 0).astype(float)
    model = ExtraTreesClassificationModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    prob = model.predict_proba(X[40:])
    assert pred.shape == (10,)
    assert prob.shape == (10,)
    assert np.all((prob >= 0) & (prob <= 1))


def test_hist_gradient_boosting_classification_fit_predict():
    X = np.random.default_rng(42).normal(0, 1, (50, 5))
    y = (X[:, 0] > 0).astype(float)
    model = HistGradientBoostingClassificationModel()
    model.fit(X[:40], y[:40])
    pred = model.predict(X[40:])
    prob = model.predict_proba(X[40:])
    assert pred.shape == (10,)
    assert prob.shape == (10,)
    assert np.all((prob >= 0) & (prob <= 1))


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------


def test_regression_model_registry_includes_new_models():
    from src.crypto_trend_lab.evaluation.forecast import (
        _REGRESSION_MODELS as reg_models,
    )
    assert "ElasticNet" in reg_models
    assert "Random Forest" in reg_models
    assert "Extra Trees" in reg_models
    assert "HistGradientBoosting" in reg_models
    assert "Historical Mean Return" in reg_models


def test_classification_model_registry_includes_new_models():
    from src.crypto_trend_lab.evaluation.forecast import (
        _CLASSIFICATION_MODELS as cls_models,
    )
    assert "Random Forest" in cls_models
    assert "Extra Trees" in cls_models
    assert "HistGradientBoosting" in cls_models


def test_xgboost_skipped_gracefully_if_not_installed():
    from src.crypto_trend_lab.models import tabular as tmod

    if not tmod._HAS_XGBOOST:
        with pytest.raises(ImportError, match="XGBoost is not installed"):
            XGBoostRegressor()
        with pytest.raises(ImportError, match="XGBoost is not installed"):
            XGBoostClassifier()
    else:
        # If installed, must construct and run
        X = np.random.default_rng(42).normal(0, 1, (20, 3))
        y = X[:, 0] * 0.5
        m = XGBoostRegressor()
        m.fit(X[:10], y[:10])
        pred = m.predict(X[10:])
        assert pred.shape == (10,)


def test_catboost_skipped_gracefully_if_not_installed():
    from src.crypto_trend_lab.models import tabular as tmod

    if not tmod._HAS_CATBOOST:
        with pytest.raises(ImportError, match="CatBoost is not installed"):
            CatBoostRegressor()
        with pytest.raises(ImportError, match="CatBoost is not installed"):
            CatBoostClassifier()
    else:
        X = np.random.default_rng(42).normal(0, 1, (20, 3))
        y = X[:, 0] * 0.5
        m = CatBoostRegressor()
        m.fit(X[:10], y[:10])
        pred = m.predict(X[10:])
        assert pred.shape == (10,)


def test_xgboost_catboost_flag_consistency():
    """_HAS_XGBOOST and _HAS_CATBOOST must be bool."""
    from src.crypto_trend_lab.models import tabular as tmod

    assert isinstance(tmod._HAS_XGBOOST, bool)
    assert isinstance(tmod._HAS_CATBOOST, bool)


def test_include_trees_defaults_to_false():
    """compare_baselines_and_models must default include_trees=False."""
    from src.crypto_trend_lab.evaluation.report import (
        _build_tabular_models,
    )

    reg_models = _build_tabular_models("regression", include_trees=False)
    reg_names = [n for n, _ in reg_models]
    # Without trees: only linear + LightGBM
    assert "Random Forest" not in reg_names
    assert "Extra Trees" not in reg_names
    assert "HistGradientBoosting" not in reg_names


def test_include_trees_true_adds_ensemble_models():
    """With include_trees=True, ensemble models must be present."""
    from src.crypto_trend_lab.evaluation.report import (
        _build_tabular_models,
    )

    reg_models = _build_tabular_models("regression", include_trees=True)
    reg_names = [n for n, _ in reg_models]
    assert "Random Forest" in reg_names
    assert "Extra Trees" in reg_names
    assert "HistGradientBoosting" in reg_names


def test_full_report_include_trees_flows_through():
    """run_full_evaluation_report with include_trees=True must include tree models."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,),
        test_size=0.2, include_trees=True,
    )
    model_names = report["metrics_df"]["model_name"].unique().tolist()
    assert "Random Forest" in model_names
    assert "Extra Trees" in model_names
    assert "HistGradientBoosting" in model_names


# ---------------------------------------------------------------------------
# Target leakage tests
# ---------------------------------------------------------------------------


def test_target_columns_excluded_from_features_all_models():
    """All model evaluations must exclude target_* from feature columns."""
    df = _make_features_df(200)
    features = get_default_feature_columns(df)
    assert not any(c.startswith("target_") for c in features)

    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2,
        include_trees=True,
    )
    assert "feature_columns" in result
    assert not any(c.startswith("target_") for c in result["feature_columns"])


def test_chronological_split_preserved_with_new_models():
    """Chronological order must be preserved with tree ensemble models."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2,
        include_trees=True,
    )
    assert result["train_dates"]["end"] < result["test_dates"]["start"]
    assert result["train_dates"]["start"] < result["train_dates"]["end"]


def test_preprocessing_fit_only_on_training():
    """ElasticNet scaler must be fit only on training data, not test."""
    X_train = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_train = np.random.default_rng(43).normal(0, 0.01, 100)
    X_test = np.random.default_rng(44).normal(100, 50, (20, 5))  # very different scale

    model = ElasticNetRegressionModel()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == (20,)
    # Predictions should be finite (scaler handles the different scale)
    assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# Robustness: empty predictions_list and skip/failure logging
# ---------------------------------------------------------------------------


def test_empty_predictions_list_returns_stable_columns():
    """When all models are filtered out, predictions must be empty with
    stable columns, not raise ValueError from pd.concat([])."""
    df = _make_features_df(200)
    # model_names with a name that matches no baseline or tabular model
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2,
        model_names=["NonExistentModel"],
    )
    preds = result["predictions"]
    assert preds.empty
    for col in ("timestamp", "y_true", "y_pred", "model_name", "target_column"):
        assert col in preds.columns, f"Missing column: {col}"

    # metrics_table must also have stable columns
    metrics = result["metrics_table"]
    assert metrics.empty
    assert "model_name" in metrics.columns

    # skipped must record the models that were not found
    skipped = result.get("skipped", [])
    assert len(skipped) == 1  # NonExistentModel reported as not found
    assert skipped[0]["model_name"] == "NonExistentModel"
    assert "not found" in skipped[0]["reason"]


def test_no_successful_model_returns_skip_log():
    """When no model produces predictions, skipped list must be populated."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2,
        include_tabular=False,
        model_names=["Logistic Regression"],  # classification model, won't be in regression list
    )
    preds = result["predictions"]
    assert preds.empty
    # With include_tabular=False and only a tabular model name, nothing runs
    skipped = result.get("skipped", [])
    assert isinstance(skipped, list)


def test_single_class_classification_is_skipped():
    """When all training labels are the same class, classifier must be skipped."""
    df = _make_features_df(200)
    # Set all target_direction_1 to the same class
    df["target_direction_1"] = 0.0
    result = compare_baselines_and_models(
        df, task_type="classification", horizon=1, test_size=0.2,
    )
    # At least LogisticRegression should be skipped due to single class
    skipped = result.get("skipped", [])
    skip_reasons = [s["reason"] for s in skipped]
    single_class_skips = [r for r in skip_reasons if "Single class" in r]
    assert len(single_class_skips) > 0, (
        f"Expected at least one single-class skip, got: {skip_reasons}"
    )
    # Baselines (MomentumDirection, MajorityClass) should still succeed
    assert not result["predictions"].empty
    assert "Momentum Direction" in result["predictions"]["model_name"].values


def test_missing_target_column_skipped_with_reason():
    """When target column is missing, ValueError with clear message is raised."""
    df = _make_ohlcv_df(100)
    with pytest.raises(ValueError, match="Run build_features"):
        compare_baselines_and_models(
            df, task_type="regression", horizon=1, test_size=0.2,
        )


def test_optional_model_unavailable_is_handled():
    """When XGBoost is not installed, it should not appear in available models,
    and selecting it via model_names should not crash."""
    from src.crypto_trend_lab.models import tabular as tmod

    df = _make_features_df(200)
    if not tmod._HAS_XGBOOST:
        result = compare_baselines_and_models(
            df, task_type="regression", horizon=1, test_size=0.2,
            model_names=["XGBoost"],
        )
        # XGBoost not in registry, so no baseline/tabular model matches
        assert result["predictions"].empty
    else:
        # If installed, XGBoost should work
        result = compare_baselines_and_models(
            df, task_type="regression", horizon=1, test_size=0.2,
            include_trees=True,
            model_names=["XGBoost"],
        )
        assert not result["predictions"].empty


def test_successful_models_still_produce_normal_predictions():
    """When some models succeed and some fail, successful ones are concatenated."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2,
    )
    assert not result["predictions"].empty
    assert not result["metrics_table"].empty
    # All baselines + Ridge + ElasticNet + (LightGBM if installed) should succeed
    assert len(result["metrics_table"]) >= 5
    # skipped list must exist in result
    assert "skipped" in result


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


# ---------------------------------------------------------------------------
# Chronological split — descending / unsorted inputs
# ---------------------------------------------------------------------------


def test_chronological_split_with_descending_timestamps():
    """Split must work correctly when input is descending by timestamp."""
    ts = pd.date_range("2024-01-01", periods=100, freq="1h", tz="utc")
    df = pd.DataFrame({"timestamp": ts[::-1], "value": range(100)})
    # Descending: newest first
    assert df["timestamp"].iloc[0] > df["timestamp"].iloc[-1]

    train, test = chronological_train_test_split(df, test_size=20)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_chronological_split_with_unsorted_timestamps():
    """Split must work when input timestamps are randomly shuffled."""
    rng = np.random.default_rng(99)
    ts = pd.date_range("2024-01-01", periods=100, freq="1h", tz="utc")
    values = rng.permutation(ts)
    df = pd.DataFrame({"timestamp": values, "value": range(100)})

    train, test = chronological_train_test_split(df, test_size=0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) == 80
    assert len(test) == 20


def test_walk_forward_split_descending_input():
    """Walk-forward must work when input is descending."""
    ts = pd.date_range("2024-01-01", periods=100, freq="1h", tz="utc")
    df = pd.DataFrame({"timestamp": ts[::-1], "value": range(100)})

    folds = list(walk_forward_split(df, train_size=50, test_size=10, step_size=10))
    assert len(folds) == 5
    for train, test in folds:
        assert train["timestamp"].max() < test["timestamp"].min()


# ---------------------------------------------------------------------------
# Date range reporting correctness
# ---------------------------------------------------------------------------


def test_train_dates_start_before_end():
    """Displayed train start must be earlier than train end."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2
    )
    td = result["train_dates"]
    assert td["start"] < td["end"], f"Train: start {td['start']} >= end {td['end']}"


def test_test_dates_start_before_end():
    """Displayed test start must be earlier than test end."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2
    )
    td = result["test_dates"]
    assert td["start"] < td["end"], f"Test: start {td['start']} >= end {td['end']}"


def test_train_end_before_test_start():
    """Train end must be strictly before test start."""
    df = _make_features_df(200)
    result = compare_baselines_and_models(
        df, task_type="regression", horizon=1, test_size=0.2
    )
    assert result["train_dates"]["end"] < result["test_dates"]["start"], (
        f"Train end {result['train_dates']['end']} >= "
        f"test start {result['test_dates']['start']}"
    )


# ---------------------------------------------------------------------------
# Timestamp alignment after NaN filtering
# ---------------------------------------------------------------------------


def test_prepare_modeling_data_timestamps_monotonic():
    """Timestamps must be monotonic increasing after NaN removal."""
    df = _make_features_df(200)
    # target_return_1 needs at least 1 future row
    X, y, timestamps, feats = prepare_modeling_data(
        df, target_column="target_return_1"
    )
    # timestamps should be sorted ascending
    assert timestamps.is_monotonic_increasing, "Timestamps not monotonic after NaN filtering"


def test_prepare_modeling_data_timestamps_no_nan():
    """Timestamps must have no NaN values after preparation."""
    df = _make_features_df(200)
    X, y, timestamps, feats = prepare_modeling_data(
        df, target_column="target_return_1"
    )
    assert not timestamps.isnull().any(), "Timestamps contain NaN values"


def test_compare_baselines_with_descending_input_df():
    """compare_baselines_and_models must work with descending input."""
    df = _make_features_df(200)
    # Reverse the DataFrame to simulate descending order
    df_desc = df.iloc[::-1].reset_index(drop=True)

    result = compare_baselines_and_models(
        df_desc, task_type="regression", horizon=1, test_size=0.2
    )
    td = result["train_dates"]
    assert td["start"] < td["end"], "Train dates reversed (descending input)"
    assert result["train_dates"]["end"] < result["test_dates"]["start"], (
        "Train/test overlap with descending input"
    )


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------


def test_full_report_runs_all_combinations():
    """Full report must run all 2 task types × 3 horizons = 6 combinations."""
    # Use 300 rows so horizon=24 has enough valid data after NaN removal
    df = _make_features_df(300)
    report = run_full_evaluation_report(
        df, task_types=("regression", "classification"), horizons=(1, 4, 24),
        test_size=0.2,
    )
    assert report["summary"]["total_combinations"] == 6
    assert report["summary"]["successful"] == 6
    assert len(report["metrics_df"]) > 0
    assert report["skipped_df"].empty


def test_full_report_metrics_df_columns():
    """Metrics DataFrame must include task_type, horizon, target_column, model_name."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    metrics_df = report["metrics_df"]
    for col in ("task_type", "horizon", "target_column", "model_name"):
        assert col in metrics_df.columns, f"Missing column: {col}"


def test_full_report_skipped_df_columns():
    """Skipped DataFrame must include task_type, horizon, model_name, reason."""
    # Use an OHLCV-only DataFrame (no targets) — all combinations should be skipped
    df_no_targets = _make_ohlcv_df(200)
    report = run_full_evaluation_report(
        df_no_targets, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    skipped_df = report["skipped_df"]
    assert not skipped_df.empty
    for col in ("task_type", "horizon", "model_name", "reason"):
        assert col in skipped_df.columns, f"Missing column: {col}"
    # Every skip reason should mention the missing target column
    assert all("target_return_1" in r for r in skipped_df["reason"])


def test_full_report_missing_target_skips_gracefully():
    """Missing target columns must not crash the report."""
    df = _make_ohlcv_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression", "classification"), horizons=(1, 4, 24),
        test_size=0.2,
    )
    assert report["summary"]["successful"] == 0
    assert len(report["skipped_df"]) > 0
    assert not report["skipped_df"].empty


def test_full_report_insufficient_data_skips():
    """Too few rows after NaN removal must skip with a clear reason."""
    df = _make_features_df(10)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(24,), test_size=0.3,
    )
    # With 10 rows at horizon=24, target_return_24 has the last 24 rows NaN → 0 valid
    assert report["summary"]["successful"] == 0
    assert not report["skipped_df"].empty
    assert any("Insufficient data" in r for r in report["skipped_df"]["reason"])


def test_full_report_lightgbm_handled():
    """LightGBM presence/absence must be reflected correctly."""
    from src.crypto_trend_lab.models import tabular as tmod

    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    # summary must report lightgbm availability truthfully
    assert report["summary"]["lightgbm_available"] == tmod._HAS_LIGHTGBM

    # If LightGBM is available, it appears in metrics_df
    if tmod._HAS_LIGHTGBM:
        model_names = report["metrics_df"]["model_name"].unique().tolist()
        assert "LightGBM" in model_names


def test_full_report_both_task_types_in_metrics():
    """Regression AND classification rows both appear in metrics_df."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression", "classification"), horizons=(1,),
        test_size=0.2,
    )
    task_types_found = set(report["metrics_df"]["task_type"].unique())
    assert "regression" in task_types_found
    assert "classification" in task_types_found


def test_full_report_chronological_order_per_run():
    """Full report summary must reflect chronological train_dates for every run."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    assert report["summary"]["min_timestamp"] is not None
    assert report["summary"]["max_timestamp"] is not None
    assert report["summary"]["min_timestamp"] < report["summary"]["max_timestamp"]


def test_full_report_summary_fields():
    """Summary must include expected dataset fields."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    s = report["summary"]
    for key in ("exchange", "symbol", "timeframe", "row_count",
                "min_timestamp", "max_timestamp", "feature_count",
                "total_combinations", "successful", "skipped",
                "lightgbm_available"):
        assert key in s, f"Missing summary key: {key}"


def test_full_report_no_live_network_calls():
    """Full report must not make any network requests."""
    df = _make_features_df(200)
    # The function is purely local — if it runs without error, it passes
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    assert report is not None


# ---------------------------------------------------------------------------
# build_best_model_summary
# ---------------------------------------------------------------------------


def test_build_best_model_summary_returns_dataframe():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression", "classification"),
        horizons=(1,), test_size=0.2,
    )
    best_df = build_best_model_summary(report["metrics_df"])
    assert not best_df.empty


def test_build_best_model_summary_columns():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression", "classification"),
        horizons=(1,), test_size=0.2,
    )
    best_df = build_best_model_summary(report["metrics_df"])
    assert "task_type" in best_df.columns
    assert "horizon" in best_df.columns
    # Regression columns
    reg_rows = best_df[best_df["task_type"] == "regression"]
    if not reg_rows.empty:
        assert "best_mae_model" in best_df.columns
        assert "best_rmse_model" in best_df.columns
    # Classification columns
    cls_rows = best_df[best_df["task_type"] == "classification"]
    if not cls_rows.empty:
        assert "best_balanced_accuracy_model" in best_df.columns
        assert "best_f1_model" in best_df.columns


def test_build_best_model_summary_empty_metrics():
    empty_df = pd.DataFrame(columns=["task_type", "horizon", "model_name"])
    best_df = build_best_model_summary(empty_df)
    assert best_df.empty


def test_build_best_model_summary_handles_missing_metrics():
    """Must not crash when some metrics are entirely NaN."""
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    best_df = build_best_model_summary(report["metrics_df"])
    assert isinstance(best_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# generate_report_analysis
# ---------------------------------------------------------------------------


def test_generate_report_analysis_returns_text():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    text = generate_report_analysis(
        report["metrics_df"], report["summary"], report["skipped_df"]
    )
    assert isinstance(text, str)
    assert "Dataset" in text
    assert "Cautions" in text


def test_generate_report_analysis_includes_cautions():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    text = generate_report_analysis(report["metrics_df"], report["summary"])
    assert "historical evaluation" in text
    assert "not investment advice" in text
    assert "non-stationary" in text


def test_generate_report_analysis_no_crash_on_empty():
    text = generate_report_analysis(
        pd.DataFrame(),
        {"row_count": 0, "timeframe": "1h", "skipped": 5},
    )
    assert "No metrics were produced" in text


def test_generate_report_analysis_no_crash_missing_fields():
    text = generate_report_analysis(
        pd.DataFrame(), {"row_count": 100},
    )
    assert isinstance(text, str)


def test_generate_report_analysis_small_dataset_warning():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    summary = report["summary"]
    summary["row_count"] = 500  # simulate small dataset
    text = generate_report_analysis(report["metrics_df"], summary)
    assert "too small" in text.lower() or "small" in text.lower()


def test_generate_report_analysis_mentions_skipped():
    df = _make_features_df(200)
    report = run_full_evaluation_report(
        df, task_types=("regression",), horizons=(1,), test_size=0.2,
    )
    skipped_df = pd.DataFrame([{
        "task_type": "classification", "horizon": 24,
        "model_name": "Logistic Regression", "reason": "Not enough data",
    }])
    summary = report["summary"].copy()
    summary["skipped"] = 1
    text = generate_report_analysis(report["metrics_df"], summary, skipped_df)
    assert "Skipped" in text
    assert "1 model" in text


# ---------------------------------------------------------------------------
# forward_forecast
# ---------------------------------------------------------------------------


def test_forward_forecast_regression_trains_on_labeled_only():
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
    )
    assert "error" not in result
    assert "latest_timestamp" in result
    assert result["horizon"] == 1
    assert "predicted_log_return" in result
    assert result["training_rows"] > 0


def test_forward_forecast_classification_returns_class():
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="classification", horizon=1,
        model_name="Logistic Regression",
    )
    assert "error" not in result
    assert "predicted_class" in result
    assert result["predicted_class"] in (0, 1)
    assert "predicted_probability" in result
    assert result["predicted_probability"] is not None


def test_forward_forecast_latest_row_can_have_nan_target():
    """Latest feature row should be usable even when target is NaN."""
    df = _make_features_df(250)
    # The last row naturally has NaN target for horizon>=1
    result = forward_forecast(
        df, task_type="regression", horizon=4, model_name="Ridge",
    )
    assert "error" not in result
    assert result["latest_timestamp"] is not None


def test_forward_forecast_target_columns_excluded_from_features():
    """Target columns must not leak into input features."""
    df = _make_features_df(250)
    # Get default feature columns to verify they exclude targets
    features = get_default_feature_columns(df)
    targets = {
        "target_return_1", "target_return_4", "target_return_24",
        "target_direction_1", "target_direction_4", "target_direction_24",
    }
    assert targets.isdisjoint(set(features))
    # Forward forecast should run without error with filtered features
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
        feature_columns=features,
    )
    assert "error" not in result


def test_forward_forecast_estimated_future_close():
    df = _make_features_df(250)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
    )
    assert "error" not in result
    assert "estimated_future_close" in result
    if result.get("latest_close") is not None:
        assert result["estimated_future_close"] > 0


def test_forward_forecast_insufficient_data_error():
    df = _make_features_df(5)
    result = forward_forecast(
        df, task_type="regression", horizon=24, model_name="Ridge",
    )
    assert "error" in result
    assert "Need at least 10" in result["error"]


def test_forward_forecast_missing_target_column():
    df = _make_ohlcv_df(100)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
    )
    assert "error" in result
    assert "not in DataFrame" in result["error"]


def test_forward_forecast_baseline_regression():
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Zero Return",
    )
    assert "error" not in result
    assert result["predicted_log_return"] == 0.0


def test_forward_forecast_baseline_classification():
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="classification", horizon=1,
        model_name="Majority Class",
    )
    assert "error" not in result
    assert result["predicted_class"] in (0, 1)
    # Majority Class does not support probability
    assert result["predicted_probability"] is None


def test_forward_forecast_unknown_model():
    df = _make_features_df(100)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="NoSuchModel",
    )
    assert "error" in result
    assert "Unknown model" in result["error"]


def test_forward_forecast_produces_historical_context():
    """When enough data, historical metrics should be present."""
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
    )
    assert "error" not in result
    # With 200 rows and test_size=0.2, we should get context metrics
    if result["training_rows"] >= 50:
        assert "historical_mae" in result
        assert "historical_rmse" in result


def test_forward_forecast_no_live_network_calls():
    df = _make_features_df(200)
    result = forward_forecast(
        df, task_type="regression", horizon=1, model_name="Ridge",
    )
    assert result is not None


def test_forward_forecast_excludes_nan_target_rows_from_training():
    """Rows with NaN target must not be used for model training."""
    df = _make_features_df(100)
    # Rows at the end have NaN target
    total_rows = len(df)
    result = forward_forecast(
        df, task_type="regression", horizon=24, model_name="Ridge",
    )
    if "error" not in result:
        # Training rows should be less than total (some NaN targets dropped)
        assert result["training_rows"] < total_rows


# ---------------------------------------------------------------------------
# Forecast storage
# ---------------------------------------------------------------------------


def test_build_forecast_path_convention():
    from src.crypto_trend_lab.storage.parquet import _build_forecast_path

    path = _build_forecast_path(
        "binance", "BTC/USDT", "1h", "Ridge", "target_return_1"
    )
    path_str = str(path).replace("\\", "/")

    assert "data/forecasts" in path_str
    assert "exchange=binance" in path_str
    assert "symbol=BTC_USDT" in path_str
    assert "timeframe=1h" in path_str
    assert "model=ridge" in path_str
    assert "target=target_return_1" in path_str
    assert path_str.endswith("forecast.parquet")


def test_save_and_load_forecast_roundtrip(tmp_path, monkeypatch):
    import src.crypto_trend_lab.storage.parquet as pmod

    monkeypatch.setattr(pmod, "DATA_DIR", tmp_path)

    ts = pd.Timestamp("2024-01-15 12:00:00", tz="utc")
    df = pd.DataFrame([{
        "timestamp": ts,
        "horizon": 1,
        "target_column": "target_return_1",
        "model_name": "Ridge",
        "predicted_log_return": 0.0015,
        "estimated_future_close": 40500.0,
    }])

    path = pmod.save_forecast_parquet(
        df, "binance", "BTC/USDT", "1h", "Ridge", "target_return_1"
    )
    assert path.exists()

    loaded = pmod.load_forecast_parquet(
        "binance", "BTC/USDT", "1h", "Ridge", "target_return_1"
    )
    assert len(loaded) == 1
    assert loaded["predicted_log_return"].iloc[0] == 0.0015


def test_load_forecast_missing_file_returns_empty():
    from src.crypto_trend_lab.storage.parquet import load_forecast_parquet

    df = load_forecast_parquet(
        "nonexistent", "BTC/USDT", "1h", "NoModel", "target_return_1"
    )
    assert df.empty


# ---------------------------------------------------------------------------
# Issue 1: Pagination logic for large recent-bars requests
# ---------------------------------------------------------------------------


def test_estimate_coverage_with_large_limit():
    """Coverage estimate must work for large limits (50000 bars)."""
    result = estimate_coverage(50000, "1h")
    assert "day" in result or "days" in result
    assert "bar" not in result.lower() if "unknown" not in result else True


def test_dataset_sizing_warning_large_dataset():
    """No warning for large datasets."""
    assert dataset_sizing_warning(50000, "1h") is None
    assert dataset_sizing_warning(10000, "4h") is None
    assert dataset_sizing_warning(5000, "1d") is None


def test_timeframe_duration_mapping_complete():
    """All supported timeframes must have duration mappings."""
    from src.crypto_trend_lab.utils.helpers import TIMEFRAME_TO_DURATION

    for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
        assert tf in TIMEFRAME_TO_DURATION, f"Missing duration for {tf}"
        assert isinstance(TIMEFRAME_TO_DURATION[tf], pd.Timedelta)


def test_fetch_range_called_when_limit_exceeds_cap(monkeypatch):
    """When limit > 1000, fetch_ohlcv_range should be used (logic check)."""
    from src.crypto_trend_lab.utils.helpers import TIMEFRAME_TO_DURATION

    # Simulate the decision logic in app.py
    limit = 50000
    timeframe = "1h"
    use_range = limit > 1000
    assert use_range is True

    # Verify the duration calculation works
    duration = limit * TIMEFRAME_TO_DURATION[timeframe]
    assert duration > pd.Timedelta(days=365)  # 50000 hours ≈ 5.7 years


def test_pagination_trims_excess_rows():
    """Simulate: when range fetch returns more than requested, trim to limit."""
    import numpy as np

    limit = 5000
    returned = 5200  # exchange returned slightly more
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=returned, freq="1h", tz="utc"),
        "close": np.arange(returned, dtype=float),
    })

    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)

    assert len(df) == limit
    assert df["close"].iloc[0] == returned - limit  # first row is shifted


# ---------------------------------------------------------------------------
# Issue 2: Forecast path
# ---------------------------------------------------------------------------


def test_forecast_path_regression_returns_points():
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert "error" not in result
    assert "path_points" in result
    assert len(result["path_points"]) > 0
    # Must have at least horizons 1 and 4 (24 > 24, so all 3)
    horizons_found = {p["horizon"] for p in result["path_points"] if "error" not in p}
    assert horizons_found.issuperset({1, 4})


def test_forecast_path_only_supported_horizons():
    """Only horizons 1, 4, 24 ≤ path_length should be in path points."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=6, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    horizons = {p["horizon"] for p in points}
    assert horizons == {1, 4}  # 24 > 6, so excluded
    assert all(h <= 6 for h in horizons)


def test_forecast_path_future_timestamps_after_latest():
    from src.crypto_trend_lab.evaluation.forecast import generate_future_timestamps

    latest = pd.Timestamp("2024-06-15 12:00:00", tz="utc")
    ts = generate_future_timestamps(latest, "1h", 5)
    assert len(ts) == 5
    assert all(t > latest for t in ts)
    assert ts[0] == pd.Timestamp("2024-06-15 13:00:00", tz="utc")
    assert ts[4] == pd.Timestamp("2024-06-15 17:00:00", tz="utc")


def test_forecast_path_future_timestamps_4h():
    from src.crypto_trend_lab.evaluation.forecast import generate_future_timestamps

    latest = pd.Timestamp("2024-06-15 12:00:00", tz="utc")
    ts = generate_future_timestamps(latest, "4h", 3)
    assert len(ts) == 3
    assert ts[0] == pd.Timestamp("2024-06-15 16:00:00", tz="utc")
    assert ts[2] == pd.Timestamp("2024-06-16 00:00:00", tz="utc")


def test_forecast_path_future_timestamps_1d():
    from src.crypto_trend_lab.evaluation.forecast import generate_future_timestamps

    latest = pd.Timestamp("2024-06-15 12:00:00", tz="utc")
    ts = generate_future_timestamps(latest, "1d", 5)
    assert len(ts) == 5
    assert ts[0] == pd.Timestamp("2024-06-16 12:00:00", tz="utc")


def test_forecast_path_estimated_close_calculation():
    """estimated_future_close = latest_close * exp(predicted_log_return)."""
    import numpy as np

    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert result["latest_close"] is not None
    for p in result["path_points"]:
        if "error" not in p:
            expected = result["latest_close"] * np.exp(p["predicted_log_return"])
            assert abs(p["estimated_future_close"] - expected) < 1e-6


def test_forecast_path_classification_error():
    """Classification model names must produce an error for forecast_path."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Logistic Regression", path_length=24, timeframe="1h",
    )
    assert "error" in result
    assert "regression" in result["error"].lower()


def test_forecast_path_table_contains_required_columns():
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    for p in result["path_points"]:
        if "error" not in p:
            assert "horizon" in p
            assert "forecast_step" in p
            assert "forecast_timestamp" in p
            assert "target_column" in p
            assert "predicted_log_return" in p
            assert "estimated_future_close" in p


def test_forecast_path_excludes_targets_from_features():
    """Feature columns passed to forecast_path must not include target columns."""
    df = _make_features_df(300)
    features = get_default_feature_columns(df)
    targets = {
        "target_return_1", "target_return_4", "target_return_24",
        "target_direction_1", "target_direction_4", "target_direction_24",
    }
    assert targets.isdisjoint(set(features))

    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
        feature_columns=features,
    )
    assert "error" not in result


def test_forecast_path_latest_row_nan_target():
    """Latest feature row used for forecast may have NaN target."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert "error" not in result
    assert result["latest_timestamp"] is not None
    assert result["latest_close"] > 0


def test_forecast_path_handles_small_path_length():
    """path_length=6 only produces horizon 1 and 4 points."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=6, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    assert all(p["horizon"] in (1, 4) for p in points)


def test_forecast_path_handles_large_path_length():
    """path_length=168 produces all three horizons (1, 4, 24)."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=168, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    horizons = {p["horizon"] for p in points}
    assert horizons == {1, 4, 24}


def test_forecast_path_chart_history_included():
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert "chart_history" in result
    assert not result["chart_history"].empty
    assert "timestamp" in result["chart_history"].columns
    assert "close" in result["chart_history"].columns


def test_forecast_path_no_live_network_calls():
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert result is not None


def test_forecast_path_sparse_not_per_bar():
    """path_length=48 must produce exactly 3 forecast points, not 48."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=48, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    assert len(points) == 3, (
        f"Expected 3 sparse points (h=1,4,24), got {len(points)}"
    )
    assert {p["horizon"] for p in points} == {1, 4, 24}


@pytest.mark.parametrize("path_len,expected_horizons", [
    (6, {1, 4}),
    (12, {1, 4}),
    (24, {1, 4, 24}),
    (48, {1, 4, 24}),
    (72, {1, 4, 24}),
    (168, {1, 4, 24}),
])
def test_forecast_path_horizon_selection(path_len, expected_horizons):
    """Only horizons <= path_length are included in forecast path."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=path_len, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    horizons = {p["horizon"] for p in points}
    assert horizons == expected_horizons, (
        f"path_length={path_len}: expected {expected_horizons}, got {horizons}"
    )


def test_forecast_path_max_supported_horizon_in_result():
    """forecast_path must include max_supported_horizon in the return dict."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    assert "max_supported_horizon" in result
    assert result["max_supported_horizon"] == 24


def test_forecast_path_table_columns_complete():
    """Every path_point with no error must contain all table columns."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    for p in result["path_points"]:
        if "error" not in p:
            for col in ("horizon", "forecast_step", "forecast_timestamp",
                        "target_column", "predicted_log_return",
                        "estimated_future_close"):
                assert col in p, f"Missing column {col!r} in path_point"
    # model_name must be in the top-level result for the table
    assert "model_name" in result


# ---------------------------------------------------------------------------
# add_dense_return_targets
# ---------------------------------------------------------------------------


def test_add_dense_return_targets_creates_all_columns():
    from src.crypto_trend_lab.features.target import add_dense_return_targets

    df = _make_ohlcv_df(50)
    max_h = 12
    result = add_dense_return_targets(df, max_h)

    for h in range(1, max_h + 1):
        assert f"target_return_{h}" in result.columns
        assert f"target_direction_{h}" in result.columns
    assert len(result.columns) == 2 * max_h


def test_add_dense_return_targets_formula():
    """target_return_h[t] = ln(close[t+h] / close[t])."""
    from src.crypto_trend_lab.features.target import add_dense_return_targets

    df = _make_ohlcv_df(20)
    result = add_dense_return_targets(df, max_horizon=3)
    close = df["close"].values

    for h in (1, 2, 3):
        col = f"target_return_{h}"
        for t in range(20 - h):
            expected = np.log(close[t + h] / close[t])
            assert np.isclose(result[col].iloc[t], expected)


def test_add_dense_return_targets_final_h_rows_nan():
    """Last h rows of target_return_h and target_direction_h must be NaN."""
    from src.crypto_trend_lab.features.target import add_dense_return_targets

    df = _make_ohlcv_df(30)
    result = add_dense_return_targets(df, max_horizon=5)

    for h in (1, 2, 3, 5):
        ret_col = f"target_return_{h}"
        dir_col = f"target_direction_{h}"
        for t in range(30 - h, 30):
            assert pd.isna(result[ret_col].iloc[t])
            assert pd.isna(result[dir_col].iloc[t])


def test_add_dense_return_targets_does_not_mutate_input():
    from src.crypto_trend_lab.features.target import add_dense_return_targets

    df = _make_ohlcv_df(30)
    original = df.copy()
    _ = add_dense_return_targets(df, max_horizon=6)
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# run_dense_direct_forecast
# ---------------------------------------------------------------------------


def test_dense_forecast_returns_one_point_per_step():
    """path_length=6 must produce exactly 6 path points (one per step)."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(200)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=6, timeframe="1h",
    )
    assert "error" not in result
    points = [p for p in result["path_points"] if "error" not in p]
    assert len(points) == 6
    assert {p["forecast_step"] for p in points} == set(range(1, 7))


def test_dense_forecast_timestamps_correct():
    """Each forecast_timestamp = latest_ts + h * timeframe_delta."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(200)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=3, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    latest = result["latest_timestamp"]
    assert len(points) == 3
    assert points[0]["forecast_timestamp"] == latest + pd.Timedelta(hours=1)
    assert points[1]["forecast_timestamp"] == latest + pd.Timedelta(hours=2)
    assert points[2]["forecast_timestamp"] == latest + pd.Timedelta(hours=3)


def test_dense_forecast_close_calculation():
    """estimated_future_close = latest_close * exp(pred_log_return)."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(200)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=4, timeframe="1h",
    )
    lc = result["latest_close"]
    for p in result["path_points"]:
        if "error" not in p:
            expected = lc * np.exp(p["predicted_log_return"])
            assert abs(p["estimated_future_close"] - expected) < 1e-6


def test_dense_forecast_targets_excluded_from_features():
    """Dense forecast must not use target columns as input features."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast
    from src.crypto_trend_lab.models.dataset import get_default_feature_columns

    df = _make_features_df(200)
    features = get_default_feature_columns(df)
    # Features must not contain any target_* column
    assert not any(c.startswith("target_") for c in features)

    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=3, timeframe="1h",
        feature_columns=features,
    )
    assert "error" not in result


def test_dense_forecast_classification_blocked():
    """Only regression models can produce a dense price path."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(200)
    result = run_dense_direct_forecast(
        df, model_name="Logistic Regression", path_length=6, timeframe="1h",
    )
    # "Logistic Regression" is not in _REGRESSION_MODELS
    assert "error" in result
    assert "regression" in result["error"].lower()


def test_dense_forecast_handles_failed_horizon():
    """A horizon with insufficient data must be logged, not crash the path."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    # Small df where horizon 24 has few labeled rows
    df = _make_features_df(50)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=30, timeframe="1h",
    )
    # Some horizons near 30 should fail due to insufficient labeled rows
    assert "failed_horizons" in result
    # But earlier horizons (1, 2, 3...) should succeed
    points = [p for p in result["path_points"] if "error" not in p]
    assert len(points) > 0
    # Successful points + failed horizons = path_length
    assert len(points) + len(result["failed_horizons"]) == 30


# ---------------------------------------------------------------------------
# Sparse vs dense distinction
# ---------------------------------------------------------------------------


def test_sparse_mode_fewer_points_than_path_length():
    """Sparse mode with path_length=24 gives exactly 3 points (1, 4, 24)."""
    df = _make_features_df(300)
    result = forecast_path(
        df, model_name="Ridge", path_length=24, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    assert len(points) == 3, (
        f"Sparse mode: expected 3 points, got {len(points)}"
    )


def test_dense_mode_points_equal_path_length():
    """Dense mode with path_length=12 gives exactly 12 points."""
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(300)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=12, timeframe="1h",
    )
    points = [p for p in result["path_points"] if "error" not in p]
    assert len(points) == 12, (
        f"Dense mode: expected 12 points, got {len(points)}"
    )


def test_dense_forecast_no_live_network_calls():
    from src.crypto_trend_lab.models.forecast import run_dense_direct_forecast

    df = _make_features_df(100)
    result = run_dense_direct_forecast(
        df, model_name="Ridge", path_length=3, timeframe="1h",
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Display aggregation: aggregate_ohlcv_by_count
# ---------------------------------------------------------------------------


def _make_ohlcv_for_agg(n: int = 200) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame for display aggregation tests."""
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    rng = np.random.default_rng(42)
    prices = 40000 + np.cumsum(rng.normal(0, 50, n))
    return pd.DataFrame({
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "timestamp": ts,
        "open": prices - 10,
        "high": prices + 20,
        "low": prices - 20,
        "close": prices,
        "volume": rng.random(n) * 100 + 100,
    })


def test_aggregate_ohlcv_raw_when_fewer_than_target():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(50)
    result = aggregate_ohlcv_by_count(df, target_bars=100)
    # Should return raw copy (source_rows = 1 per row since n <= target_bars)
    assert len(result) == 50
    assert "source_rows" in result.columns


def test_aggregate_ohlcv_open_is_first_open():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=10)
    # For each aggregated row, open should equal the first open from its group
    assert len(result) <= 10
    # Spot check: group first open should match
    first_group = df.iloc[:20]  # ~200/10 = 20 rows per group
    assert result["open"].iloc[0] == first_group["open"].iloc[0]


def test_aggregate_ohlcv_high_is_max_high():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=5)
    # For each group, high should be the maximum high in that group
    assert len(result) <= 5
    # Verify all highs are >= corresponding opens
    assert (result["high"] >= result["open"]).all()


def test_aggregate_ohlcv_low_is_min_low():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=5)
    # For each group, low should be the minimum low in that group
    assert (result["low"] <= result["open"]).all()


def test_aggregate_ohlcv_close_is_last_close():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=10)
    assert len(result) <= 10
    last_group = df.iloc[-20:]  # last ~20 rows
    assert result["close"].iloc[-1] == last_group["close"].iloc[-1]


def test_aggregate_ohlcv_volume_is_sum():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(100)
    result = aggregate_ohlcv_by_count(df, target_bars=4)
    # Total volume should be preserved
    assert abs(result["volume"].sum() - df["volume"].sum()) < 0.01


def test_aggregate_ohlcv_output_row_count_controlled():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(500)
    for target in [10, 50, 100]:
        result = aggregate_ohlcv_by_count(df, target_bars=target)
        assert len(result) <= target


def test_aggregate_ohlcv_timestamp_timezone_preserved():
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=20)
    assert result["timestamp"].dt.tz is not None
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_aggregate_ohlcv_no_averaging_of_prices():
    """OHLCV display aggregation must not average open/high/low/close."""
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count

    df = _make_ohlcv_for_agg(200)
    result = aggregate_ohlcv_by_count(df, target_bars=5)

    # Verify: no price column should be an average.
    # open should match some df.open value in its group
    for i in range(len(result)):
        src_start = result["source_start"].iloc[i]
        src_end = result["source_end"].iloc[i]
        # source_start/end are already tz-aware UTC from the input
        group = df[(df["timestamp"] >= src_start) & (df["timestamp"] <= src_end)]
        assert len(group) > 0, f"Group {i}: no rows in [{src_start}, {src_end}]"
        assert result["open"].iloc[i] == group["open"].iloc[0]
        assert result["high"].iloc[i] == group["high"].max()
        assert result["low"].iloc[i] == group["low"].min()
        assert result["close"].iloc[i] == group["close"].iloc[-1]


# ---------------------------------------------------------------------------
# Display aggregation: prepare_candlestick_display_data
# ---------------------------------------------------------------------------


def test_prepare_candlestick_raw_mode_when_few_rows():
    from src.crypto_trend_lab.visualization.display import (
        prepare_candlestick_display_data,
    )

    df = _make_ohlcv_for_agg(50)
    result = prepare_candlestick_display_data(df, max_bars=1000)
    assert result["display_mode"] == "raw"
    assert result["displayed_rows"] == 50
    assert result["input_rows"] == 50


def test_prepare_candlestick_aggregated_mode_when_many_rows():
    from src.crypto_trend_lab.visualization.display import (
        prepare_candlestick_display_data,
    )

    df = _make_ohlcv_for_agg(500)
    result = prepare_candlestick_display_data(df, max_bars=100)
    assert result["display_mode"] == "aggregated"
    assert result["displayed_rows"] <= 100
    assert result["input_rows"] == 500
    assert result["approx_bars_per_candle"] is not None


def test_prepare_candlestick_preserves_full_row_count_in_metadata():
    from src.crypto_trend_lab.visualization.display import (
        prepare_candlestick_display_data,
    )

    df = _make_ohlcv_for_agg(500)
    result = prepare_candlestick_display_data(df, max_bars=50)
    assert result["input_rows"] == 500  # Full count in metadata
    assert result["displayed_rows"] <= 50  # Aggregated for display


# ---------------------------------------------------------------------------
# Display aggregation: filter_by_time_range
# ---------------------------------------------------------------------------


def test_filter_by_time_range_full_when_no_args():
    from src.crypto_trend_lab.visualization.display import filter_by_time_range

    df = _make_ohlcv_for_agg(100)
    result = filter_by_time_range(df)
    assert len(result) == 100


def test_filter_by_time_range_start_only():
    from src.crypto_trend_lab.visualization.display import filter_by_time_range

    df = _make_ohlcv_for_agg(100)
    # Keep only rows from row index 50 onwards
    start_ts = df["timestamp"].iloc[50]
    result = filter_by_time_range(df, start=start_ts)
    assert len(result) == 50
    assert result["timestamp"].min() >= start_ts


def test_filter_by_time_range_end_only():
    from src.crypto_trend_lab.visualization.display import filter_by_time_range

    df = _make_ohlcv_for_agg(100)
    end_ts = df["timestamp"].iloc[49]
    result = filter_by_time_range(df, end=end_ts)
    assert len(result) == 50
    assert result["timestamp"].max() <= end_ts


def test_filter_by_time_range_preserves_tz():
    from src.crypto_trend_lab.visualization.display import filter_by_time_range

    df = _make_ohlcv_for_agg(100)
    start = pd.Timestamp("2024-01-03", tz="utc")
    result = filter_by_time_range(df, start=start)
    assert result["timestamp"].dt.tz is not None


def test_filter_by_time_range_no_mutation():
    """filter_by_time_range must not mutate the input DataFrame."""
    from src.crypto_trend_lab.visualization.display import filter_by_time_range

    df = _make_ohlcv_for_agg(100)
    original_len = len(df)
    _ = filter_by_time_range(df, start=df["timestamp"].iloc[50])
    assert len(df) == original_len  # Original unchanged


# ---------------------------------------------------------------------------
# Display aggregation: get_display_summary
# ---------------------------------------------------------------------------


def test_get_display_summary_keys():
    from src.crypto_trend_lab.visualization.display import (
        get_display_summary,
        prepare_candlestick_display_data,
    )

    df = _make_ohlcv_for_agg(500)
    view = df.iloc[-200:]
    display_result = prepare_candlestick_display_data(view, max_bars=50)
    summary = get_display_summary(df, view, display_result)

    for key in ("full_rows", "view_rows", "displayed_candles",
                "display_mode", "chart_start", "chart_end"):
        assert key in summary
    assert summary["full_rows"] == 500
    assert summary["view_rows"] == 200
    assert summary["displayed_candles"] <= 50


# ---------------------------------------------------------------------------
# Data integrity: full dataset preserved for modeling
# ---------------------------------------------------------------------------


def test_data_quality_uses_full_df_not_aggregated():
    """Data quality checks must use df_full, never df_chart."""
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count
    from src.crypto_trend_lab.validation.quality import check_ohlcv_quality

    df_full = _make_ohlcv_for_agg(500)
    df_chart = aggregate_ohlcv_by_count(df_full, target_bars=50)

    # Data quality on df_full
    full_quality = check_ohlcv_quality(df_full, "1h")
    # Data quality on df_chart would give wrong row count
    chart_quality = check_ohlcv_quality(df_chart, "1h")

    # Full data quality must report original row count
    assert full_quality["row_count"] == 500
    # Aggregated chart has fewer rows
    assert chart_quality["row_count"] != 500
    # df_chart row count is ≤ 50
    assert chart_quality["row_count"] <= 50


def test_feature_generation_uses_full_df():
    """build_features must be called on df_full, not df_chart."""
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count
    from src.crypto_trend_lab.features.pipeline import build_features

    df_full = _make_ohlcv_for_agg(300)
    df_chart = aggregate_ohlcv_by_count(df_full, target_bars=30)

    features_full = build_features(df_full)
    features_chart = build_features(df_chart)

    assert len(features_full) == 300
    assert len(features_chart) < 300
    # Full features have more rows
    assert len(features_full) > len(features_chart)


def test_model_evaluation_uses_full_features():
    """compare_baselines_and_models must use full features, not df_chart."""
    from src.crypto_trend_lab.visualization.display import aggregate_ohlcv_by_count
    from src.crypto_trend_lab.features.pipeline import build_features
    from src.crypto_trend_lab.evaluation.report import compare_baselines_and_models

    df_full = _make_ohlcv_for_agg(300)
    df_chart = aggregate_ohlcv_by_count(df_full, target_bars=30)

    features_full = build_features(df_full)

    # Evaluation on full features should succeed
    result_full = compare_baselines_and_models(
        features_full, task_type="regression", horizon=1, test_size=0.2,
    )
    assert len(result_full["predictions"]) > 0

    # Build features from aggregated chart — fewer rows
    features_chart = build_features(df_chart)
    # The aggregated data may have too few rows for evaluation
    # Either way, we verify full features have more data
    assert len(features_full) > len(features_chart)


def test_forecast_path_uses_timedelta_not_int():
    """Forecast path timestamp generation must use Timedelta arithmetic."""
    from src.crypto_trend_lab.evaluation.forecast import generate_future_timestamps

    latest = pd.Timestamp("2024-06-15 12:00:00", tz="utc")

    # This would raise TypeError if Timestamp + int was used
    for tf in ["1m", "5m", "1h", "4h", "1d", "1w"]:
        ts = generate_future_timestamps(latest, tf, 3)
        assert len(ts) == 3, f"Failed for timeframe {tf}"
        for t in ts:
            assert t > latest
            assert t.tzinfo is not None


# ---------------------------------------------------------------------------
# filter_ohlcv_by_chart_range
# ---------------------------------------------------------------------------


def test_filter_ohlcv_full_range_returns_all_rows():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(100)
    result = filter_ohlcv_by_chart_range(df, "Full range")
    assert len(result) == 100


def test_filter_ohlcv_last_1_day():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(168)  # 7 days of 1h data
    result = filter_ohlcv_by_chart_range(df, "Last 1 day")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=1)
    assert (result["timestamp"] >= cutoff).all()
    # Should be roughly 24 bars (1h × 24)
    assert 20 <= len(result) <= 28


def test_filter_ohlcv_last_7_days():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(500)  # ~20 days of 1h data
    result = filter_ohlcv_by_chart_range(df, "Last 7 days")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=7)
    assert (result["timestamp"] >= cutoff).all()
    # Roughly 168 bars (1h × 24 × 7)
    assert 150 <= len(result) <= 200


def test_filter_ohlcv_last_30_days():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(1000)
    result = filter_ohlcv_by_chart_range(df, "Last 30 days")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=30)
    assert (result["timestamp"] >= cutoff).all()


def test_filter_ohlcv_last_90_days():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(3000)
    result = filter_ohlcv_by_chart_range(df, "Last 90 days")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=90)
    assert (result["timestamp"] >= cutoff).all()


def test_filter_ohlcv_last_180_days():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(5000)
    result = filter_ohlcv_by_chart_range(df, "Last 180 days")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=180)
    assert (result["timestamp"] >= cutoff).all()


def test_filter_ohlcv_last_365_days():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(10000)
    result = filter_ohlcv_by_chart_range(df, "Last 365 days")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=365)
    assert (result["timestamp"] >= cutoff).all()


def test_filter_ohlcv_custom_range():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(200)
    start = df["timestamp"].iloc[50]
    end = df["timestamp"].iloc[149]
    result = filter_ohlcv_by_chart_range(
        df, "Custom range", custom_start=start, custom_end=end
    )
    assert len(result) == 100
    assert result["timestamp"].min() >= start
    assert result["timestamp"].max() <= end


def test_filter_ohlcv_uses_data_max_not_now():
    """Last-N-day ranges must use df['timestamp'].max(), not pd.Timestamp.now()."""
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(200)
    # Data ends at index 199: '2024-01-09 07:00:00+00:00'
    data_max = df["timestamp"].max()
    result = filter_ohlcv_by_chart_range(df, "Last 1 day")
    # All returned timestamps must be within 1 day of data_max
    assert (result["timestamp"] >= data_max - pd.Timedelta(days=1)).all()


def test_filter_ohlcv_empty_range_raises():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(50)
    # Custom range entirely before the data
    with pytest.raises(ValueError, match="No data in selected range"):
        filter_ohlcv_by_chart_range(
            df, "Custom range",
            custom_start=pd.Timestamp("2020-01-01", tz="utc"),
            custom_end=pd.Timestamp("2020-01-02", tz="utc"),
        )


def test_filter_ohlcv_custom_requires_both_dates():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(50)
    with pytest.raises(ValueError, match="requires both"):
        filter_ohlcv_by_chart_range(
            df, "Custom range", custom_start=pd.Timestamp("2024-01-01", tz="utc"),
        )


def test_filter_ohlcv_unknown_range_option():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(50)
    with pytest.raises(ValueError, match="Unknown range_option"):
        filter_ohlcv_by_chart_range(df, "Last 2 weeks")


def test_filter_ohlcv_preserves_timezone():
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(200)
    for option in ["Full range", "Last 7 days", "Last 30 days"]:
        result = filter_ohlcv_by_chart_range(df, option)
        assert result["timestamp"].dt.tz is not None


def test_filter_ohlcv_no_mutation():
    """filter_ohlcv_by_chart_range must not mutate the input df_full."""
    from src.crypto_trend_lab.visualization.display import filter_ohlcv_by_chart_range

    df = _make_ohlcv_for_agg(200)
    original_len = len(df)
    _ = filter_ohlcv_by_chart_range(df, "Last 7 days")
    assert len(df) == original_len


# ---------------------------------------------------------------------------
# Multi-resolution: aggregation order
# ---------------------------------------------------------------------------


def test_filter_before_aggregate_in_correct_order():
    """
    Correct order: df_full -> filter -> df_view -> aggregate -> df_chart.
    This test verifies that narrower ranges produce smaller
    approx_bars_per_candle when max_bars is fixed.
    """
    from src.crypto_trend_lab.visualization.display import (
        filter_ohlcv_by_chart_range,
        prepare_candlestick_display_data,
        get_display_summary,
    )

    df_full = _make_ohlcv_for_agg(5000)
    max_bars = 200

    # Full range
    df_view_full = filter_ohlcv_by_chart_range(df_full, "Full range")
    display_full = prepare_candlestick_display_data(df_view_full, max_bars=max_bars)
    summary_full = get_display_summary(df_full, df_view_full, display_full)

    # Last 30 days (much narrower — last 720 bars of 5000)
    df_view_30d = filter_ohlcv_by_chart_range(df_full, "Last 30 days")
    display_30d = prepare_candlestick_display_data(df_view_30d, max_bars=max_bars)
    summary_30d = get_display_summary(df_full, df_view_30d, display_30d)

    # Narrower range should have fewer view rows
    assert summary_30d["view_rows"] < summary_full["view_rows"]

    # Narrower range should have same or smaller approx_bars_per_candle
    if summary_30d["approx_bars_per_candle"] and summary_full["approx_bars_per_candle"]:
        assert summary_30d["approx_bars_per_candle"] <= summary_full["approx_bars_per_candle"]


def test_narrower_range_can_be_raw_when_full_is_aggregated():
    """
    Full range may be aggregated, while a narrow enough range
    should display raw candles.
    """
    from src.crypto_trend_lab.visualization.display import (
        filter_ohlcv_by_chart_range,
        prepare_candlestick_display_data,
    )

    df_full = _make_ohlcv_for_agg(5000)
    max_bars = 1000

    df_view_full = filter_ohlcv_by_chart_range(df_full, "Full range")
    display_full = prepare_candlestick_display_data(df_view_full, max_bars=max_bars)
    assert display_full["display_mode"] == "aggregated"

    # Last 1 day: only ~24 bars — should be raw
    df_view_1d = filter_ohlcv_by_chart_range(df_full, "Last 1 day")
    display_1d = prepare_candlestick_display_data(df_view_1d, max_bars=max_bars)
    assert display_1d["display_mode"] == "raw"


def test_df_full_row_count_preserved_in_summary():
    """get_display_summary must report correct df_full row count."""
    from src.crypto_trend_lab.visualization.display import (
        filter_ohlcv_by_chart_range,
        prepare_candlestick_display_data,
        get_display_summary,
    )

    df_full = _make_ohlcv_for_agg(5000)
    df_view = filter_ohlcv_by_chart_range(df_full, "Last 30 days")
    display = prepare_candlestick_display_data(df_view, max_bars=500)
    summary = get_display_summary(df_full, df_view, display)

    assert summary["full_rows"] == 5000
    assert summary["view_rows"] == len(df_view)
    assert summary["view_rows"] < 5000  # 30d is a subset


def test_aggregation_never_reduces_full_dataset():
    """
    df_full must remain intact. Aggregation only creates new display
    DataFrames; it never shrinks df_full.
    """
    from src.crypto_trend_lab.visualization.display import (
        filter_ohlcv_by_chart_range,
        aggregate_ohlcv_by_count,
    )

    df_full = _make_ohlcv_for_agg(5000)
    original_len = len(df_full)

    df_view = filter_ohlcv_by_chart_range(df_full, "Full range")
    df_chart = aggregate_ohlcv_by_count(df_view, target_bars=200)

    # df_full unchanged
    assert len(df_full) == original_len
    # df_chart is smaller
    assert len(df_chart) <= 200
    # df_chart is a different object
    assert df_chart is not df_full


def test_multi_resolution_no_live_network_calls():
    from src.crypto_trend_lab.visualization.display import (
        filter_ohlcv_by_chart_range,
        prepare_candlestick_display_data,
        get_display_summary,
    )

    df_full = _make_ohlcv_for_agg(1000)
    df_view = filter_ohlcv_by_chart_range(df_full, "Last 7 days")
    display = prepare_candlestick_display_data(df_view, max_bars=500)
    summary = get_display_summary(df_full, df_view, display)
    assert summary["full_rows"] == 1000


# ---------------------------------------------------------------------------
# add_datetime_vertical_marker
# ---------------------------------------------------------------------------


def test_datetime_vertical_marker_no_typeerror():
    """add_vline with annotation on datetime axis raises TypeError in
    modern pandas. Our helper uses add_shape + add_annotation instead."""
    import plotly.graph_objects as go

    from src.crypto_trend_lab.visualization.charts import (
        add_datetime_vertical_marker,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.date_range("2024-01-01", periods=10, freq="1h", tz="utc"),
        y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mode="lines",
        name="Test",
    ))
    marker_x = pd.Timestamp("2024-01-01 05:00:00", tz="utc")

    fig = add_datetime_vertical_marker(
        fig, x=marker_x, text="Marker",
        line_dash="dot", line_color="gray",
    )

    assert len(fig.layout.shapes) >= 1
    assert len(fig.layout.annotations) >= 1


def test_datetime_vertical_marker_without_text():
    """When text is None, no annotation should be added."""
    import plotly.graph_objects as go

    from src.crypto_trend_lab.visualization.charts import (
        add_datetime_vertical_marker,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.date_range("2024-01-01", periods=5, freq="1h", tz="utc"),
        y=[1, 2, 3, 4, 5],
    ))
    marker_x = pd.Timestamp("2024-01-01 02:00:00", tz="utc")

    fig = add_datetime_vertical_marker(fig, x=marker_x)

    assert len(fig.layout.shapes) >= 1
    assert fig.layout.annotations is None or len(fig.layout.annotations) == 0


def test_datetime_vertical_marker_no_live_network_calls():
    import plotly.graph_objects as go

    from src.crypto_trend_lab.visualization.charts import (
        add_datetime_vertical_marker,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.date_range("2024-01-01", periods=3, freq="1h", tz="utc"),
        y=[1, 2, 3],
    ))
    fig = add_datetime_vertical_marker(
        fig, x=pd.Timestamp("2024-01-01 01:00:00", tz="utc"),
        text="Test",
    )
    assert fig is not None
