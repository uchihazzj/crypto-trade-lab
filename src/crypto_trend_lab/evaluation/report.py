"""Model evaluation workflow.

Orchestrates chronological splitting, model fitting, metric computation,
and result collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from src.crypto_trend_lab.evaluation.split import chronological_train_test_split
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
    _HAS_CATBOOST,
    _HAS_LIGHTGBM,
    _HAS_XGBOOST,
    CatBoostClassifier,
    CatBoostRegressor,
    ElasticNetRegressionModel,
    ExtraTreesClassificationModel,
    ExtraTreesRegressionModel,
    HistGradientBoostingClassificationModel,
    HistGradientBoostingRegressionModel,
    LightGBMClassifier,
    LightGBMRegressor,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestRegressionModel,
    RidgeRegressionModel,
    XGBoostClassifier,
    XGBoostRegressor,
)


def _build_baselines(task_type: str) -> list[tuple[str, object]]:
    """Build the list of naive baseline models for a task type."""
    if task_type == "regression":
        return [
            ("Zero Return", ZeroReturnBaseline()),
            ("Last Return", LastReturnBaseline()),
            ("Moving Average", MovingAverageReturnBaseline()),
            ("Historical Mean Return", HistoricalMeanReturnBaseline()),
        ]
    else:
        return [
            ("Momentum Direction", MomentumDirectionBaseline()),
            ("Majority Class", MajorityClassBaseline()),
        ]


def _build_linear_models(task_type: str) -> list[tuple[str, object]]:
    """Build linear / regularized models."""
    if task_type == "regression":
        return [
            ("Ridge", RidgeRegressionModel()),
            ("ElasticNet", ElasticNetRegressionModel()),
        ]
    else:
        return [
            ("Logistic Regression", LogisticRegressionModel()),
        ]


def _build_tree_models(task_type: str) -> list[tuple[str, object]]:
    """Build sklearn tree-ensemble models."""
    if task_type == "regression":
        return [
            ("Random Forest", RandomForestRegressionModel()),
            ("Extra Trees", ExtraTreesRegressionModel()),
            ("HistGradientBoosting", HistGradientBoostingRegressionModel()),
        ]
    else:
        return [
            ("Random Forest", RandomForestClassificationModel()),
            ("Extra Trees", ExtraTreesClassificationModel()),
            ("HistGradientBoosting", HistGradientBoostingClassificationModel()),
        ]


def _build_external_models(task_type: str) -> list[tuple[str, object]]:
    """Build optional external-library models (LightGBM, XGBoost, CatBoost)."""
    models: list[tuple[str, object]] = []
    if _HAS_LIGHTGBM:
        if task_type == "regression":
            models.append(("LightGBM", LightGBMRegressor()))
        else:
            models.append(("LightGBM", LightGBMClassifier()))
    if _HAS_XGBOOST:
        if task_type == "regression":
            models.append(("XGBoost", XGBoostRegressor()))
        else:
            models.append(("XGBoost", XGBoostClassifier()))
    if _HAS_CATBOOST:
        if task_type == "regression":
            models.append(("CatBoost", CatBoostRegressor()))
        else:
            models.append(("CatBoost", CatBoostClassifier()))
    return models


def _build_tabular_models(
    task_type: str,
    include_trees: bool = False,
) -> list[tuple[str, object]]:
    """Build the list of tabular models for a task type.

    Parameters
    ----------
    task_type : str
    include_trees : bool
        If True, include tree ensembles and external-library models.
        If False (default), only linear models + LightGBM.
    """
    models: list[tuple[str, object]] = []
    models.extend(_build_linear_models(task_type))

    if _HAS_LIGHTGBM:
        if task_type == "regression":
            models.append(("LightGBM", LightGBMRegressor()))
        else:
            models.append(("LightGBM", LightGBMClassifier()))

    if include_trees:
        models.extend(_build_tree_models(task_type))
        # External models (XGBoost, CatBoost) included with trees
        for name, model in _build_external_models(task_type):
            if name != "LightGBM":  # already added above
                models.append((name, model))

    return models


def evaluate_model(
    model: object,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> dict:
    """Fit a model on train, predict on test, compute metrics.

    Parameters
    ----------
    model : object
        Any object with ``fit(X, y)`` and ``predict(X)`` methods.
    model_name : str
        Display name for this model.
    X_train, y_train, X_test, y_test : np.ndarray
        Training and test data.
    task_type : str
        "regression" or "classification".

    Returns
    -------
    dict
        Keys: model_name, y_true, y_pred, y_prob (classifier only), metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    if task_type == "regression":
        metrics = regression_metrics(y_test, y_pred)
    else:
        metrics = classification_metrics(y_test, y_pred, y_prob)

    result: dict = {
        "model_name": model_name,
        "y_true": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }
    if y_prob is not None:
        result["y_prob"] = y_prob

    return result


# Stable column schema for empty predictions / metrics DataFrames.
_PREDICTIONS_COLUMNS = ["timestamp", "y_true", "y_pred", "model_name", "target_column"]
_METRICS_REGRESSION_COLUMNS = ["mae", "rmse", "directional_accuracy", "spearman_r"]
_METRICS_CLASSIFICATION_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc",
]


def compare_baselines_and_models(
    df: pd.DataFrame,
    task_type: str = "regression",
    horizon: int = 1,
    test_size: int | float = 0.2,
    feature_columns: list[str] | None = None,
    include_tabular: bool = True,
    include_trees: bool = False,
    model_names: list[str] | None = None,
) -> dict:
    """Run baselines and tabular models on a chronological split.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.
    task_type : str
        "regression" or "classification".
    horizon : int
        Forecast horizon: 1, 4, or 24.
    test_size : int or float
        Number or fraction of rows to hold out for testing.
    feature_columns : list[str] or None
        Feature columns to use. If None, auto-detected from *df*.
    include_tabular : bool
        If True, run linear models + LightGBM.
    include_trees : bool
        If True, also run tree ensembles and optional external models
        (XGBoost, CatBoost if installed). Default False for speed.
    model_names : list[str] or None
        If provided, run only models whose display names are in this list.
        If None, run all applicable models.

    Returns
    -------
    dict
        Keys: task_type, horizon, target_column, feature_columns,
        train_dates (start, end), test_dates (start, end),
        metrics_table (pd.DataFrame), predictions (pd.DataFrame).
    """
    target_column = get_target_column(task_type, horizon)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not found in DataFrame. "
            f"Run build_features() first."
        )

    if feature_columns is None:
        feature_columns = get_default_feature_columns(df)

    X, y, timestamps, features = prepare_modeling_data(
        df, target_column, feature_columns
    )

    if len(X) < 2:
        raise ValueError(
            f"Insufficient data after NaN removal: {len(X)} rows. "
            f"Need at least 2 rows for train/test split."
        )

    # Build temporary DataFrame for chronological split.
    # Use .to_numpy() to avoid index-alignment issues if timestamps
    # has a non-sequential index.
    full = pd.DataFrame({"timestamp": timestamps.to_numpy()})
    full["_x_idx"] = range(len(X))

    train_df, test_df = chronological_train_test_split(
        full, test_size=test_size, time_col="timestamp"
    )

    train_idx = train_df["_x_idx"].values
    test_idx = test_df["_x_idx"].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Use the sorted train_df/test_df timestamps directly for dates
    train_ts = train_df["timestamp"]
    test_ts = test_df["timestamp"]
    test_timestamps = test_ts.values

    # Run baselines
    results = []
    predictions_list = []
    skipped: list[dict] = []

    def _safe_eval(
        name: str, model: object, check_single_class: bool = False,
    ) -> dict | None:
        """Evaluate one model, returning None and recording reason on failure."""
        if check_single_class and task_type == "classification":
            train_classes = np.unique(y_train)
            if len(train_classes) < 2:
                skipped.append({
                    "model_name": name,
                    "task_type": task_type,
                    "horizon": horizon,
                    "target_column": target_column,
                    "reason": (
                        f"Single class ({int(train_classes[0])}) in training data. "
                        f"Cannot train a classifier."
                    ),
                })
                return None

        try:
            return evaluate_model(
                model, name, X_train, y_train, X_test, y_test, task_type
            )
        except Exception as exc:
            skipped.append({
                "model_name": name,
                "task_type": task_type,
                "horizon": horizon,
                "target_column": target_column,
                "reason": f"Model fit/predict failed: {exc}",
            })
            return None

    for name, model in _build_baselines(task_type):
        if model_names and name not in model_names:
            continue
        result = _safe_eval(name, model, check_single_class=False)
        if result is None:
            continue
        results.append(result)
        predictions_list.append(
            _make_prediction_df(
                test_timestamps, result["y_true"], result["y_pred"],
                name, target_column,
                y_prob=result.get("y_prob"),
            )
        )

    # Run tabular models
    if include_tabular:
        for name, model in _build_tabular_models(task_type, include_trees=include_trees):
            if model_names and name not in model_names:
                continue
            result = _safe_eval(name, model, check_single_class=True)
            if result is None:
                continue
            results.append(result)
            predictions_list.append(
                _make_prediction_df(
                    test_timestamps, result["y_true"], result["y_pred"],
                    name, target_column,
                    y_prob=result.get("y_prob"),
                )
            )

    # Build metrics table
    metrics_table = _build_metrics_table(results, task_type)

    # Build predictions DataFrame (handle empty list)
    if predictions_list:
        predictions_df = pd.concat(predictions_list, ignore_index=True)
    else:
        predictions_df = pd.DataFrame(columns=_PREDICTIONS_COLUMNS)

    return {
        "task_type": task_type,
        "horizon": horizon,
        "target_column": target_column,
        "feature_columns": features,
        "train_dates": {
            "start": train_ts.min(),
            "end": train_ts.max(),
        },
        "test_dates": {
            "start": test_ts.min(),
            "end": test_ts.max(),
        },
        "metrics_table": metrics_table,
        "predictions": predictions_df,
        "skipped": skipped,
    }


def _make_prediction_df(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_column: str,
    y_prob: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a tidy predictions DataFrame."""
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": y_pred,
            "model_name": model_name,
            "target_column": target_column,
        }
    )
    if y_prob is not None:
        df["y_prob"] = y_prob
    return df


def _build_metrics_table(
    results: list[dict], task_type: str = "regression",
) -> pd.DataFrame:
    """Build a metrics comparison DataFrame from evaluation results.

    Returns an empty DataFrame with stable columns when *results* is empty.
    """
    if not results:
        metric_cols = (
            _METRICS_REGRESSION_COLUMNS
            if task_type == "regression"
            else _METRICS_CLASSIFICATION_COLUMNS
        )
        return pd.DataFrame(columns=["model_name"] + list(metric_cols))

    rows = []
    for r in results:
        row = {"model_name": r["model_name"]}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
