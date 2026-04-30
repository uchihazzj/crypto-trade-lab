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
    _HAS_LIGHTGBM,
    LightGBMClassifier,
    LightGBMRegressor,
    LogisticRegressionModel,
    RidgeRegressionModel,
)


def _build_baselines(task_type: str) -> list[tuple[str, object]]:
    """Build the list of baseline models for a task type."""
    if task_type == "regression":
        return [
            ("Zero Return", ZeroReturnBaseline()),
            ("Last Return", LastReturnBaseline()),
            ("Moving Average", MovingAverageReturnBaseline()),
        ]
    else:
        return [
            ("Momentum Direction", MomentumDirectionBaseline()),
            ("Majority Class", MajorityClassBaseline()),
        ]


def _build_tabular_models(task_type: str) -> list[tuple[str, object]]:
    """Build the list of tabular models for a task type."""
    models: list[tuple[str, object]] = []
    if task_type == "regression":
        models.append(("Ridge", RidgeRegressionModel()))
    else:
        models.append(("Logistic Regression", LogisticRegressionModel()))

    if _HAS_LIGHTGBM:
        if task_type == "regression":
            models.append(("LightGBM", LightGBMRegressor()))
        else:
            models.append(("LightGBM", LightGBMClassifier()))
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


def compare_baselines_and_models(
    df: pd.DataFrame,
    task_type: str = "regression",
    horizon: int = 1,
    test_size: int | float = 0.2,
    feature_columns: list[str] | None = None,
    include_tabular: bool = True,
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
        If True, also run Ridge/Logistic/LightGBM models.

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

    for name, model in _build_baselines(task_type):
        result = evaluate_model(
            model, name, X_train, y_train, X_test, y_test, task_type
        )
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
        for name, model in _build_tabular_models(task_type):
            result = evaluate_model(
                model, name, X_train, y_train, X_test, y_test, task_type
            )
            results.append(result)
            predictions_list.append(
                _make_prediction_df(
                    test_timestamps, result["y_true"], result["y_pred"],
                    name, target_column,
                    y_prob=result.get("y_prob"),
                )
            )

    # Build metrics table
    metrics_table = _build_metrics_table(results)

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
        "predictions": pd.concat(predictions_list, ignore_index=True),
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


def _build_metrics_table(results: list[dict]) -> pd.DataFrame:
    """Build a metrics comparison DataFrame from evaluation results."""
    rows = []
    for r in results:
        row = {"model_name": r["model_name"]}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
