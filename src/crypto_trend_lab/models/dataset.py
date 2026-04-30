"""Modeling dataset helpers.

Prepare X, y, timestamps and feature column lists from feature DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.features.pipeline import get_model_input_columns
from src.crypto_trend_lab.features.target import TARGET_COLUMNS


def get_default_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns in *df* suitable for model input.

    Intersects `get_model_input_columns()` with actual *df* columns
    and keeps only numeric columns.
    """
    candidates = [c for c in get_model_input_columns() if c in df.columns]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def get_target_column(task_type: str, horizon: int) -> str:
    """Return the canonical target column name.

    Parameters
    ----------
    task_type : str
        "regression" → target_return_{horizon}
        "classification" → target_direction_{horizon}
    horizon : int
        Forecast horizon (1, 4, or 24).
    """
    if task_type == "regression":
        col = f"target_return_{horizon}"
    elif task_type == "classification":
        col = f"target_direction_{horizon}"
    else:
        raise ValueError(
            f"Unknown task_type {task_type!r}. Use 'regression' or 'classification'."
        )

    if col not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target column {col!r} for horizon {horizon}")

    return col


def prepare_modeling_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Prepare X, y, timestamps, and feature columns for modeling.

    Drops rows where *target_column* or any feature column is NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.
    target_column : str
        Name of the target column.
    feature_columns : list[str] or None
        Feature columns to include. If None, uses `get_default_feature_columns`.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    timestamps : pd.DatetimeIndex
    features : list[str]
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not in DataFrame")

    if feature_columns is None:
        feature_columns = get_default_feature_columns(df)

    if not feature_columns:
        raise ValueError("No feature columns selected")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not in DataFrame: {missing}")

    # Exclude target columns from features (safety check)
    features = [c for c in feature_columns if c not in TARGET_COLUMNS]

    subset = df[features + [target_column]].copy()

    # Drop rows where target is NaN
    subset = subset.dropna(subset=[target_column])

    # Drop rows where any feature is NaN
    subset = subset.dropna(subset=features)

    X = subset[features].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    timestamps = df.loc[subset.index, "timestamp"]

    return X, y, timestamps, features
