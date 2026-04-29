"""Feature engineering pipeline combining technical indicators and targets."""

from __future__ import annotations

import pandas as pd

from src.crypto_trend_lab.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    add_technical_features,
)
from src.crypto_trend_lab.features.target import TARGET_COLUMNS, add_prediction_targets
from src.crypto_trend_lab.utils.helpers import OHLCV_COLUMNS, validate_ohlcv_schema

NUMERIC_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_ohlcv_columns() -> list[str]:
    """Return the stable OHLCV schema column names."""
    return list(OHLCV_COLUMNS)


def get_technical_feature_columns() -> list[str]:
    """Return the list of technical feature column names."""
    return list(TECHNICAL_FEATURE_COLUMNS)


def get_target_columns() -> list[str]:
    """Return the list of prediction target column names."""
    return list(TARGET_COLUMNS)


def get_model_input_columns() -> list[str]:
    """Return columns usable as model inputs.

    Includes numeric OHLCV columns and technical features.
    Excludes metadata columns (exchange, symbol, timeframe, timestamp)
    and prediction targets.
    """
    return NUMERIC_OHLCV_COLUMNS + TECHNICAL_FEATURE_COLUMNS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Validates the OHLCV schema, then adds all technical features and
    prediction targets. Does not mutate *df*.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with the stable schema.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with technical feature and target columns appended.
    """
    if df.empty:
        all_columns = (
            list(OHLCV_COLUMNS)
            + TECHNICAL_FEATURE_COLUMNS
            + TARGET_COLUMNS
        )
        return pd.DataFrame(columns=all_columns)

    validate_ohlcv_schema(df)

    base = df[OHLCV_COLUMNS].copy()
    features = add_technical_features(base)
    targets = add_prediction_targets(base)

    return pd.concat([base, features, targets], axis=1)
