"""Prediction targets for crypto OHLCV data.

Targets use future close prices and must never be used as input features.
Rows with unavailable future targets remain NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "target_return_1",
    "target_return_4",
    "target_return_24",
    "target_direction_1",
    "target_direction_4",
    "target_direction_24",
]


def add_prediction_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add prediction targets to a new DataFrame.

    Targets are forward-looking log returns and binary directions:
      target_return_h = ln(close[t+h] / close[t])
      target_direction_h = 1 if target_return_h > 0 else 0

    The final *h* rows for each target_return_h are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column: close.

    Returns
    -------
    pd.DataFrame
        Target columns keyed to the same index.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    for h in (1, 4, 24):
        future_close = close.shift(-h)
        target_return = np.log(future_close / close)
        result[f"target_return_{h}"] = target_return
        result[f"target_direction_{h}"] = np.where(
            pd.notna(target_return),
            (target_return > 0).astype(float),
            np.nan,
        )

    # Reorder columns to match TARGET_COLUMNS convention
    return result[TARGET_COLUMNS]


def add_dense_return_targets(
    df: pd.DataFrame,
    max_horizon: int,
) -> pd.DataFrame:
    """Add dense return and direction targets for horizons 1..*max_horizon*.

    For each horizon *h*, creates::

        target_return_h    = ln(close[t+h] / close[t])
        target_direction_h = 1 if target_return_h > 0 else 0

    The final *h* rows of ``target_return_h`` and ``target_direction_h``
    are NaN.  Does **not** mutate *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``close`` column.
    max_horizon : int
        Number of forward horizons to generate.

    Returns
    -------
    pd.DataFrame
        Columns: ``target_return_1`` ... ``target_return_max_horizon``,
        ``target_direction_1`` ... ``target_direction_max_horizon``.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    for h in range(1, max_horizon + 1):
        future_close = close.shift(-h)
        target_return = np.log(future_close / close)
        result[f"target_return_{h}"] = target_return
        result[f"target_direction_{h}"] = np.where(
            pd.notna(target_return),
            (target_return > 0).astype(float),
            np.nan,
        )

    return result
