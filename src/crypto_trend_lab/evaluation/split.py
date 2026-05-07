"""Chronological data splits for time-series evaluation.

All splits preserve temporal order. No shuffling. No future leakage.
"""

from __future__ import annotations

import pandas as pd


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: int | float = 0.2,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* chronologically into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sortable by *time_col*.
    test_size : int or float
        If int, number of test rows. If float, fraction of total rows.
    time_col : str
        Column to sort by before splitting.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        Train rows precede test rows chronologically.
    """
    n = len(df)

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size fraction must be in (0, 1), got {test_size}")
        n_test = max(1, int(n * test_size))
    else:
        n_test = test_size

    if n_test < 1:
        raise ValueError(f"Test size must be at least 1 row, got {n_test}")
    if n_test >= n:
        raise ValueError(
            f"Test size {n_test} >= total rows {n}. Need at least 1 train row."
        )

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    # Validate timestamps are monotonic after sorting
    if not df_sorted[time_col].is_monotonic_increasing:
        raise ValueError(
            f"Timestamps are not monotonic after sorting by {time_col!r}. "
            f"Check for null or duplicate timestamp values."
        )

    train = df_sorted.iloc[:-n_test]
    test = df_sorted.iloc[-n_test:]

    # Defensive: train must be strictly before test
    if len(train) > 0 and len(test) > 0:
        train_max = train[time_col].max()
        test_min = test[time_col].min()
        if train_max >= test_min:
            raise ValueError(
                f"Chronological order violated: train max {train_max} >= "
                f"test min {test_min}. Check for null timestamps."
            )

    return train, test


def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int = 1,
    time_col: str = "timestamp",
):
    """Generate chronological walk-forward train/test folds.

    Each fold produces non-overlapping test windows that march forward
    in time. Train windows include all data up to the start of the test
    window.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sortable by *time_col*.
    train_size : int
        Minimum number of rows in each train fold.
    test_size : int
        Number of rows in each test fold.
    step_size : int
        Number of rows to advance the test window each fold.
    time_col : str
        Column to sort by.

    Yields
    ------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    n = len(df)

    if train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, got {step_size}")
    if train_size + test_size > n:
        raise ValueError(
            f"train_size + test_size = {train_size + test_size} > total rows {n}"
        )

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    start = train_size
    while start + test_size <= n:
        train = df_sorted.iloc[:start]
        test = df_sorted.iloc[start : start + test_size]
        yield train, test
        start += step_size
