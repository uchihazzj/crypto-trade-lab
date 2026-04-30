"""Evaluation metrics for regression and classification tasks.

All functions handle NaN by dropping affected rows before computation.
Returns empty dict if no valid samples remain.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

# scipy is optional — spearman correlation only computed if available.
try:
    from scipy.stats import spearmanr as _spearmanr
    from scipy.stats import ConstantInputWarning

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _drop_nan(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None):
    """Remove indices where y_true or y_pred is NaN."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if y_prob is not None:
        y_prob = y_prob[mask]
    return y_t, y_p, y_prob


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Returns dict with keys: mae, rmse, directional_accuracy, spearman_r (if scipy).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_t, y_p, _ = _drop_nan(y_true, y_pred)

    if len(y_t) == 0:
        return {}

    result: dict[str, float] = {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "directional_accuracy": float(
            np.mean((np.sign(y_t) == np.sign(y_p)).astype(float))
        ),
    }

    if _HAS_SCIPY:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConstantInputWarning)
            try:
                rho, _ = _spearmanr(y_t, y_p)
                result["spearman_r"] = float(rho)
            except ConstantInputWarning:
                result["spearman_r"] = float("nan")

    return result


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Returns dict with keys: accuracy, balanced_accuracy, precision, recall, f1,
    and auc (if *y_prob* provided).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_t, y_p, y_prob = _drop_nan(y_true, y_pred, y_prob)

    if len(y_t) == 0:
        return {}

    # sklearn classification metrics expect int labels
    y_t_int = y_t.astype(int)
    y_p_int = y_p.astype(int)

    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y_t_int, y_p_int)),
        "balanced_accuracy": float(balanced_accuracy_score(y_t_int, y_p_int)),
        "precision": float(precision_score(y_t_int, y_p_int, zero_division=0)),
        "recall": float(recall_score(y_t_int, y_p_int, zero_division=0)),
        "f1": float(f1_score(y_t_int, y_p_int, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_t_int)) > 1:
        result["auc"] = float(roc_auc_score(y_t_int, y_prob))

    return result
