"""Simple baseline models.

Baselines use minimal information and must be beaten by any useful model.
"""

from __future__ import annotations

import numpy as np


class ZeroReturnBaseline:
    """Predict future return as zero."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class LastReturnBaseline:
    """Predict future return as the last observed training return."""

    def __init__(self) -> None:
        self.last_value_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.last_value_ = float(y[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.last_value_, dtype=float)


class MovingAverageReturnBaseline:
    """Predict future return as the mean of training returns."""

    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


class MomentumDirectionBaseline:
    """Predict direction as the sign of the last training return."""

    def __init__(self) -> None:
        self.last_direction_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.last_direction_ = 1.0 if y[-1] > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.last_direction_, dtype=float)


class MajorityClassBaseline:
    """Predict the most frequent class in training labels."""

    def __init__(self) -> None:
        self.majority_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        vals, counts = np.unique(y, return_counts=True)
        self.majority_ = float(vals[np.argmax(counts)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_, dtype=float)
