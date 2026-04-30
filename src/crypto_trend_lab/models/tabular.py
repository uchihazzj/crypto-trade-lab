"""Tabular models: Ridge, Logistic Regression, and LightGBM.

All models use scikit-learn pipelines where applicable. Scalers are fit
only on the training set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _check_lightgbm() -> bool:
    """Return True if LightGBM is installed."""
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        logger.warning("LightGBM is not installed. LightGBM models are disabled.")
        return False


_HAS_LIGHTGBM = _check_lightgbm()


class RidgeRegressionModel:
    """Ridge regression with feature scaling."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)


class LogisticRegressionModel:
    """Logistic regression with feature scaling."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_iter", 1000)
        kwargs.setdefault("class_weight", "balanced")
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(**kwargs)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)[:, 1]


class LightGBMRegressor:
    """LightGBM regressor. Raises ImportError if lightgbm is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        import lightgbm as lgb

        kwargs.setdefault("verbosity", -1)
        kwargs.setdefault("random_state", 42)
        self._model = lgb.LGBMRegressor(**kwargs)
        self._feature_names_: list[str] = []

    def _as_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self._feature_names_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMRegressor":
        self._feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        self._model.fit(self._as_dataframe(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._as_dataframe(X))


class LightGBMClassifier:
    """LightGBM classifier. Raises ImportError if lightgbm is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        import lightgbm as lgb

        kwargs.setdefault("verbosity", -1)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("class_weight", "balanced")
        self._model = lgb.LGBMClassifier(**kwargs)
        self._feature_names_: list[str] = []

    def _as_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self._feature_names_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        self._feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        self._model.fit(self._as_dataframe(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._as_dataframe(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._as_dataframe(X))[:, 1]
