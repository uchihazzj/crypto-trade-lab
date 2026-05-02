"""Tabular models: linear, tree ensembles, and optional external libraries.

All models use scikit-learn pipelines where applicable. Scalers are fit
only on the training set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------


def _check_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        logger.warning("LightGBM is not installed. LightGBM models are disabled.")
        return False


def _check_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        logger.warning("XGBoost is not installed. XGBoost models are disabled.")
        return False


def _check_catboost() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        logger.warning("CatBoost is not installed. CatBoost models are disabled.")
        return False


_HAS_LIGHTGBM = _check_lightgbm()
_HAS_XGBOOST = _check_xgboost()
_HAS_CATBOOST = _check_catboost()


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------


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


class ElasticNetRegressionModel:
    """ElasticNet regression with feature scaling."""

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                          max_iter=2000, random_state=42)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegressionModel":
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


# ---------------------------------------------------------------------------
# Tree ensemble models — regression
# ---------------------------------------------------------------------------


class RandomForestRegressionModel:
    """Random Forest regressor with conservative defaults."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        self._model = RandomForestRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class ExtraTreesRegressionModel:
    """Extra Trees regressor with conservative defaults."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        self._model = ExtraTreesRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class HistGradientBoostingRegressionModel:
    """Histogram-based Gradient Boosting regressor (sklearn)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_iter", 200)
        kwargs.setdefault("learning_rate", 0.05)
        kwargs.setdefault("random_state", 42)
        self._model = HistGradientBoostingRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Tree ensemble models — classification
# ---------------------------------------------------------------------------


class RandomForestClassificationModel:
    """Random Forest classifier with conservative defaults."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("class_weight", "balanced")
        self._model = RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassificationModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class ExtraTreesClassificationModel:
    """Extra Trees classifier with conservative defaults."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("class_weight", "balanced")
        self._model = ExtraTreesClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesClassificationModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class HistGradientBoostingClassificationModel:
    """Histogram-based Gradient Boosting classifier (sklearn)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_iter", 200)
        kwargs.setdefault("learning_rate", 0.05)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("class_weight", "balanced")
        self._model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingClassificationModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# LightGBM (optional)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# XGBoost (optional)
# ---------------------------------------------------------------------------


class XGBoostRegressor:
    """XGBoost regressor. Raises ImportError if xgboost is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        import xgboost as xgb

        kwargs.setdefault("verbosity", 0)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_estimators", 200)
        self._model = xgb.XGBRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class XGBoostClassifier:
    """XGBoost classifier. Raises ImportError if xgboost is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        import xgboost as xgb

        kwargs.setdefault("verbosity", 0)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_estimators", 200)
        self._model = xgb.XGBClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# CatBoost (optional)
# ---------------------------------------------------------------------------


class CatBoostRegressor:
    """CatBoost regressor. Raises ImportError if catboost is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_CATBOOST:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )
        import catboost as cb

        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("random_seed", 42)
        kwargs.setdefault("iterations", 200)
        self._model = cb.CatBoostRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class CatBoostClassifier:
    """CatBoost classifier. Raises ImportError if catboost is not installed."""

    def __init__(self, **kwargs) -> None:
        if not _HAS_CATBOOST:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )
        import catboost as cb

        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("random_seed", 42)
        kwargs.setdefault("iterations", 200)
        self._model = cb.CatBoostClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]
