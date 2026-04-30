"""Models: baselines and tabular models for trend prediction."""

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
    LightGBMClassifier,
    LightGBMRegressor,
    LogisticRegressionModel,
    RidgeRegressionModel,
)
