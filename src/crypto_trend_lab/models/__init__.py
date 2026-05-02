"""Models: baselines and tabular models for trend prediction."""

from src.crypto_trend_lab.models.baseline import (
    HistoricalMeanReturnBaseline as HistoricalMeanReturnBaseline,
    LastReturnBaseline as LastReturnBaseline,
    MajorityClassBaseline as MajorityClassBaseline,
    MomentumDirectionBaseline as MomentumDirectionBaseline,
    MovingAverageReturnBaseline as MovingAverageReturnBaseline,
    ZeroReturnBaseline as ZeroReturnBaseline,
)
from src.crypto_trend_lab.models.dataset import (
    get_default_feature_columns as get_default_feature_columns,
    get_target_column as get_target_column,
    prepare_modeling_data as prepare_modeling_data,
)
from src.crypto_trend_lab.models.tabular import (
    _HAS_CATBOOST as _HAS_CATBOOST,
    _HAS_LIGHTGBM as _HAS_LIGHTGBM,
    _HAS_XGBOOST as _HAS_XGBOOST,
    CatBoostClassifier as CatBoostClassifier,
    CatBoostRegressor as CatBoostRegressor,
    ElasticNetRegressionModel as ElasticNetRegressionModel,
    ExtraTreesClassificationModel as ExtraTreesClassificationModel,
    ExtraTreesRegressionModel as ExtraTreesRegressionModel,
    HistGradientBoostingClassificationModel as HistGradientBoostingClassificationModel,
    HistGradientBoostingRegressionModel as HistGradientBoostingRegressionModel,
    LightGBMClassifier as LightGBMClassifier,
    LightGBMRegressor as LightGBMRegressor,
    LogisticRegressionModel as LogisticRegressionModel,
    RandomForestClassificationModel as RandomForestClassificationModel,
    RandomForestRegressionModel as RandomForestRegressionModel,
    RidgeRegressionModel as RidgeRegressionModel,
    XGBoostClassifier as XGBoostClassifier,
    XGBoostRegressor as XGBoostRegressor,
)
