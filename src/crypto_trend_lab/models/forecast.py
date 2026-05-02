"""Dense direct-horizon forecast path.

Trains one independent regression model per future step (1..path_length).
Not recursive forecasting, not a sequence model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.evaluation.forecast import generate_future_timestamps
from src.crypto_trend_lab.features.target import add_dense_return_targets
from src.crypto_trend_lab.models.baseline import (
    HistoricalMeanReturnBaseline,
    LastReturnBaseline,
    MovingAverageReturnBaseline,
    ZeroReturnBaseline,
)
from src.crypto_trend_lab.models.dataset import get_default_feature_columns
from src.crypto_trend_lab.models.tabular import (
    _HAS_CATBOOST,
    _HAS_LIGHTGBM,
    _HAS_XGBOOST,
    CatBoostRegressor,
    ElasticNetRegressionModel,
    ExtraTreesRegressionModel,
    HistGradientBoostingRegressionModel,
    LightGBMRegressor,
    RandomForestRegressionModel,
    RidgeRegressionModel,
    XGBoostRegressor,
)

# Regression model name → constructor mapping.
# Independent of evaluation/forecast.py's registry.
_REGRESSION_MODELS: dict[str, type] = {
    "Zero Return": ZeroReturnBaseline,
    "Last Return": LastReturnBaseline,
    "Moving Average": MovingAverageReturnBaseline,
    "Historical Mean Return": HistoricalMeanReturnBaseline,
    "Ridge": RidgeRegressionModel,
    "ElasticNet": ElasticNetRegressionModel,
    "Random Forest": RandomForestRegressionModel,
    "Extra Trees": ExtraTreesRegressionModel,
    "HistGradientBoosting": HistGradientBoostingRegressionModel,
}
if _HAS_LIGHTGBM:
    _REGRESSION_MODELS["LightGBM"] = LightGBMRegressor
if _HAS_XGBOOST:
    _REGRESSION_MODELS["XGBoost"] = XGBoostRegressor
if _HAS_CATBOOST:
    _REGRESSION_MODELS["CatBoost"] = CatBoostRegressor


def run_dense_direct_forecast(
    df: pd.DataFrame,
    model_name: str = "Ridge",
    path_length: int = 24,
    feature_columns: list[str] | None = None,
    timeframe: str = "1h",
) -> dict:
    """Run a dense direct-horizon forecast path.

    Trains one independent regression model per step h = 1..*path_length*.
    Each model predicts ``target_return_h`` from the latest feature row.
    Estimated future close at step h = ``latest_close × exp(pred_log_return)``.

    This is **not** recursive forecasting.  Each horizon model is fitted
    independently on the same set of features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.  Must have ``close`` and
        ``timestamp`` columns plus feature columns.
    model_name : str
        Regression model name (e.g. "Ridge", "LightGBM").
    path_length : int
        Number of future steps to forecast (1..path_length).
    feature_columns : list[str] or None
        Feature columns.  If None, auto-detected.
    timeframe : str
        CCXT timeframe string used for future timestamp generation.

    Returns
    -------
    dict
        Keys: path_points (list of dicts), latest_timestamp, latest_close,
        model_name, training_rows, path_length, timeframe, chart_history,
        failed_horizons (list), error (if overall failure).
    """
    if model_name not in _REGRESSION_MODELS:
        return {
            "error": (
                f"Dense forecast path requires a regression model. "
                f"{model_name!r} is not a regression model. "
                f"Available: {sorted(_REGRESSION_MODELS.keys())}"
            )
        }

    if feature_columns is None:
        feature_columns = get_default_feature_columns(df)

    # Drop entirely-NaN feature columns
    feature_columns = [
        c for c in feature_columns
        if c in df.columns and not df[c].isna().all()
    ]
    if not feature_columns:
        return {"error": "No usable feature columns after dropping all-NaN columns."}

    # Latest row with valid features (target may be NaN)
    forecast_candidates = df.dropna(subset=feature_columns)
    if forecast_candidates.empty:
        return {"error": "No rows with complete feature values."}

    forecast_row = forecast_candidates.iloc[-1:]
    latest_ts = pd.Timestamp(df.loc[forecast_row.index[0], "timestamp"])
    latest_close = float(df["close"].iloc[-1])

    if not (np.isfinite(latest_close) and latest_close > 0):
        return {"error": "No valid close price available for forecast path."}

    # Generate future timestamps for all steps
    future_ts = generate_future_timestamps(latest_ts, timeframe, path_length)
    if not future_ts:
        return {"error": f"Unknown timeframe {timeframe!r}."}

    # Ensure dense targets exist in a working copy.
    # Only add columns that are missing — some targets (1, 4, 24) may already
    # exist from build_features().
    target_cols_needed = [f"target_return_{h}" for h in range(1, path_length + 1)]
    missing = [c for c in target_cols_needed if c not in df.columns]
    if missing:
        dense_targets = add_dense_return_targets(df, max_horizon=path_length)
        # Drop columns that already exist in df to avoid duplicates
        new_cols = dense_targets[
            [c for c in dense_targets.columns if c not in df.columns]
        ]
        df_work = pd.concat([df, new_cols], axis=1)
    else:
        df_work = df

    model_cls = _REGRESSION_MODELS[model_name]
    path_points: list[dict] = []
    failed_horizons: list[dict] = []
    total_labeled = 0

    for h in range(1, path_length + 1):
        target_col = f"target_return_{h}"

        if target_col not in df_work.columns:
            failed_horizons.append({
                "horizon": h,
                "target_column": target_col,
                "reason": f"Target column {target_col!r} not found.",
            })
            continue

        # Training data: rows where this target AND all features are valid
        labeled = df_work.dropna(subset=[target_col]).dropna(subset=feature_columns)
        if len(labeled) < 10:
            failed_horizons.append({
                "horizon": h,
                "target_column": target_col,
                "reason": (
                    f"Only {len(labeled)} valid rows for {target_col} "
                    f"(need at least 10)."
                ),
            })
            continue

        total_labeled = max(total_labeled, len(labeled))

        X_train = labeled[feature_columns].to_numpy(dtype=float)
        y_train = labeled[target_col].to_numpy(dtype=float)
        X_pred = forecast_row[feature_columns].to_numpy(dtype=float)

        try:
            model = model_cls()
            model.fit(X_train, y_train)
            pred_log_return = float(model.predict(X_pred)[0])
        except Exception as exc:
            failed_horizons.append({
                "horizon": h,
                "target_column": target_col,
                "reason": f"Model training/prediction failed: {exc}",
            })
            continue

        est_future_close = latest_close * np.exp(pred_log_return)
        forecast_ts_val = future_ts[h - 1]

        path_points.append({
            "horizon": h,
            "forecast_step": h,
            "forecast_timestamp": forecast_ts_val,
            "target_column": target_col,
            "predicted_log_return": pred_log_return,
            "estimated_future_close": est_future_close,
        })

    # Chart history for left-side OHLCV context
    chart_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    available_chart_cols = [c for c in chart_cols if c in df.columns]
    chart_history = df[available_chart_cols].copy()
    if "timestamp" in chart_history.columns:
        chart_history["timestamp"] = pd.to_datetime(
            chart_history["timestamp"], utc=True
        )
        chart_history = chart_history.sort_values("timestamp")

    return {
        "latest_timestamp": latest_ts,
        "latest_close": latest_close,
        "path_points": path_points,
        "model_name": model_name,
        "training_rows": total_labeled,
        "path_length": path_length,
        "timeframe": timeframe,
        "chart_history": chart_history,
        "failed_horizons": failed_horizons,
    }
