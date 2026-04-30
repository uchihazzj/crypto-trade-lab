"""Forward forecast: train on all labeled history, predict latest feature row.

This is separate from historical evaluation — it produces an experimental
forecast for the most recent observation where the future target is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from src.crypto_trend_lab.evaluation.split import chronological_train_test_split
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
)
from src.crypto_trend_lab.models.tabular import (
    _HAS_LIGHTGBM,
    LightGBMClassifier,
    LightGBMRegressor,
    LogisticRegressionModel,
    RidgeRegressionModel,
)

# Maps model display names to constructors
_REGRESSION_MODELS: dict[str, type] = {
    "Zero Return": ZeroReturnBaseline,
    "Last Return": LastReturnBaseline,
    "Moving Average": MovingAverageReturnBaseline,
    "Ridge": RidgeRegressionModel,
}
_CLASSIFICATION_MODELS: dict[str, type] = {
    "Momentum Direction": MomentumDirectionBaseline,
    "Majority Class": MajorityClassBaseline,
    "Logistic Regression": LogisticRegressionModel,
}

if _HAS_LIGHTGBM:
    _REGRESSION_MODELS["LightGBM"] = LightGBMRegressor
    _CLASSIFICATION_MODELS["LightGBM"] = LightGBMClassifier


def _extract_latest_close(df: pd.DataFrame) -> float | None:
    """Return the latest close price if available."""
    if "close" in df.columns:
        return float(df["close"].iloc[-1])
    return None


def forward_forecast(
    df: pd.DataFrame,
    task_type: str = "regression",
    horizon: int = 1,
    model_name: str = "Ridge",
    feature_columns: list[str] | None = None,
    historical_test_size: int | float = 0.2,
) -> dict:
    """Train a model on all labeled historical rows and forecast the latest row.

    Uses every row with valid features AND a valid target for training.
    Forecasts from the latest row that has valid features (target may be NaN).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.
    task_type : str
        "regression" or "classification".
    horizon : int
        Forecast horizon: 1, 4, or 24.
    model_name : str
        Display name of the model to use (e.g. "Ridge", "LightGBM").
    feature_columns : list[str] or None
        Feature columns. If None, auto-detected.
    historical_test_size : int or float
        Test size for computing historical context metrics. Evaluated on a
        chronological hold-out split purely to provide context — the final
        forecast model is trained on ALL labeled data.

    Returns
    -------
    dict
        Keys depend on task_type. Always includes: latest_timestamp, horizon,
        training_rows, and potential error.
    """
    target_column = get_target_column(task_type, horizon)

    if target_column not in df.columns:
        return {"error": f"Target column {target_column!r} not in DataFrame."}

    if feature_columns is None:
        feature_columns = get_default_feature_columns(df)

    if not feature_columns:
        return {"error": "No feature columns available."}

    # Drop feature columns that are entirely NaN (e.g. rolling_vol_168
    # when fewer than 168 rows are available)
    feature_columns = [
        c for c in feature_columns
        if c in df.columns and not df[c].isna().all()
    ]
    if not feature_columns:
        return {
            "error": (
                "All feature columns are entirely NaN. "
                "Fetch more data or check feature pipeline."
            )
        }

    # Select the model class
    if task_type == "regression":
        model_cls = _REGRESSION_MODELS.get(model_name)
    else:
        model_cls = _CLASSIFICATION_MODELS.get(model_name)

    if model_cls is None:
        available = sorted(
            (_REGRESSION_MODELS if task_type == "regression"
             else _CLASSIFICATION_MODELS).keys()
        )
        return {
            "error": (
                f"Unknown model {model_name!r} for {task_type}. "
                f"Available: {available}"
            )
        }

    # Validate required columns
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return {"error": f"Feature columns not in DataFrame: {missing}"}

    # Separate labeled (target known) vs unlabeled (target NaN) rows
    labeled = df.dropna(subset=[target_column]).copy()
    if len(labeled) < 10:
        return {
            "error": (
                f"Only {len(labeled)} rows with valid {target_column}. "
                f"Need at least 10 for training."
            )
        }

    # Features must also not be NaN
    labeled = labeled.dropna(subset=feature_columns)
    if len(labeled) < 10:
        return {
            "error": (
                f"Only {len(labeled)} rows after dropping NaN features. "
                f"Need at least 10."
            )
        }

    # Latest row with valid features (may have NaN target)
    forecast_candidates = df.dropna(subset=feature_columns)
    if forecast_candidates.empty:
        return {"error": "No rows with complete feature values."}

    forecast_row = forecast_candidates.iloc[-1:]
    forecast_ts = pd.Timestamp(df.loc[forecast_row.index[0], "timestamp"])

    # --- Compute historical context metrics ---
    # Evaluate on a chronological split to provide context (not used for
    # training the final forecast model).
    context_metrics: dict = {}
    try:
        if len(labeled) >= 50:
            hist_split = chronological_train_test_split(
                labeled, test_size=historical_test_size, time_col="timestamp"
            )
            if len(hist_split[0]) >= 10 and len(hist_split[1]) >= 2:
                train_ctx, test_ctx = hist_split
                X_train_ctx = train_ctx[feature_columns].to_numpy(dtype=float)
                y_train_ctx = train_ctx[target_column].to_numpy(dtype=float)
                X_test_ctx = test_ctx[feature_columns].to_numpy(dtype=float)
                y_test_ctx = test_ctx[target_column].to_numpy(dtype=float)

                ctx_model = model_cls()
                ctx_model.fit(X_train_ctx, y_train_ctx)
                y_pred_ctx = ctx_model.predict(X_test_ctx)

                if task_type == "regression":
                    ctx_m = regression_metrics(y_test_ctx, y_pred_ctx)
                    if "mae" in ctx_m:
                        context_metrics["historical_mae"] = ctx_m["mae"]
                    if "rmse" in ctx_m:
                        context_metrics["historical_rmse"] = ctx_m["rmse"]
                    if "directional_accuracy" in ctx_m:
                        context_metrics["historical_directional_accuracy"] = (
                            ctx_m["directional_accuracy"]
                        )
                else:
                    y_prob_ctx = None
                    if hasattr(ctx_model, "predict_proba"):
                        y_prob_ctx = ctx_model.predict_proba(X_test_ctx)
                    ctx_m = classification_metrics(
                        y_test_ctx, y_pred_ctx, y_prob_ctx
                    )
                    if "balanced_accuracy" in ctx_m:
                        context_metrics["historical_balanced_accuracy"] = (
                            ctx_m["balanced_accuracy"]
                        )
    except Exception:
        # Historical context is best-effort; failure should not block forecast
        pass

    # --- Train on ALL labeled data ---
    X_train = labeled[feature_columns].to_numpy(dtype=float)
    y_train = labeled[target_column].to_numpy(dtype=float)
    X_forecast = forecast_row[feature_columns].to_numpy(dtype=float)

    model = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_forecast)

    # --- Build forecast result ---
    result: dict = {
        "latest_timestamp": forecast_ts,
        "horizon": horizon,
        "training_rows": len(labeled),
        "target_column": target_column,
        "model_name": model_name,
    }
    result.update(context_metrics)

    if task_type == "regression":
        predicted_log_return = float(y_pred[0])
        latest_close = _extract_latest_close(labeled)

        result["predicted_log_return"] = predicted_log_return
        result["latest_close"] = latest_close

        if latest_close is not None and latest_close > 0:
            result["estimated_future_close"] = (
                latest_close * np.exp(predicted_log_return)
            )
        else:
            result["estimated_future_close"] = None
    else:
        predicted_class = int(y_pred[0])
        result["predicted_class"] = predicted_class

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_forecast)
            result["predicted_probability"] = float(proba[0])
        else:
            result["predicted_probability"] = None

    return result


# Supported horizons for sparse direct-horizon forecast path.
_SPARSE_HORIZONS = (1, 4, 24)


def generate_future_timestamps(
    latest_ts: pd.Timestamp,
    timeframe: str,
    path_length: int,
) -> list[pd.Timestamp]:
    """Generate future timestamps continuing after *latest_ts*.

    All future timestamps are generated using ``pd.Timedelta`` arithmetic —
    never integer addition. This avoids ``TypeError`` from pandas when
    Timestamp + int is attempted.

    Parameters
    ----------
    latest_ts : pd.Timestamp
        The last observed timestamp. Must be a scalar Timestamp.
    timeframe : str
        CCXT timeframe string (e.g. '1h', '4h', '1d').
    path_length : int
        Number of future bars to generate timestamps for.

    Returns
    -------
    list[pd.Timestamp]
        Empty list if the timeframe is unsupported.
    """
    from src.crypto_trend_lab.utils.helpers import timeframe_to_timedelta

    try:
        delta = timeframe_to_timedelta(timeframe)
    except KeyError:
        return []

    # Ensure latest_ts is a scalar Timestamp (not a DatetimeIndex element)
    ts = pd.Timestamp(latest_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("utc")

    return [ts + delta * step for step in range(1, path_length + 1)]


def forecast_path(
    df: pd.DataFrame,
    model_name: str = "Ridge",
    path_length: int = 24,
    feature_columns: list[str] | None = None,
    timeframe: str = "1h",
) -> dict:
    """Produce a sparse direct-horizon forecast path.

    Trains one model per supported horizon (1, 4, 24) on all labeled
    historical rows, then predicts each horizon from the latest feature
    row. Returns timestamped forecast points and recent OHLCV context
    for charting.

    Only supports regression — classification does not produce a
    close-price forecast path.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.
    model_name : str
        Regression model name: "Ridge" or "LightGBM".
    path_length : int
        Desired forecast length in bars. Only supported horizons ≤
        path_length are included.
    feature_columns : list[str] or None
        Feature columns. If None, auto-detected.
    timeframe : str
        CCXT timeframe string used to generate future timestamps.

    Returns
    -------
    dict
        Keys: latest_timestamp, latest_close, path_points (list of dicts),
        model_name, training_rows, path_length, chart_history (DataFrame),
        error (if regression not possible).
    """
    if model_name not in _REGRESSION_MODELS:
        return {
            "error": (
                f"Forecast path requires a regression model. "
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
    latest_close = _extract_latest_close(df)

    if latest_close is None or latest_close <= 0:
        return {"error": "No valid close price available for forecast path."}

    # Determine which horizons to forecast
    active_horizons = [h for h in _SPARSE_HORIZONS if h <= path_length]

    if not active_horizons:
        return {"error": f"path_length {path_length} < minimum supported horizon 1."}

    # Generate future timestamps for all steps
    future_ts = generate_future_timestamps(latest_ts, timeframe, path_length)
    if not future_ts:
        return {"error": f"Unknown timeframe {timeframe!r}."}

    path_points: list[dict] = []
    total_labeled = 0

    for h in active_horizons:
        target_col = f"target_return_{h}"
        if target_col not in df.columns:
            path_points.append({
                "horizon": h,
                "target_column": target_col,
                "error": f"Target column {target_col!r} not in DataFrame.",
            })
            continue

        # Training data: rows with valid target AND valid features
        labeled = df.dropna(subset=[target_col]).dropna(subset=feature_columns)
        if len(labeled) < 10:
            path_points.append({
                "horizon": h,
                "target_column": target_col,
                "error": f"Only {len(labeled)} valid rows for {target_col}.",
            })
            continue

        total_labeled = max(total_labeled, len(labeled))

        X_train = labeled[feature_columns].to_numpy(dtype=float)
        y_train = labeled[target_col].to_numpy(dtype=float)
        X_pred = forecast_row[feature_columns].to_numpy(dtype=float)

        model_cls = _REGRESSION_MODELS[model_name]
        model = model_cls()
        model.fit(X_train, y_train)
        pred_log_return = float(model.predict(X_pred)[0])

        est_future_close = latest_close * np.exp(pred_log_return)

        # Map to future timestamp: horizon h means h bars ahead
        ts_idx = h - 1  # 0-indexed
        forecast_ts_val = future_ts[ts_idx] if ts_idx < len(future_ts) else future_ts[-1]

        path_points.append({
            "horizon": h,
            "forecast_step": h,
            "forecast_timestamp": forecast_ts_val,
            "target_column": target_col,
            "predicted_log_return": pred_log_return,
            "estimated_future_close": est_future_close,
        })

    # Chart history: recent OHLCV for left-side context
    chart_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    available_chart_cols = [c for c in chart_cols if c in df.columns]
    chart_history = df[available_chart_cols].copy()
    chart_history["timestamp"] = pd.to_datetime(chart_history["timestamp"], utc=True)
    chart_history = chart_history.sort_values("timestamp")

    return {
        "latest_timestamp": latest_ts,
        "latest_close": latest_close,
        "path_points": path_points,
        "model_name": model_name,
        "training_rows": total_labeled,
        "path_length": path_length,
        "max_supported_horizon": max(_SPARSE_HORIZONS),
        "timeframe": timeframe,
        "chart_history": chart_history,
    }
