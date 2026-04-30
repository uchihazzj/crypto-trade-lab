"""Crypto Trend Lab - Research Dashboard

A local Streamlit app for crypto market data exploration.
All model outputs shown here are experimental research results, not investment advice.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
import yaml

from src.crypto_trend_lab.data_ingestion.fetcher import fetch_ohlcv
from src.crypto_trend_lab.features.pipeline import (
    build_features,
    get_model_input_columns,
    get_ohlcv_columns,
    get_target_columns,
    get_technical_feature_columns,
)
from src.crypto_trend_lab.evaluation.report import compare_baselines_and_models
from src.crypto_trend_lab.models.dataset import (
    get_default_feature_columns,
    get_target_column,
)
from src.crypto_trend_lab.models.tabular import _HAS_LIGHTGBM
from src.crypto_trend_lab.storage.parquet import (
    load_features_parquet,
    load_ohlcv_parquet,
    save_features_parquet,
    save_ohlcv_parquet,
    save_predictions_parquet,
)
from src.crypto_trend_lab.validation.quality import check_ohlcv_quality
from src.crypto_trend_lab.visualization.charts import candlestick_chart

st.set_page_config(
    page_title="Crypto Trend Lab",
    page_icon="\U0001f52c",
    layout="wide",
)

st.title("\U0001f52c Crypto Trend Lab")
st.caption(
    "Research-oriented crypto market data exploration. "
    "All results shown here are experimental and do not constitute financial advice."
)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

CONFIG_PATH = "config/symbols.yaml"


@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

EXCHANGES = list(config["exchanges"].keys())
TIMEFRAMES = config["timeframes"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Data Controls")

exchange = st.sidebar.selectbox("Exchange", EXCHANGES, index=0)

symbols = config["exchanges"][exchange]["symbols"]
symbol = st.sidebar.selectbox("Symbol", symbols, index=0)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    TIMEFRAMES,
    index=TIMEFRAMES.index(config["defaults"]["timeframe"]),
)

limit = st.sidebar.slider("Bars to fetch", min_value=50, max_value=1000, value=500, step=50)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_fetch, tab_local, tab_quality, tab_features, tab_models = st.tabs(
    ["Fetch & Chart", "Local Storage", "Data Quality", "Feature Preview", "Model Evaluation"]
)

# ---------------------------------------------------------------------------
# Tab 1: Fetch & Chart
# ---------------------------------------------------------------------------

with tab_fetch:
    st.subheader("Fetch OHLCV Data")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Fetch OHLCV", type="primary", width="stretch"):
            with st.spinner(f"Fetching {symbol} {timeframe} from {exchange}..."):
                try:
                    df = fetch_ohlcv(exchange, symbol, timeframe, limit=limit)
                    st.session_state["fetch_df"] = df
                    st.success(f"Fetched {len(df)} bars")
                except Exception as exc:
                    st.error(f"Fetch failed: {exc}")

        st.metric("Rows fetched", len(st.session_state.get("fetch_df", pd.DataFrame())))

        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            st.divider()
            st.subheader("Save to Local")

            if st.button("Save Raw", width="stretch"):
                df = st.session_state["fetch_df"]
                path = save_ohlcv_parquet(df, exchange, symbol, timeframe, layer="raw")
                st.success(f"Saved to {path}")

            if st.button("Save Processed", width="stretch"):
                df = st.session_state["fetch_df"]
                path = save_ohlcv_parquet(
                    df, exchange, symbol, timeframe, layer="processed"
                )
                st.success(f"Saved to {path}")

    with col2:
        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            df = st.session_state["fetch_df"]
            fig = candlestick_chart(
                df, title=f"{exchange} {symbol} {timeframe}"
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("Raw Data Preview"):
                st.dataframe(df.tail(20), width="stretch")
        else:
            st.info("Click 'Fetch OHLCV' to load market data.")

# ---------------------------------------------------------------------------
# Tab 2: Local Storage
# ---------------------------------------------------------------------------

with tab_local:
    st.subheader("Load Local Parquet Data")

    layer = st.selectbox("Data Layer", ["raw", "processed"], index=0)

    if st.button("Load from Local", type="primary"):
        with st.spinner(f"Loading {symbol} {timeframe} from {layer}..."):
            try:
                df = load_ohlcv_parquet(exchange, symbol, timeframe, layer=layer)
                if df.empty:
                    st.warning("No local data found for these parameters.")
                    st.session_state["local_df"] = pd.DataFrame()
                else:
                    st.session_state["local_df"] = df
                    st.success(f"Loaded {len(df)} bars")
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    if "local_df" in st.session_state and not st.session_state["local_df"].empty:
        df = st.session_state["local_df"]
        fig = candlestick_chart(
            df,
            title=f"{exchange} {symbol} {timeframe} ({layer})",
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Data Preview"):
            st.dataframe(df, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: Data Quality
# ---------------------------------------------------------------------------

with tab_quality:
    st.subheader("Data Quality Checks")

    source = st.radio("Source", ["Fetched", "Local"], horizontal=True)

    df_source = None
    source_label = ""

    if source == "Fetched":
        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            df_source = st.session_state["fetch_df"]
            source_label = "fetched"
    else:
        if "local_df" in st.session_state and not st.session_state["local_df"].empty:
            df_source = st.session_state["local_df"]
            source_label = "local"

    if df_source is not None and not df_source.empty:
        quality = check_ohlcv_quality(df_source, timeframe)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Row Count", quality["row_count"])
        col2.metric("Missing Bars", quality["missing_bars"]["missing_bar_count"])
        col3.metric("Duplicate Timestamps", quality["duplicates"]["duplicate_timestamps"])
        col4.metric("Schema Valid", str(quality["schema"]["schema_valid"]))

        st.divider()
        st.caption(f"Min Timestamp: {quality['min_timestamp']}")
        st.caption(f"Max Timestamp: {quality['max_timestamp']}")

        st.subheader("Null Values")
        if quality["nulls"]["null_counts"]:
            st.json(quality["nulls"]["null_counts"])
        else:
            st.info("No null values.")

        st.subheader("Full Quality Report")
        st.json(json.loads(json.dumps(quality, default=str)))
    else:
        st.info(f"No {source_label} data available. Fetch or load data first.")

# ---------------------------------------------------------------------------
# Tab 4: Feature Preview
# ---------------------------------------------------------------------------

with tab_features:
    st.subheader("Feature Engineering Preview")

    feature_source = st.radio(
        "Data Source",
        ["Fetched", "Local", "Saved Features"],
        horizontal=True,
        key="feature_source",
    )

    df_input: pd.DataFrame | None = None

    if feature_source == "Fetched":
        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            df_input = st.session_state["fetch_df"]
        else:
            st.warning("No fetched data available. Fetch data first in 'Fetch & Chart' tab.")
    elif feature_source == "Local":
        if "local_df" in st.session_state and not st.session_state["local_df"].empty:
            df_input = st.session_state["local_df"]
        else:
            st.warning("No local data loaded. Load data first in 'Local Storage' tab.")
    else:
        try:
            df_input = load_features_parquet(exchange, symbol, timeframe)
            if df_input.empty:
                st.warning("No saved features found. Build features from another source first.")
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    if df_input is not None and not df_input.empty:
        col_a, col_b = st.columns([1, 3])

        with col_a:
            if st.button("Run Feature Pipeline", type="primary", width="stretch"):
                with st.spinner("Building features..."):
                    try:
                        df_feat = build_features(df_input)
                        st.session_state["features_df"] = df_feat
                        st.success(f"Built {len(df_feat)} rows with "
                                   f"{len(get_technical_feature_columns())} features "
                                   f"and {len(get_target_columns())} targets")
                    except Exception as exc:
                        st.error(f"Pipeline failed: {exc}")

            if "features_df" in st.session_state and not st.session_state["features_df"].empty:
                df_feat = st.session_state["features_df"]

                st.metric("Feature Rows", len(df_feat))

                st.divider()
                st.subheader("Save Features")

                if st.button("Save Features to Parquet", width="stretch"):
                    path = save_features_parquet(
                        df_feat, exchange, symbol, timeframe
                    )
                    st.success(f"Saved to {path}")

                st.divider()
                st.subheader("Column Summary")

                st.caption(f"OHLCV: {get_ohlcv_columns()}")
                st.caption(f"Technical Features: {get_technical_feature_columns()}")
                st.caption(f"Targets: {get_target_columns()}")

                st.divider()
                st.subheader("NaN Counts")

                nan_counts = df_feat.isnull().sum()
                nan_nonzero = nan_counts[nan_counts > 0]
                if not nan_nonzero.empty:
                    st.dataframe(
                        nan_nonzero.rename("NaN count"),
                        width="stretch",
                    )
                else:
                    st.info("No NaN values.")

        with col_b:
            if "features_df" in st.session_state and not st.session_state["features_df"].empty:
                df_feat = st.session_state["features_df"]

                st.subheader("Indicator Charts")

                chart_indicator = st.selectbox(
                    "Indicator",
                    ["close", "ma_25", "rsi_14", "macd", "bollinger_width"],
                    index=0,
                )

                if chart_indicator in df_feat.columns:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df_feat["timestamp"],
                            y=df_feat[chart_indicator],
                            mode="lines",
                            name=chart_indicator,
                        )
                    )
                    fig.update_layout(
                        title=f"{chart_indicator}",
                        xaxis_title="Time (UTC)",
                        yaxis_title=chart_indicator,
                        template="plotly_dark",
                        height=400,
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info(f"Column {chart_indicator!r} not found in feature data.")

                st.divider()

                with st.expander("Feature Data Preview"):
                    st.dataframe(df_feat.tail(50), width="stretch")
    else:
        st.info("Load or fetch OHLCV data first, then run the feature pipeline.")

# ---------------------------------------------------------------------------
# Tab 5: Model Evaluation
# ---------------------------------------------------------------------------

with tab_models:
    st.subheader("Model Evaluation")
    st.caption(
        "Chronological train/test evaluation of baseline and tabular models. "
        "All results are experimental research signals, not investment advice."
    )

    # --- Data source ---
    eval_source = st.radio(
        "Data Source",
        ["Features DataFrame", "Saved Features"],
        horizontal=True,
        key="eval_source",
    )

    df_eval: pd.DataFrame | None = None

    if eval_source == "Features DataFrame":
        if "features_df" in st.session_state and not st.session_state["features_df"].empty:
            df_eval = st.session_state["features_df"]
        else:
            st.warning("No features data. Run the feature pipeline in 'Feature Preview' tab first.")
    else:
        try:
            df_eval = load_features_parquet(exchange, symbol, timeframe)
            if df_eval.empty:
                st.warning("No saved features found for this symbol/timeframe.")
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    if df_eval is not None and not df_eval.empty:
        # --- Controls ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            task_type = st.selectbox(
                "Task Type",
                ["regression", "classification"],
                index=0,
                help="Regression predicts log returns; classification predicts direction.",
            )

        with col2:
            horizon = st.selectbox(
                "Horizon",
                [1, 4, 24],
                index=0,
                help="Forecast horizon in bars (1h each).",
            )

        with col3:
            test_pct = st.slider(
                "Test Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of most recent data held out for testing.",
            ) / 100.0

        with col4:
            include_tabular = st.checkbox(
                "Include Tabular Models",
                value=True,
                help="Run Ridge/Logistic Regression and LightGBM if available.",
            )

        target_col = get_target_column(task_type, horizon)
        feature_cols = get_default_feature_columns(df_eval)

        # --- Info ---
        st.divider()
        st.caption(f"Target: **{target_col}**")
        st.caption(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

        if not _HAS_LIGHTGBM:
            st.info("LightGBM is not installed. LightGBM models will be skipped.")

        # --- Run evaluation ---
        if st.button("Run Evaluation", type="primary", width="stretch"):
            with st.spinner(f"Evaluating {task_type} models for horizon {horizon}..."):
                try:
                    result = compare_baselines_and_models(
                        df_eval,
                        task_type=task_type,
                        horizon=horizon,
                        test_size=test_pct,
                        include_tabular=include_tabular,
                    )
                    st.session_state["eval_result"] = result
                    st.success(
                        f"Evaluated {len(result['metrics_table'])} models "
                        f"({len(result['predictions'])} test predictions)"
                    )
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        # --- Results ---
        if "eval_result" in st.session_state:
            result = st.session_state["eval_result"]

            st.divider()
            st.subheader("Train / Test Dates")

            c1, c2 = st.columns(2)
            c1.caption(
                f"Train: {result['train_dates']['start']} → {result['train_dates']['end']}"
            )
            c2.caption(
                f"Test: {result['test_dates']['start']} → {result['test_dates']['end']}"
            )

            st.divider()
            st.subheader("Metrics Comparison")

            metrics_df = result["metrics_table"]
            st.dataframe(
                metrics_df.set_index("model_name"),
                width="stretch",
            )

            st.divider()
            st.subheader("Predictions: y_true vs y_pred")

            predictions = result["predictions"]
            model_names = predictions["model_name"].unique()

            selected_model = st.selectbox(
                "Model",
                model_names,
                index=0,
                key="eval_model_select",
            )

            pred_subset = predictions[predictions["model_name"] == selected_model]

            import plotly.graph_objects as go

            fig = go.Figure()

            if task_type == "regression":
                fig.add_trace(
                    go.Scatter(
                        x=pred_subset["timestamp"],
                        y=pred_subset["y_true"],
                        mode="lines+markers",
                        name="Actual (y_true)",
                        marker=dict(size=4),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred_subset["timestamp"],
                        y=pred_subset["y_pred"],
                        mode="lines+markers",
                        name="Predicted (y_pred)",
                        marker=dict(size=4),
                    )
                )
            else:
                # Classification: show probability if available, otherwise class overlay
                fig.add_trace(
                    go.Scatter(
                        x=pred_subset["timestamp"],
                        y=pred_subset["y_true"],
                        mode="lines+markers",
                        name="Actual Direction",
                        marker=dict(size=4),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred_subset["timestamp"],
                        y=pred_subset["y_pred"],
                        mode="markers",
                        name="Predicted Direction",
                        marker=dict(size=6, symbol="x"),
                    )
                )
                if "y_prob" in pred_subset.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pred_subset["timestamp"],
                            y=pred_subset["y_prob"],
                            mode="lines",
                            name="Predicted Probability",
                            line=dict(dash="dot"),
                        )
                    )

            fig.update_layout(
                title=f"{selected_model} — {target_col}",
                xaxis_title="Time (UTC)",
                yaxis_title=target_col,
                template="plotly_dark",
                height=450,
            )
            st.plotly_chart(fig, width="stretch")

            st.divider()
            st.subheader("Save Predictions")

            if st.button("Save Predictions to Parquet", width="stretch"):
                pred_to_save = predictions[predictions["model_name"] == selected_model]
                path = save_predictions_parquet(
                    pred_to_save,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_name=selected_model,
                    target_column=target_col,
                )
                st.success(f"Saved {len(pred_to_save)} predictions to {path}")

            with st.expander("Prediction Data Preview"):
                st.dataframe(predictions.tail(50), width="stretch")
