"""Crypto Trend Lab - Research Dashboard

A local Streamlit app for crypto market data exploration.
All model outputs shown here are experimental research results, not investment advice.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
import yaml

from src.crypto_trend_lab.data_ingestion.fetcher import fetch_ohlcv, fetch_ohlcv_range
from src.crypto_trend_lab.features.pipeline import (
    build_features,
    get_model_input_columns,
    get_ohlcv_columns,
    get_target_columns,
    get_technical_feature_columns,
)
from src.crypto_trend_lab.evaluation.full_report import (
    build_best_model_summary,
    generate_report_analysis,
    run_full_evaluation_report,
)
from src.crypto_trend_lab.evaluation.forecast import forecast_path, forward_forecast
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
    save_forecast_parquet,
    save_ohlcv_parquet,
    save_predictions_parquet,
)
from src.crypto_trend_lab.utils.helpers import (
    dataset_sizing_warning,
    estimate_coverage,
    timeframe_to_timedelta,
)
from src.crypto_trend_lab.validation.quality import check_ohlcv_quality
from src.crypto_trend_lab.visualization.charts import (
    add_datetime_vertical_marker,
    candlestick_chart,
)
from src.crypto_trend_lab.visualization.display import (
    DEFAULT_MAX_CANDLES,
    DEFAULT_PREVIEW_ROWS,
    aggregate_ohlcv_by_count,
    filter_ohlcv_by_chart_range,
    get_display_summary,
    prepare_candlestick_display_data,
)

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

limit = st.sidebar.number_input(
    "Bars to fetch",
    min_value=100,
    max_value=50000,
    value=1000,
    step=100,
    help=(
        "Number of recent OHLCV bars to fetch. "
        "100–1000 is sufficient for UI testing. "
        "For model evaluation, use 5000+ (1h), 2000+ (4h), or 500+ (1d). "
        "Large fetches may require exchange pagination and take time."
    ),
)

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

    fetch_mode = st.radio(
        "Fetch Mode",
        ["Recent Bars", "Date Range"],
        horizontal=True,
        key="fetch_mode",
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if fetch_mode == "Recent Bars":
            st.caption(
                f"Estimated coverage: **{estimate_coverage(limit, timeframe)}**"
                f" ({limit} × {timeframe})"
            )

            if st.button("Fetch OHLCV", type="primary", width="stretch"):
                with st.spinner(f"Fetching up to {limit} {symbol} {timeframe} bars from {exchange}..."):
                    try:
                        if limit <= 1000:
                            df = fetch_ohlcv(exchange, symbol, timeframe, limit=limit)
                        else:
                            from datetime import datetime as _dt, timezone as _tz

                            duration = limit * timeframe_to_timedelta(timeframe)
                            end_dt = _dt.now(_tz.utc)
                            start_dt = end_dt - duration
                            df = fetch_ohlcv_range(
                                exchange, symbol, timeframe,
                                start=start_dt, end=end_dt,
                            )
                            if len(df) > limit:
                                df = df.iloc[-limit:].reset_index(drop=True)

                        st.session_state["fetch_df"] = df
                        st.session_state["fetch_source"] = "fetched"
                        st.session_state["fetch_requested"] = limit
                        if len(df) < limit:
                            st.warning(
                                f"Requested {limit} bars, received {len(df)}. "
                                f"The exchange may not have this much history."
                            )
                        else:
                            st.success(f"Fetched {len(df)} bars")
                    except Exception as exc:
                        st.error(f"Fetch failed: {exc}")

        else:
            from datetime import datetime, timedelta, timezone

            default_end = datetime.now(timezone.utc)
            default_start = default_end - timedelta(days=30)

            date_start = st.date_input(
                "Start date (UTC)",
                value=default_start.date(),
                key="date_start",
            )
            date_end = st.date_input(
                "End date (UTC)",
                value=default_end.date(),
                key="date_end",
            )

            if st.button("Fetch Date Range", type="primary", width="stretch"):
                start_dt = datetime.combine(date_start, datetime.min.time(), tzinfo=timezone.utc)
                end_dt = datetime.combine(date_end, datetime.max.time(), tzinfo=timezone.utc)
                with st.spinner(
                    f"Fetching {symbol} {timeframe} {date_start} → {date_end}..."
                ):
                    try:
                        df = fetch_ohlcv_range(
                            exchange, symbol, timeframe,
                            start=start_dt, end=end_dt,
                        )
                        st.session_state["fetch_df"] = df
                        st.session_state["fetch_source"] = "fetched"
                        st.session_state["fetch_requested"] = None
                        st.success(f"Fetched {len(df)} bars")
                    except Exception as exc:
                        st.error(f"Fetch failed: {exc}")

        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            df = st.session_state["fetch_df"]
            st.metric("Rows fetched", len(df))

            # --- Source indicator ---
            source_label = st.session_state.get("fetch_source", "unknown")
            if source_label == "fetched":
                requested = st.session_state.get("fetch_requested")
                if requested:
                    st.caption(f"Source: freshly fetched  |  Requested: {requested} bars  |  Received: {len(df)} bars")
                else:
                    st.caption("Source: freshly fetched")
            elif source_label == "local":
                st.caption("Source: loaded from local Parquet")

            # --- Coverage summary ---
            st.divider()
            st.subheader("Data Coverage")

            t_min = df["timestamp"].min()
            t_max = df["timestamp"].max()
            duration = t_max - t_min

            c1, c2, c3 = st.columns(3)
            c1.caption(f"Start: {t_min}")
            c2.caption(f"End: {t_max}")
            c3.caption(f"Duration: {duration}")

            # --- Coverage delta ---
            requested = st.session_state.get("fetch_requested")
            if requested and len(df) < requested:
                st.warning(
                    f"Received {len(df)} of {requested} requested bars "
                    f"({requested - len(df)} fewer). The exchange may not "
                    f"have this much history for {timeframe}."
                )

            # --- Sizing warning ---
            warning = dataset_sizing_warning(len(df), timeframe)
            if warning:
                st.warning(warning)

            st.divider()
            st.subheader("Save to Local")

            if st.button("Save Raw", width="stretch"):
                path = save_ohlcv_parquet(df, exchange, symbol, timeframe, layer="raw")
                st.success(f"Saved to {path}")

            if st.button("Save Processed", width="stretch"):
                path = save_ohlcv_parquet(
                    df, exchange, symbol, timeframe, layer="processed"
                )
                st.success(f"Saved to {path}")

    with col2:
        if "fetch_df" in st.session_state and not st.session_state["fetch_df"].empty:
            df_full = st.session_state["fetch_df"]

            # --- Chart time range controls ---
            st.caption("**Chart Display Controls**")
            st.caption(
                "Select a narrower time range to see finer-grained candles. "
                "The full dataset is always used for storage, features, and modeling."
            )
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                chart_range = st.selectbox(
                    "Time Range",
                    ["Full range", "Last 1 day", "Last 7 days", "Last 30 days",
                     "Last 90 days", "Last 180 days", "Last 365 days",
                     "Custom range"],
                    key="fetch_chart_range",
                )
            with chart_col2:
                max_candles = st.selectbox(
                    "Max Candles",
                    [500, 1000, 2000, 5000],
                    index=1,
                    key="fetch_max_candles",
                )

            # Custom range date pickers
            custom_start = None
            custom_end = None
            if chart_range == "Custom range":
                from datetime import datetime, timezone

                ref_max = df_full["timestamp"].max()
                ref_min = df_full["timestamp"].min()
                cc1, cc2 = st.columns(2)
                with cc1:
                    custom_start = st.date_input(
                        "Start date",
                        value=ref_min.date() if pd.notna(ref_min) else datetime.now(timezone.utc).date(),
                        key="fetch_custom_start",
                    )
                with cc2:
                    custom_end = st.date_input(
                        "End date",
                        value=ref_max.date() if pd.notna(ref_max) else datetime.now(timezone.utc).date(),
                        key="fetch_custom_end",
                    )
                custom_start = pd.Timestamp(custom_start).tz_localize("utc") if custom_start else None
                custom_end = pd.Timestamp(custom_end).tz_localize("utc") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) if custom_end else None

            # Filter by chart range -> df_view
            try:
                df_view = filter_ohlcv_by_chart_range(
                    df_full, chart_range,
                    custom_start=custom_start, custom_end=custom_end,
                )
            except ValueError as exc:
                st.error(str(exc))
                df_view = df_full

            # Aggregation for display: df_view -> df_chart
            display_result = prepare_candlestick_display_data(df_view, max_bars=max_candles)
            df_chart = display_result["df_chart"]
            disp_summary = get_display_summary(df_full, df_view, display_result)

            # Render candlestick chart from df_chart
            fig = candlestick_chart(
                df_chart, title=f"{exchange} {symbol} {timeframe}"
            )
            st.plotly_chart(fig, width="stretch")

            # Display summary
            st.divider()
            st.caption("**Display Summary**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.caption(f"Full dataset: {disp_summary['full_rows']} rows")
            sc2.caption(f"Selected range: {disp_summary['view_rows']} rows")
            sc3.caption(f"Displayed candles: {disp_summary['displayed_candles']}")
            sc4.caption(f"Mode: {disp_summary['display_mode']}")

            if disp_summary["approx_bars_per_candle"]:
                st.caption(
                    f"~{disp_summary['approx_bars_per_candle']} raw bars per displayed candle. "
                    f"To see finer-grained candles, select a narrower time range."
                )

            sc5, sc6, sc7, sc8 = st.columns(4)
            sc5.caption(f"Range start: {disp_summary['range_start']}")
            sc6.caption(f"Range end: {disp_summary['range_end']}")
            sc7.caption(f"Chart start: {disp_summary['chart_start']}")
            sc8.caption(f"Chart end: {disp_summary['chart_end']}")

            if disp_summary["base_timeframe"]:
                st.caption(f"Base timeframe: {disp_summary['base_timeframe']}")

            # Table preview
            with st.expander("Data Preview"):
                preview_mode = st.radio(
                    "Preview Mode",
                    ["Latest raw rows", "Current chart candles"],
                    horizontal=True,
                    key="fetch_preview_mode",
                )
                if preview_mode == "Latest raw rows":
                    st.caption(
                        f"Showing latest {min(DEFAULT_PREVIEW_ROWS, len(df_full))} "
                        f"of {len(df_full)} total raw rows. "
                        f"Full data is used for storage, features, modeling, and evaluation."
                    )
                    st.dataframe(df_full.tail(DEFAULT_PREVIEW_ROWS), width="stretch")
                else:
                    label = "aggregated" if disp_summary["display_mode"] == "aggregated" else "raw"
                    st.caption(
                        f"Showing {len(df_chart)} {label} chart candles. "
                        f"These are for display only — modeling uses the full dataset."
                    )
                    st.dataframe(df_chart, width="stretch")
        else:
            if fetch_mode == "Recent Bars":
                st.info("Click 'Fetch OHLCV' to load market data.")
            else:
                st.info("Select a date range and click 'Fetch Date Range' to load data.")

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
                    st.session_state["fetch_source"] = "local"
                    st.success(f"Loaded {len(df)} bars")
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    if "local_df" in st.session_state and not st.session_state["local_df"].empty:
        df_full = st.session_state["local_df"]

        st.caption("**Chart Display Controls**")
        st.caption(
            "Select a narrower time range to see finer-grained candles. "
            "The full dataset is always used for storage, features, and modeling."
        )
        loc_col1, loc_col2 = st.columns(2)
        with loc_col1:
            local_chart_range = st.selectbox(
                "Time Range",
                ["Full range", "Last 1 day", "Last 7 days", "Last 30 days",
                 "Last 90 days", "Last 180 days", "Last 365 days",
                 "Custom range"],
                key="local_chart_range",
            )
        with loc_col2:
            local_max_candles = st.selectbox(
                "Max Candles",
                [500, 1000, 2000, 5000],
                index=1,
                key="local_max_candles",
            )

        # Custom range date pickers
        local_custom_start = None
        local_custom_end = None
        if local_chart_range == "Custom range":
            from datetime import datetime, timezone

            ref_max = df_full["timestamp"].max()
            ref_min = df_full["timestamp"].min()
            lc1, lc2 = st.columns(2)
            with lc1:
                local_custom_start = st.date_input(
                    "Start date",
                    value=ref_min.date() if pd.notna(ref_min) else datetime.now(timezone.utc).date(),
                    key="local_custom_start",
                )
            with lc2:
                local_custom_end = st.date_input(
                    "End date",
                    value=ref_max.date() if pd.notna(ref_max) else datetime.now(timezone.utc).date(),
                    key="local_custom_end",
                )
            local_custom_start = pd.Timestamp(local_custom_start).tz_localize("utc") if local_custom_start else None
            local_custom_end = pd.Timestamp(local_custom_end).tz_localize("utc") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) if local_custom_end else None

        # Filter by chart range -> df_view
        try:
            df_view_local = filter_ohlcv_by_chart_range(
                df_full, local_chart_range,
                custom_start=local_custom_start, custom_end=local_custom_end,
            )
        except ValueError as exc:
            st.error(str(exc))
            df_view_local = df_full

        local_display = prepare_candlestick_display_data(
            df_view_local, max_bars=local_max_candles
        )
        local_chart_df = local_display["df_chart"]
        local_summary = get_display_summary(df_full, df_view_local, local_display)

        fig = candlestick_chart(
            local_chart_df,
            title=f"{exchange} {symbol} {timeframe} ({layer})",
        )
        st.plotly_chart(fig, width="stretch")

        # Display summary
        st.divider()
        st.caption("**Display Summary**")
        ls1, ls2, ls3, ls4 = st.columns(4)
        ls1.caption(f"Full dataset: {local_summary['full_rows']} rows")
        ls2.caption(f"Selected range: {local_summary['view_rows']} rows")
        ls3.caption(f"Displayed candles: {local_summary['displayed_candles']}")
        ls4.caption(f"Mode: {local_summary['display_mode']}")

        if local_summary["approx_bars_per_candle"]:
            st.caption(
                f"~{local_summary['approx_bars_per_candle']} raw bars per displayed candle. "
                f"To see finer-grained candles, select a narrower time range."
            )

        with st.expander("Data Preview"):
            preview_mode = st.radio(
                "Preview Mode",
                ["Latest raw rows", "Current chart candles"],
                horizontal=True,
                key="local_preview_mode",
            )
            if preview_mode == "Latest raw rows":
                st.caption(
                    f"Showing latest {min(DEFAULT_PREVIEW_ROWS, len(df_full))} "
                    f"of {len(df_full)} total raw rows."
                )
                st.dataframe(df_full.tail(DEFAULT_PREVIEW_ROWS), width="stretch")
            else:
                label = "aggregated" if local_summary["display_mode"] == "aggregated" else "raw"
                st.caption(
                    f"Showing {len(local_chart_df)} {label} chart candles. "
                    f"These are for display only — modeling uses the full dataset."
                )
                st.dataframe(local_chart_df, width="stretch")

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
            source_label = "freshly fetched"
            fetched_req = st.session_state.get("fetch_requested")
            if fetched_req:
                source_label += f" (requested {fetched_req} bars)"
    else:
        if "local_df" in st.session_state and not st.session_state["local_df"].empty:
            df_source = st.session_state["local_df"]
            source_label = "loaded from local Parquet"

    if df_source is not None and not df_source.empty:
        st.caption(f"Active data: **{source_label}** | Rows: {len(df_source)} | "
                   f"Symbol: {df_source['symbol'].iloc[0] if 'symbol' in df_source.columns else 'N/A'} | "
                   f"Timeframe: {df_source['timeframe'].iloc[0] if 'timeframe' in df_source.columns else 'N/A'}")

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

        # -------------------------------------------------------------------
        # Full Evaluation Report
        # -------------------------------------------------------------------

        st.divider()
        st.subheader("Full Evaluation Report")
        st.caption(
            "Runs all task types × horizons × models in one batch. "
            "Failed combinations are recorded as skipped with a reason. "
            "All outputs are experimental — they do not constitute financial advice."
        )

        full_test_pct = st.slider(
            "Full Report — Test Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="full_test_pct",
        ) / 100.0

        if st.button("Run Full Evaluation Report", type="primary", width="stretch"):
            with st.spinner("Running full evaluation across all horizons..."):
                try:
                    report = run_full_evaluation_report(
                        df_eval,
                        task_types=("regression", "classification"),
                        horizons=(1, 4, 24),
                        test_size=full_test_pct,
                    )
                    report["summary"]["test_size_pct"] = full_test_pct * 100
                    st.session_state["full_report"] = report
                except Exception as exc:
                    st.error(f"Full evaluation failed: {exc}")

        if "full_report" in st.session_state:
            report = st.session_state["full_report"]
            summary = report["summary"]
            metrics_df = report["metrics_df"]
            skipped_df = report["skipped_df"]

            import plotly.graph_objects as go
            from sklearn.metrics import confusion_matrix

            # --- Dataset summary ---
            st.divider()
            st.subheader("Dataset Summary")
            ds_cols = st.columns(4)
            ds_cols[0].metric("Rows", summary["row_count"])
            ds_cols[1].metric("Features", summary["feature_count"])
            ds_cols[2].metric("Combinations", summary["total_combinations"])
            ds_cols[3].metric("Successful", summary["successful"])
            st.caption(
                f"Exchange: {summary['exchange']}  |  "
                f"Symbol: {summary['symbol']}  |  "
                f"Timeframe: {summary['timeframe']}"
            )
            st.caption(
                f"Data range: {summary['min_timestamp']} → {summary['max_timestamp']}"
            )
            if not summary["lightgbm_available"]:
                st.info("LightGBM is not installed. LightGBM models were skipped.")

            # --- Skip / Failure log ---
            if not skipped_df.empty:
                st.divider()
                st.subheader("Skipped Combinations")
                st.dataframe(skipped_df, width="stretch", hide_index=True)
                st.caption(
                    "Each skipped model/horizon combination is listed with a reason. "
                    "Common causes: missing target column, insufficient valid rows "
                    "after NaN removal, or LightGBM not installed."
                )

            if metrics_df.empty:
                st.info("No metrics were produced. All combinations were skipped.")
            else:
                # --- Full Metrics Table ---
                st.divider()
                st.subheader("Full Metrics Table")
                st.dataframe(metrics_df, width="stretch", hide_index=True)

                # --- Metric bar charts by model and horizon ---
                st.divider()
                st.subheader("Metrics by Model and Horizon")

                reg_df = metrics_df[metrics_df["task_type"] == "regression"]
                cls_df = metrics_df[metrics_df["task_type"] == "classification"]

                # Regression bar charts
                if not reg_df.empty:
                    st.caption("**Regression**")
                    reg_metrics = ["mae", "rmse", "directional_accuracy", "spearman_r"]
                    reg_cols = st.columns(min(4, len(reg_metrics)))
                    for i, metric in enumerate(reg_metrics):
                        if metric not in reg_df.columns or reg_df[metric].dropna().empty:
                            continue
                        with reg_cols[i % len(reg_cols)]:
                            pivot = reg_df.pivot_table(
                                index="model_name", columns="horizon",
                                values=metric, aggfunc="first"
                            )
                            fig_bar = go.Figure()
                            for h in pivot.columns:
                                fig_bar.add_trace(go.Bar(
                                    name=f"h={h}", x=pivot.index, y=pivot[h],
                                    text=[f"{v:.4f}" if pd.notna(v) else "" for v in pivot[h]],
                                    textposition="auto",
                                ))
                            fig_bar.update_layout(
                                title=metric.upper(),
                                barmode="group",
                                template="plotly_dark",
                                height=300,
                                margin=dict(t=40, b=80),
                                xaxis_tickangle=-30,
                            )
                            st.plotly_chart(fig_bar, width="stretch")

                # Classification bar charts
                if not cls_df.empty:
                    st.caption("**Classification**")
                    cls_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
                    cls_cols = st.columns(min(3, len(cls_metrics)))
                    for i, metric in enumerate(cls_metrics):
                        if metric not in cls_df.columns or cls_df[metric].dropna().empty:
                            continue
                        with cls_cols[i % len(cls_cols)]:
                            pivot = cls_df.pivot_table(
                                index="model_name", columns="horizon",
                                values=metric, aggfunc="first"
                            )
                            fig_bar = go.Figure()
                            for h in pivot.columns:
                                fig_bar.add_trace(go.Bar(
                                    name=f"h={h}", x=pivot.index, y=pivot[h],
                                    text=[f"{v:.4f}" if pd.notna(v) else "" for v in pivot[h]],
                                    textposition="auto",
                                ))
                            fig_bar.update_layout(
                                title=metric.replace("_", " ").title(),
                                barmode="group",
                                template="plotly_dark",
                                height=300,
                                margin=dict(t=40, b=80),
                                xaxis_tickangle=-30,
                            )
                            st.plotly_chart(fig_bar, width="stretch")

                # --- Best Model Summary table ---
                st.divider()
                st.subheader("Best Model Summary")
                st.caption(
                    "Best model by each metric per task type and horizon. "
                    "\"Best\" refers only to the metric listed, under this "
                    "specific historical test split — not overall superiority."
                )

                best_df = build_best_model_summary(metrics_df)
                if not best_df.empty:
                    # Format numeric columns
                    for col in best_df.columns:
                        if col.startswith("best_") and "_model" not in col:
                            if "accuracy" in col or "f1" in col or "auc" in col or "dir" in col:
                                best_df[col] = best_df[col].apply(
                                    lambda x: f"{x:.4f}" if pd.notna(x) else ""
                                )
                            else:
                                best_df[col] = best_df[col].apply(
                                    lambda x: f"{x:.6f}" if pd.notna(x) else ""
                                )
                    st.dataframe(best_df, width="stretch", hide_index=True)

                    st.download_button(
                        label="Download Best Model Summary CSV",
                        data=best_df.to_csv(index=False),
                        file_name="best_model_summary.csv",
                        mime="text/csv",
                    )

                # --- Detailed model view ---
                st.divider()
                st.subheader("Detailed Model View")
                st.caption("Select a task type, horizon, and model to inspect predictions.")

                detail_col1, detail_col2, detail_col3 = st.columns(3)
                all_task_types = sorted(metrics_df["task_type"].unique())
                with detail_col1:
                    detail_tt = st.selectbox(
                        "Task Type", all_task_types, key="detail_tt"
                    )
                tt_subset = metrics_df[metrics_df["task_type"] == detail_tt]
                all_horizons = sorted(tt_subset["horizon"].unique())
                with detail_col2:
                    detail_h = st.selectbox(
                        "Horizon", all_horizons, key="detail_h"
                    )
                h_subset = tt_subset[tt_subset["horizon"] == detail_h]
                all_models = sorted(h_subset["model_name"].unique())
                with detail_col3:
                    detail_model = st.selectbox(
                        "Model", all_models, key="detail_model"
                    )

                # Re-run the single model to get predictions for charts
                target_col = get_target_column(detail_tt, detail_h)
                if target_col in df_eval.columns:
                    detail_result = compare_baselines_and_models(
                        df_eval,
                        task_type=detail_tt,
                        horizon=detail_h,
                        test_size=full_test_pct,
                        include_tabular=True,
                    )
                    detail_preds = detail_result["predictions"]
                    detail_preds = detail_preds[
                        detail_preds["model_name"] == detail_model
                    ]

                    if not detail_preds.empty:
                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            # y_true vs y_pred line chart
                            fig_line = go.Figure()
                            fig_line.add_trace(go.Scatter(
                                x=detail_preds["timestamp"],
                                y=detail_preds["y_true"],
                                mode="lines+markers",
                                name="Actual",
                                marker=dict(size=3),
                            ))
                            fig_line.add_trace(go.Scatter(
                                x=detail_preds["timestamp"],
                                y=detail_preds["y_pred"],
                                mode="lines+markers",
                                name="Predicted",
                                marker=dict(size=3),
                            ))
                            fig_line.update_layout(
                                title=f"{detail_model} — {target_col}",
                                xaxis_title="Time (UTC)",
                                template="plotly_dark",
                                height=350,
                            )
                            st.plotly_chart(fig_line, width="stretch")

                        with chart_col2:
                            if detail_tt == "regression":
                                # Scatter: y_true vs y_pred
                                fig_scatter = go.Figure()
                                fig_scatter.add_trace(go.Scatter(
                                    x=detail_preds["y_true"],
                                    y=detail_preds["y_pred"],
                                    mode="markers",
                                    marker=dict(size=5, opacity=0.6),
                                    name="Predictions",
                                ))
                                # Identity line
                                vals = detail_preds["y_true"]
                                lo, hi = float(vals.min()), float(vals.max())
                                fig_scatter.add_trace(go.Scatter(
                                    x=[lo, hi], y=[lo, hi],
                                    mode="lines",
                                    name="Perfect",
                                    line=dict(dash="dash", color="gray"),
                                ))
                                fig_scatter.update_layout(
                                    title=f"{detail_model} — Actual vs Predicted",
                                    xaxis_title="Actual",
                                    yaxis_title="Predicted",
                                    template="plotly_dark",
                                    height=350,
                                )
                                st.plotly_chart(fig_scatter, width="stretch")

                                # Residual plot
                                residuals = (
                                    detail_preds["y_true"].values
                                    - detail_preds["y_pred"].values
                                )
                                fig_res = go.Figure()
                                fig_res.add_trace(go.Scatter(
                                    x=detail_preds["timestamp"],
                                    y=residuals,
                                    mode="markers",
                                    marker=dict(size=4, opacity=0.6),
                                    name="Residuals",
                                ))
                                fig_res.add_hline(
                                    y=0, line_dash="dash", line_color="gray"
                                )
                                fig_res.update_layout(
                                    title=f"{detail_model} — Residuals",
                                    xaxis_title="Time (UTC)",
                                    yaxis_title="Residual (Actual − Predicted)",
                                    template="plotly_dark",
                                    height=300,
                                )
                                st.plotly_chart(fig_res, width="stretch")
                            else:
                                # Confusion matrix
                                yt = detail_preds["y_true"].values
                                yp = detail_preds["y_pred"].values
                                cm = confusion_matrix(yt, yp)
                                labels = ["Down (0)", "Up (1)"]
                                fig_cm = go.Figure(data=go.Heatmap(
                                    z=cm,
                                    x=labels,
                                    y=labels,
                                    text=cm,
                                    texttemplate="%{text}",
                                    colorscale="Blues",
                                    showscale=False,
                                ))
                                fig_cm.update_layout(
                                    title=f"{detail_model} — Confusion Matrix",
                                    xaxis_title="Predicted",
                                    yaxis_title="Actual",
                                    template="plotly_dark",
                                    height=350,
                                )
                                st.plotly_chart(fig_cm, width="stretch")

                                # Probability chart if available
                                if "y_prob" in detail_preds.columns:
                                    fig_prob = go.Figure()
                                    fig_prob.add_trace(go.Scatter(
                                        x=detail_preds["timestamp"],
                                        y=detail_preds["y_prob"],
                                        mode="lines+markers",
                                        name="Predicted Probability",
                                        marker=dict(size=3),
                                    ))
                                    fig_prob.add_hline(
                                        y=0.5, line_dash="dash", line_color="gray",
                                        annotation_text="0.5 threshold"
                                    )
                                    fig_prob.update_layout(
                                        title=f"{detail_model} — Probability",
                                        xaxis_title="Time (UTC)",
                                        yaxis_title="P(Up)",
                                        template="plotly_dark",
                                        height=300,
                                    )
                                    st.plotly_chart(fig_prob, width="stretch")

                # --- Cautious textual analysis ---
                st.divider()
                st.subheader("Report Analysis")
                analysis_text = generate_report_analysis(
                    metrics_df, summary, skipped_df
                )
                st.markdown(analysis_text)

                st.download_button(
                    label="Download Analysis (Markdown)",
                    data=analysis_text,
                    file_name="full_evaluation_analysis.md",
                    mime="text/markdown",
                )

                # --- Downloads ---
                st.divider()
                st.subheader("Downloads")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="Download Metrics CSV",
                        data=metrics_df.to_csv(index=False),
                        file_name="full_evaluation_metrics.csv",
                        mime="text/csv",
                    )
                with dl_col2:
                    st.download_button(
                        label="Download Skipped CSV",
                        data=skipped_df.to_csv(index=False),
                        file_name="full_evaluation_skipped.csv",
                        mime="text/csv",
                    )

        # -------------------------------------------------------------------
        # Forward Forecast
        # -------------------------------------------------------------------

        st.divider()
        st.subheader("Forward Forecast")
        st.caption(
            "Train a model on ALL available labeled historical data and produce "
            "an experimental forecast. "
            "This is NOT a trading signal — it is a research output that uses "
            "the most recent observation for which the future outcome is unknown."
        )

        forecast_source = st.radio(
            "Forecast Data",
            ["Use Full Report Features", "Use Session Features"],
            horizontal=True,
            key="forecast_source",
        )

        df_forecast: pd.DataFrame | None = None

        if forecast_source == "Use Full Report Features":
            df_forecast = df_eval
        else:
            if "features_df" in st.session_state and not st.session_state["features_df"].empty:
                df_forecast = st.session_state["features_df"]
            else:
                st.warning("No session features. Run the feature pipeline first.")

        if df_forecast is not None and not df_forecast.empty:
            # --- Single-Point Forecast ---
            st.divider()
            st.subheader("Single-Point Forecast")

            fc_col1, fc_col2, fc_col3 = st.columns(3)

            with fc_col1:
                fc_task_type = st.selectbox(
                    "Task Type",
                    ["regression", "classification"],
                    key="fc_task_type",
                )
            with fc_col2:
                fc_horizon = st.selectbox(
                    "Horizon",
                    [1, 4, 24],
                    key="fc_horizon",
                )
            with fc_col3:
                fc_target = get_target_column(fc_task_type, fc_horizon)
                avail_models: list[str] = []
                if fc_task_type == "regression":
                    avail_models = ["Ridge"]
                    if _HAS_LIGHTGBM:
                        avail_models.append("LightGBM")
                    avail_models.extend(["Last Return", "Moving Average", "Zero Return"])
                else:
                    avail_models = ["Logistic Regression"]
                    if _HAS_LIGHTGBM:
                        avail_models.append("LightGBM")
                    avail_models.extend(["Momentum Direction", "Majority Class"])

                fc_model = st.selectbox(
                    "Model",
                    avail_models,
                    key="fc_model",
                )

            fc_feature_cols = get_default_feature_columns(df_forecast)
            st.caption(f"Target: **{fc_target}** | Features: {len(fc_feature_cols)} columns")
            st.caption(
                "Training uses all rows with valid features and known targets. "
                "The forecast is the latest row with valid features — its target "
                "is unknown (future outcome)."
            )

            if st.button("Run Forward Forecast", type="primary", width="stretch"):
                with st.spinner(
                    f"Training {fc_model} on all labeled data, "
                    f"forecasting {fc_target}..."
                ):
                    try:
                        fc_result = forward_forecast(
                            df_forecast,
                            task_type=fc_task_type,
                            horizon=fc_horizon,
                            model_name=fc_model,
                            feature_columns=fc_feature_cols,
                        )
                        st.session_state["forecast_result"] = fc_result
                    except Exception as exc:
                        st.error(f"Forward forecast failed: {exc}")

            if "forecast_result" in st.session_state:
                fr = st.session_state["forecast_result"]

                st.divider()
                st.subheader("Forecast Output")

                if "error" in fr:
                    st.error(fr["error"])
                else:
                    if fc_task_type == "regression":
                        fc_cols = st.columns(3)
                        fc_cols[0].metric("Latest Timestamp", str(fr["latest_timestamp"]))
                        fc_cols[1].metric("Horizon (bars)", fr["horizon"])
                        fc_cols[2].metric(
                            "Predicted Log Return",
                            f"{fr['predicted_log_return']:.8f}",
                        )

                        fc_cols2 = st.columns(3)
                        fc_cols2[0].metric(
                            "Latest Close",
                            f"{fr['latest_close']:.4f}" if fr.get("latest_close") else "N/A",
                        )
                        fc_cols2[1].metric(
                            "Implied Future Close",
                            f"{fr['estimated_future_close']:.4f}" if fr.get("estimated_future_close") else "N/A",
                        )
                        fc_cols2[2].metric("Training Rows", fr["training_rows"])

                        if "historical_mae" in fr:
                            st.caption(
                                f"Historical test MAE (for context only): "
                                f"{fr['historical_mae']:.6f}"
                            )
                        if "historical_rmse" in fr:
                            st.caption(
                                f"Historical test RMSE (for context only): "
                                f"{fr['historical_rmse']:.6f}"
                            )
                    else:
                        fc_cols = st.columns(3)
                        fc_cols[0].metric("Latest Timestamp", str(fr["latest_timestamp"]))
                        fc_cols[1].metric("Horizon (bars)", fr["horizon"])
                        fc_cols[2].metric(
                            "Predicted Direction",
                            "Up (1)" if fr["predicted_class"] == 1 else "Down (0)",
                        )

                        if fr.get("predicted_probability") is not None:
                            fc_cols2 = st.columns(3)
                            fc_cols2[0].metric(
                                "Predicted Probability",
                                f"{fr['predicted_probability']:.4f}",
                            )
                            fc_cols2[1].metric("Training Rows", fr["training_rows"])

                    st.warning(
                        "This is an **experimental forecast** based on historical "
                        "patterns. It does NOT constitute financial advice. "
                        "Crypto markets are noisy, non-stationary, and high-risk."
                    )

                    # Save forecast
                    st.divider()
                    st.subheader("Save Forecast")
                    if st.button("Save Forecast to Parquet", width="stretch"):
                        fc_save = pd.DataFrame([{
                            "timestamp": fr["latest_timestamp"],
                            "horizon": fr["horizon"],
                            "target_column": fc_target,
                            "model_name": fc_model,
                            **{k: v for k, v in fr.items()
                               if k not in ("latest_timestamp", "horizon",
                                            "latest_close", "training_rows",
                                            "historical_mae", "historical_rmse",
                                            "historical_balanced_accuracy",
                                            "historical_directional_accuracy")},
                        }])
                        path = save_forecast_parquet(
                            fc_save, exchange=exchange, symbol=symbol,
                            timeframe=timeframe, model_name=fc_model,
                            target_column=fc_target,
                        )
                        st.success(f"Saved forecast to {path}")

            # --- Forecast Path Chart ---
            st.divider()
            st.subheader("Forecast Path Chart")
            st.caption(
                "Train regression models on supported horizons (1, 4, 24) and "
                "plot an experimental future close-price path. "
                "Only sparse direct-horizon points are predicted — intermediate "
                "points are connected by interpolation for visualization."
            )

            fp_col1, fp_col2, fp_col3 = st.columns(3)

            with fp_col1:
                fp_model = st.selectbox(
                    "Regression Model",
                    ["Ridge"] + (["LightGBM"] if _HAS_LIGHTGBM else []),
                    key="fp_model",
                )
            with fp_col2:
                fp_path_len = st.selectbox(
                    "Path Length (bars)",
                    [6, 12, 24, 48, 72, 168],
                    index=2,  # default 24
                    key="fp_path_len",
                )
            with fp_col3:
                chart_history_bars = st.selectbox(
                    "Chart History (bars)",
                    [100, 200, 500, 1000],
                    index=1,  # default 200
                    key="chart_history_bars",
                )

            if st.button("Generate Forecast Path", type="primary", width="stretch"):
                with st.spinner("Generating sparse direct-horizon forecast path..."):
                    try:
                        fp_result = forecast_path(
                            df_forecast,
                            model_name=fp_model,
                            path_length=fp_path_len,
                            feature_columns=fc_feature_cols,
                            timeframe=timeframe,
                        )
                        st.session_state["forecast_path_result"] = fp_result
                    except Exception as exc:
                        st.error(f"Forecast path failed: {exc}")

            if "forecast_path_result" in st.session_state:
                fpr = st.session_state["forecast_path_result"]

                if "error" in fpr:
                    st.error(fpr["error"])
                else:
                    # --- Build chart ---
                    import plotly.graph_objects as go

                    chart_df = fpr["chart_history"]
                    latest_ts = fpr["latest_timestamp"]

                    # Restrict to requested chart history
                    if len(chart_df) > chart_history_bars:
                        chart_df = chart_df.iloc[-chart_history_bars:]

                    fig_fp = go.Figure()

                    # Add candlestick for actual data
                    if all(c in chart_df.columns for c in ["open", "high", "low", "close"]):
                        fig_fp.add_trace(go.Candlestick(
                            x=chart_df["timestamp"],
                            open=chart_df["open"],
                            high=chart_df["high"],
                            low=chart_df["low"],
                            close=chart_df["close"],
                            name="Actual OHLCV",
                            increasing_line_color="#26a69a",
                            decreasing_line_color="#ef5350",
                        ))

                    # Add predicted close path
                    points = [p for p in fpr["path_points"] if "error" not in p]
                    if points:
                        # Start from latest observed close
                        path_ts = [latest_ts]
                        path_closes = [fpr["latest_close"]]

                        for p in points:
                            path_ts.append(p["forecast_timestamp"])
                            path_closes.append(p["estimated_future_close"])

                        fig_fp.add_trace(go.Scatter(
                            x=path_ts,
                            y=path_closes,
                            mode="lines+markers",
                            name="Experimental Forecast Path",
                            line=dict(color="#ffa726", width=2, dash="dash"),
                            marker=dict(size=8, symbol="diamond", color="#ffa726"),
                        ))

                    # Vertical marker at latest observed timestamp
                    add_datetime_vertical_marker(
                        fig_fp, x=latest_ts, text="Latest observed",
                        line_dash="dot", line_color="gray",
                    )

                    fig_fp.update_layout(
                        title=(
                            f"{fp_model} — Experimental Forecast Path "
                            f"({symbol} {timeframe})"
                        ),
                        xaxis_title="Time (UTC)",
                        yaxis_title="Close Price",
                        template="plotly_dark",
                        height=500,
                        hovermode="x unified",
                    )

                    # Add padding around forecast
                    if points:
                        last_future_ts = points[-1]["forecast_timestamp"]
                        fig_fp.update_xaxes(
                            range=[
                                chart_df["timestamp"].min() if not chart_df.empty else latest_ts,
                                last_future_ts,
                            ]
                        )

                    st.plotly_chart(fig_fp, width="stretch")

                    # --- Forecast path table ---
                    st.divider()
                    st.subheader("Forecast Path Table")
                    st.caption(
                        "Sparse direct-horizon forecast path. "
                        "Only supported horizons (1, 4, 24) are predicted. "
                        "Missing intermediate steps are interpolated on the chart "
                        "for visualization only."
                    )

                    table_rows = []
                    for p in fpr["path_points"]:
                        row = {
                            "step": p.get("forecast_step", p.get("horizon")),
                            "timestamp": p.get("forecast_timestamp", ""),
                            "target": p.get("target_column", ""),
                            "predicted_log_return": (
                                f"{p['predicted_log_return']:.8f}"
                                if "predicted_log_return" in p else "N/A"
                            ),
                            "estimated_close": (
                                f"{p['estimated_future_close']:.4f}"
                                if "estimated_future_close" in p else "N/A"
                            ),
                        }
                        if "error" in p:
                            row["error"] = p["error"]
                        table_rows.append(row)
                    table_df = pd.DataFrame(table_rows)
                    st.dataframe(table_df, width="stretch", hide_index=True)

                    # Context info
                    st.caption(
                        f"Latest observed: {fpr['latest_timestamp']} | "
                        f"Latest close: {fpr['latest_close']:.4f} | "
                        f"Model: {fpr['model_name']} | "
                        f"Training rows: {fpr['training_rows']}"
                    )

                    # Cautions
                    st.warning(
                        "**Experimental forward forecast based on the selected "
                        "model and available historical features.**\n\n"
                        "This is not a trading signal or investment advice. "
                        "The forecast path is model-estimated and may be unstable, "
                        "especially with small datasets. "
                        "Crypto markets are noisy, non-stationary, and high-risk. "
                        "Past patterns do not guarantee future outcomes.\n\n"
                        "Only horizons 1, 4, and 24 bars are directly predicted "
                        "(sparse direct-horizon). The connecting line is "
                        "interpolated for visualization."
                    )
