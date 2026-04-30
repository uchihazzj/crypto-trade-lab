"""OHLCV display aggregation for chart rendering and table preview.

Display aggregation reduces the number of visible candles without affecting
the full dataset. The full dataset is used only for: Parquet storage, data
quality checks, feature generation, model evaluation, and forecast fitting.

Key invariant: df_full is never mutated. df_chart and df_preview are
derived display-only artifacts.

Random sampling and every-N-row sampling are deliberately not used because
they destroy OHLCV semantics (open/high/low/close relationships).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Max candles to render by default in Plotly candlestick charts.
DEFAULT_MAX_CANDLES = 1000

# Default preview row counts.
DEFAULT_PREVIEW_ROWS = 500


def aggregate_ohlcv_by_count(
    df: pd.DataFrame,
    target_bars: int = DEFAULT_MAX_CANDLES,
) -> pd.DataFrame:
    """Aggregate OHLCV rows into approximately *target_bars* display candles.

    Groups consecutive rows into ~target_bars groups. Each output row
    preserves OHLCV semantics:

    - open  = first open in the group
    - high  = max high in the group
    - low   = min low in the group
    - close = last close in the group
    - volume = sum of volume in the group

    Prices are never averaged. Timestamp is the first timestamp in the group.

    Parameters
    ----------
    df : pd.DataFrame
        Stable-schema OHLCV DataFrame sorted by timestamp ascending.
    target_bars : int
        Desired number of output display candles.

    Returns
    -------
    pd.DataFrame
        Aggregated OHLCV DataFrame with the same column schema plus optional
        metadata columns (source_start, source_end, source_rows).
    """
    n = len(df)
    if n == 0:
        return df.copy()

    if n <= target_bars:
        result = df.copy()
        result["source_start"] = result["timestamp"]
        result["source_end"] = result["timestamp"]
        result["source_rows"] = 1
        return result

    # Group size: each display candle represents ~group_size raw bars
    group_size = int(np.ceil(n / target_bars))
    # Assign group IDs
    groups = np.arange(n) // group_size

    # Aggregate
    agg: dict = {
        "exchange": ("exchange", "first"),
        "symbol": ("symbol", "first"),
        "timeframe": ("timeframe", "first"),
        "timestamp": ("timestamp", "first"),
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "volume": ("volume", "sum"),
    }

    # Only include columns that exist in df
    agg = {col: fn for col, fn in agg.items() if col in df.columns}

    grouped = df.groupby(groups, sort=False).agg(**{
        col: pd.NamedAgg(column=col, aggfunc=fn)
        for col, (_, fn) in agg.items()
    })

    # Add metadata — preserve tz-aware timestamps by using .tolist()
    grouped["source_start"] = df.groupby(groups, sort=False)["timestamp"].min().tolist()
    grouped["source_end"] = df.groupby(groups, sort=False)["timestamp"].max().tolist()
    grouped["source_rows"] = df.groupby(groups, sort=False).size().tolist()

    return grouped.reset_index(drop=True)


def prepare_candlestick_display_data(
    df: pd.DataFrame,
    max_bars: int = DEFAULT_MAX_CANDLES,
) -> dict:
    """Prepare display data for Plotly candlestick rendering.

    Parameters
    ----------
    df : pd.DataFrame
        Full or filtered OHLCV DataFrame sorted by timestamp ascending.
    max_bars : int
        Maximum number of candles to display.

    Returns
    -------
    dict
        Keys: df_chart (pd.DataFrame), input_rows (int), displayed_rows (int),
        display_mode (str), approx_bars_per_candle (int or None),
        chart_start, chart_end.
    """
    input_rows = len(df)

    if input_rows <= max_bars:
        df_chart = df.copy()
        display_mode = "raw"
        approx = None
    else:
        df_chart = aggregate_ohlcv_by_count(df, target_bars=max_bars)
        display_mode = "aggregated"
        approx = int(np.ceil(input_rows / max_bars))

    chart_start = df_chart["timestamp"].iloc[0] if len(df_chart) > 0 else None
    chart_end = df_chart["timestamp"].iloc[-1] if len(df_chart) > 0 else None

    return {
        "df_chart": df_chart,
        "input_rows": input_rows,
        "displayed_rows": len(df_chart),
        "display_mode": display_mode,
        "approx_bars_per_candle": approx,
        "chart_start": chart_start,
        "chart_end": chart_end,
    }


def filter_by_time_range(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter a sorted OHLCV DataFrame to [*start*, *end*].

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame sorted by timestamp ascending.
    start : pd.Timestamp or None
        Inclusive start of the range.
    end : pd.Timestamp or None
        Inclusive end of the range.

    Returns
    -------
    pd.DataFrame
        Filtered view (df_view). Not mutated.
    """
    result = df.copy()

    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None and result["timestamp"].dt.tz is not None:
            start_ts = start_ts.tz_localize("utc")
        result = result[result["timestamp"] >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None and result["timestamp"].dt.tz is not None:
            end_ts = end_ts.tz_localize("utc")
        result = result[result["timestamp"] <= end_ts]

    return result


def filter_ohlcv_by_chart_range(
    df_full: pd.DataFrame,
    range_option: str,
    custom_start: pd.Timestamp | None = None,
    custom_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter *df_full* to a df_view based on a named chart range option.

    Uses ``df_full["timestamp"].max()`` as the reference point for relative
    ranges, NOT ``pd.Timestamp.now()``. This ensures historical datasets
    produce correct range windows.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full OHLCV DataFrame sorted by timestamp ascending.
    range_option : str
        One of: "Full range", "Last 1 day", "Last 7 days", "Last 30 days",
        "Last 90 days", "Last 180 days", "Last 365 days", "Custom range".
    custom_start : pd.Timestamp or None
        Start of custom range (only used when range_option is "Custom range").
    custom_end : pd.Timestamp or None
        End of custom range (only used when range_option is "Custom range").

    Returns
    -------
    pd.DataFrame
        Filtered df_view. Not mutated.

    Raises
    ------
    ValueError
        If *range_option* is "Custom range" and custom_start/custom_end
        are not provided, or if the selected range has no rows.
    """
    if df_full.empty:
        return df_full.copy()

    reference_ts = df_full["timestamp"].max()

    range_days: dict[str, int] = {
        "Last 1 day": 1,
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last 180 days": 180,
        "Last 365 days": 365,
    }

    if range_option == "Full range":
        start = None
        end = None
    elif range_option == "Custom range":
        if custom_start is None or custom_end is None:
            raise ValueError(
                "Custom range requires both custom_start and custom_end."
            )
        start = pd.Timestamp(custom_start)
        end = pd.Timestamp(custom_end)
    elif range_option in range_days:
        days = range_days[range_option]
        start = reference_ts - pd.Timedelta(days=days)
        end = reference_ts
    else:
        raise ValueError(
            f"Unknown range_option {range_option!r}. "
            f"Expected one of: Full range, Last 1 day, Last 7 days, "
            f"Last 30 days, Last 90 days, Last 180 days, Last 365 days, "
            f"Custom range."
        )

    df_view = filter_by_time_range(df_full, start=start, end=end)

    if df_view.empty:
        raise ValueError(
            f"No data in selected range: {range_option}. "
            f"Data spans {df_full['timestamp'].min()} → {reference_ts}."
        )

    return df_view


def get_display_summary(
    df_full: pd.DataFrame,
    df_view: pd.DataFrame,
    display_result: dict,
) -> dict:
    """Return a display summary dict for UI rendering.

    Parameters
    ----------
    df_full : pd.DataFrame
        The full active dataset.
    df_view : pd.DataFrame
        The time-filtered view (may equal df_full).
    display_result : dict
        Output of ``prepare_candlestick_display_data``.

    Returns
    -------
    dict
        Keys: full_rows, view_rows, displayed_candles, display_mode,
        approx_bars_per_candle, range_start, range_end,
        chart_start, chart_end, base_timeframe.
    """
    base_tf = (
        df_full["timeframe"].iloc[0]
        if "timeframe" in df_full.columns and len(df_full) > 0
        else None
    )
    range_start = df_view["timestamp"].min() if len(df_view) > 0 else None
    range_end = df_view["timestamp"].max() if len(df_view) > 0 else None

    return {
        "full_rows": len(df_full),
        "view_rows": len(df_view),
        "displayed_candles": display_result["displayed_rows"],
        "display_mode": display_result["display_mode"],
        "approx_bars_per_candle": display_result["approx_bars_per_candle"],
        "range_start": range_start,
        "range_end": range_end,
        "chart_start": display_result["chart_start"],
        "chart_end": display_result["chart_end"],
        "base_timeframe": base_tf,
    }
