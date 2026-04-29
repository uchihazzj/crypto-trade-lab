"""Plotly visualization utilities for crypto market data."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def candlestick_chart(df: pd.DataFrame, title: str = "OHLCV") -> go.Figure:
    """Build a Plotly candlestick chart with volume bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: timestamp, open, high, low, close, volume.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    candlestick = go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLCV",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )

    volume_bars = go.Bar(
        x=df["timestamp"],
        y=df["volume"],
        name="Volume",
        marker_color="rgba(128,128,128,0.3)",
        yaxis="y2",
    )

    fig.add_trace(candlestick)
    fig.add_trace(volume_bars)

    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_dark",
        height=600,
    )

    return fig
