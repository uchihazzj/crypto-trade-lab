"""Plotly visualization utilities for crypto market data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


def add_datetime_vertical_marker(
    fig: go.Figure,
    x: pd.Timestamp | datetime,
    text: str | None = None,
    line_dash: str = "dot",
    line_color: str = "gray",
    line_width: int = 1,
) -> go.Figure:
    """Add a vertical marker line at datetime *x*.

    Uses ``add_shape`` + ``add_annotation`` instead of ``add_vline``
    because ``add_vline(annotation_text=...)`` triggers internal Plotly
    Timestamp + int arithmetic that raises ``TypeError`` in modern pandas.

    Parameters
    ----------
    fig : go.Figure
        The figure to add the marker to (mutated in place).
    x : pd.Timestamp or datetime
        The x-axis position of the vertical line.
    text : str or None
        Optional annotation text shown at the top of the line.
    line_dash : str
        Plotly dash style.
    line_color : str
        CSS color string.
    line_width : int
        Line width in pixels.

    Returns
    -------
    go.Figure
        The same figure (for chaining).
    """
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color=line_color,
            width=line_width,
            dash=line_dash,
        ),
    )

    if text is not None:
        fig.add_annotation(
            x=x,
            y=1,
            xref="x",
            yref="paper",
            text=text,
            showarrow=False,
            yanchor="bottom",
            font=dict(color=line_color, size=11),
        )

    return fig


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
