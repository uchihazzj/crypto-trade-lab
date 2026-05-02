"""CCXT public market data ingestion.

Uses only read-only public endpoints. Private trading APIs are not used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger

from src.crypto_trend_lab.utils.helpers import OHLCV_COLUMNS, normalize_ohlcv


def _get_exchange(exchange_id: str) -> ccxt.Exchange:
    """Return a ccxt exchange instance with rate limiting enabled."""
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(
            f"Exchange {exchange_id!r} not found in ccxt. "
            f"Check the exchange ID spelling."
        )
    ex: ccxt.Exchange = exchange_class({"enableRateLimit": True})
    ex.load_markets()
    return ex


def fetch_ticker(exchange_id: str, symbol: str) -> dict:
    """Fetch the latest ticker for a symbol from a public exchange.

    Returns the raw ccxt ticker dict.
    """
    ex = _get_exchange(exchange_id)
    logger.info(f"Fetching ticker {exchange_id=} {symbol=}")
    return ex.fetch_ticker(symbol)


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch recent OHLCV bars from a public exchange.

    Returns a DataFrame with the stable OHLCV schema. Timestamps are UTC.
    """
    ex = _get_exchange(exchange_id)
    logger.info(f"Fetching OHLCV {exchange_id=} {symbol=} {timeframe=} limit={limit}")
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return normalize_ohlcv(df, exchange=exchange_id, symbol=symbol, timeframe=timeframe)


def fetch_ohlcv_range(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars with pagination between *start* and *end*.

    Uses CCXT pagination (since parameter) to fetch all available bars in the
    requested range.  Combines pages, deduplicates, and returns the stable schema.

    Parameters
    ----------
    exchange_id : str
        CCXT exchange ID (e.g. 'binance').
    symbol : str
        Trading symbol (e.g. 'BTC/USDT').
    timeframe : str
        OHLCV timeframe (e.g. '1h', '1d').
    start : datetime, optional
        Start of the range (inclusive). If None, no lower bound.
    end : datetime, optional
        End of the range (inclusive). If None, fetches up to the present.
    max_pages : int
        Maximum number of pages to fetch. Default 100 (~100,000 bars).
        Prevents infinite loops if the exchange returns stale timestamps.

    Returns
    -------
    pd.DataFrame
        Combined OHLCV data with stable schema.

    Raises
    ------
    RuntimeError
        If *max_pages* is exceeded.
    """
    ex = _get_exchange(exchange_id)
    since_ms: Optional[int] = None
    if start is not None:
        since_ms = int(start.timestamp() * 1000)

    logger.info(
        f"Fetching OHLCV range {exchange_id=} {symbol=} {timeframe=} "
        f"start={start} end={end}"
    )

    all_frames: list[pd.DataFrame] = []
    page_count = 0

    while True:
        page_count += 1
        if page_count > max_pages:
            raise RuntimeError(
                f"Exceeded maximum page count ({max_pages}) fetching "
                f"{exchange_id=} {symbol=} {timeframe=}. "
                f"The exchange may be returning stale timestamps."
            )
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not raw:
            logger.info(f"No more data after {page_count - 1} pages")
            break

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Filter to end bound
        if end is not None:
            end_ts = pd.Timestamp(end).tz_localize("utc") if end.tzinfo is None else pd.Timestamp(end)
            df = df[df["timestamp"] <= end_ts]

        all_frames.append(df)

        # Advance pagination cursor past the last bar
        last_ts_val = df["timestamp"].max()
        if pd.isna(last_ts_val):
            logger.warning("Last timestamp is NaT — stopping pagination")
            break
        last_ts = int(last_ts_val.timestamp() * 1000)
        since_ms = last_ts + 1

        # Stop if fewer than 1000 bars returned (reached end of available data)
        if len(raw) < 1000:
            logger.info(f"Reached end of available data after {page_count} pages")
            break

        # Stop if we've passed the end bound
        if end is not None and last_ts >= int(end.timestamp() * 1000):
            break

    if not all_frames:
        logger.warning(f"No data returned for {exchange_id=} {symbol=} {timeframe=}")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    combined = pd.concat(all_frames, ignore_index=True)
    return normalize_ohlcv(
        combined,
        exchange=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
    )
