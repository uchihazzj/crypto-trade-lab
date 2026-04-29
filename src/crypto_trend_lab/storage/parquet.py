"""Local Parquet storage for OHLCV data.

Uses symbol-safe directory names and deterministic file paths.
Raw data is never silently overwritten without explicit documentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from src.crypto_trend_lab.utils.helpers import OHLCV_COLUMNS, symbol_to_safe


BASE_DIR = Path(__file__).resolve().parents[3]  # project root
DATA_DIR = BASE_DIR / "data"


def _build_path(
    exchange: str,
    symbol: str,
    timeframe: str,
    layer: str = "raw",
    suffix: str = "ohlcv",
) -> Path:
    """Build a deterministic Parquet path for OHLCV data.

    Pattern: data/{layer}/ohlcv/exchange={exchange}/symbol={safe_symbol}/timeframe={timeframe}/{suffix}.parquet
    """
    safe_symbol = symbol_to_safe(symbol)
    return (
        DATA_DIR
        / layer
        / "ohlcv"
        / f"exchange={exchange}"
        / f"symbol={safe_symbol}"
        / f"timeframe={timeframe}"
        / f"{suffix}.parquet"
    )


def save_ohlcv_parquet(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    layer: str = "raw",
) -> Path:
    """Save an OHLCV DataFrame to local Parquet storage.

    Parameters
    ----------
    df : pd.DataFrame
        Must conform to the stable OHLCV schema.
    exchange : str
        Exchange ID (e.g. 'binance').
    symbol : str
        Trading symbol (e.g. 'BTC/USDT').
    timeframe : str
        OHLCV timeframe (e.g. '1h').
    layer : str
        Data layer: 'raw', 'processed', 'features', or 'predictions'.
        Default is 'raw'.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    path = _build_path(exchange, symbol, timeframe, layer=layer)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and layer == "raw":
        logger.info(
            f"Raw data already exists at {path}. "
            f"Use layer='processed' or delete the file to overwrite."
        )

    df_sorted = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df_sorted.to_parquet(path, index=False)
    logger.info(f"Saved {len(df_sorted)} rows to {path}")
    return path


def load_ohlcv_parquet(
    exchange: str,
    symbol: str,
    timeframe: str,
    layer: str = "raw",
) -> pd.DataFrame:
    """Load an OHLCV DataFrame from local Parquet storage.

    Returns an empty DataFrame with the stable schema if the file does not exist.
    """
    path = _build_path(exchange, symbol, timeframe, layer=layer)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def _build_features_path(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """Build a deterministic Parquet path for feature data.

    Pattern: data/features/exchange={exchange}/symbol={safe_symbol}/timeframe={timeframe}/features.parquet
    """
    safe_symbol = symbol_to_safe(symbol)
    return (
        DATA_DIR
        / "features"
        / f"exchange={exchange}"
        / f"symbol={safe_symbol}"
        / f"timeframe={timeframe}"
        / "features.parquet"
    )


def save_features_parquet(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """Save a feature DataFrame to local Parquet storage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (OHLCV columns + technical features + targets).
    exchange : str
        Exchange ID (e.g. 'binance').
    symbol : str
        Trading symbol (e.g. 'BTC/USDT').
    timeframe : str
        OHLCV timeframe (e.g. '1h').

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    path = _build_features_path(exchange, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df_sorted.to_parquet(path, index=False)
    logger.info(f"Saved {len(df_sorted)} feature rows to {path}")
    return path


def load_features_parquet(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Load a feature DataFrame from local Parquet storage.

    Returns an empty DataFrame if the file does not exist.
    """
    path = _build_features_path(exchange, symbol, timeframe)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Loaded {len(df)} feature rows from {path}")
    return df
