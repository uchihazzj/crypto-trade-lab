# Crypto Trend Lab

Research-oriented crypto market data analysis and trend forecasting tool.

**This is NOT a trading bot.** Crypto Trend Lab is designed for offline research:
historical data retrieval, feature exploration, experimental trend prediction, and
walk-forward model evaluation. All model outputs are experimental research results
and do not constitute financial advice.

## Scope

- Fetch public OHLCV data via CCXT (read-only).
- Store data locally as Parquet files.
- Validate data quality (missing bars, duplicates, schema checks).
- Build technical indicators and prediction targets (no future leakage).
- Visualize data with Plotly candlestick charts and indicator line charts in a local Streamlit dashboard.

## Non-Trading Boundary

This tool does **not** implement:
- Real trading, order placement, or position management.
- Futures, leverage, or margin logic.
- Exchange private API keys or account management.
- Buy/sell recommendations or financial advice.

## Installation

Requires Python 3.11 and the `crypto-trend-lab` conda environment.

```bash
git clone <repo-url>
cd crypto-trend-lab
conda env create -f environment.yml   # if environment.yml is available
# or use the existing crypto-trend-lab conda environment
conda run -n crypto-trend-lab python -m pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` and set your API keys (optional for Milestone 1).
2. Edit `config/symbols.yaml` to adjust exchanges, symbols, and timeframes.

## How to Run

```bash
conda run -n crypto-trend-lab streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```text
crypto-trend-lab/
  app.py              # Streamlit dashboard
  config/
    symbols.yaml      # Exchange and symbol configuration
  src/crypto_trend_lab/
    data_ingestion/   # CCXT public market data fetching
    storage/          # Local Parquet read/write
    validation/       # Data quality checks
    visualization/    # Plotly charts
    features/         # Technical indicators and prediction targets
    utils/            # Shared helpers
  tests/              # Test suite
  data/               # Local data directory (git-ignored)
    raw/              # Raw OHLCV data
    processed/        # Processed OHLCV data
    features/         # Feature DataFrames (future)
    predictions/      # Model prediction outputs (future)
```

## Running Tests

```bash
conda run -n crypto-trend-lab python -m pytest
```

## Milestone 1 Scope

- [x] CCXT OHLCV ingestion for BTC/USDT and ETH/USDT.
- [x] Local Parquet storage with deterministic paths.
- [x] Data quality checks (schema, duplicates, missing bars, nulls).
- [x] Streamlit dashboard with candlestick charts and quality stats.
- [x] Unit tests for core utilities and validation.

## Milestone 2 Scope

- [x] Technical feature engineering (20 indicators, no future leakage).
- [x] Prediction target construction (6 forward-looking targets).
- [x] Feature pipeline with schema validation and immutability.
- [x] Feature Parquet storage with deterministic paths.
- [x] Feature Preview tab in Streamlit dashboard.
- [x] Unit tests for features, targets, pipeline, and storage.

## Feature Engineering

### Technical Features

All indicators use only current and past observations. No future values
are used in feature construction.

- **Returns**: log_return_1, return_1/3/6/12/24
- **Volatility**: rolling_vol_24/72/168 (std of log returns)
- **Volume**: volume_change_24
- **Moving Averages**: ma_7, ma_25, ma_99, ema_12, ema_26
- **Oscillators**: rsi_14, macd, macd_signal, bollinger_width, atr_14

### Prediction Targets

Targets use future close prices and must never be used as model inputs:

- **target_return_h** = ln(close[t+h] / close[t])  for h in {1, 4, 24}
- **target_direction_h** = 1 if target_return_h > 0 else 0

Rows with unavailable future targets remain NaN. The final *h* rows
for each target_return_h are NaN.

### Future Leakage Prevention

- Technical indicators use `.shift()`, `.rolling()`, and `.ewm()` without
  looking ahead.
- Targets use `.shift(-h)` to reference future values but are excluded
  from `get_model_input_columns()`.
- The feature pipeline never mutates the input DataFrame.

### Feature Preview Tab

The Streamlit dashboard includes a Feature Preview tab that allows you to:
1. Select a data source (fetched, local OHLCV, or saved features).
2. Run the feature pipeline.
3. Preview the feature table.
4. View column lists (OHLCV, technical features, targets).
5. Inspect NaN counts per column.
6. Save the feature table to local Parquet.
7. View line charts for close, ma_25, rsi_14, macd, and bollinger_width.

No model training is implemented yet. Model outputs and trading logic
remain out of scope.

## Data Sources (Read-Only)

- CCXT public market data (Binance and other exchanges).
- Future milestones will add CoinGecko, FRED, yfinance, and others.

---

*Crypto markets are noisy, non-stationary, and high-risk. All outputs from this tool
are experimental research signals, not investment advice.*
