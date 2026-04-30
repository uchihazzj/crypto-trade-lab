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
    evaluation/       # Chronological splits, metrics, model comparison
    models/           # Baseline and tabular models
    utils/            # Shared helpers
  tests/              # Test suite
  data/               # Local data directory (git-ignored)
    raw/              # Raw OHLCV data
    processed/        # Processed OHLCV data
    features/         # Feature DataFrames
    predictions/      # Model prediction outputs
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

## Milestone 3 Scope

- [x] Chronological train/test split and walk-forward split (no shuffling).
- [x] Regression metrics (MAE, RMSE, directional accuracy, Spearman).
- [x] Classification metrics (accuracy, balanced accuracy, precision, recall, F1, AUC).
- [x] Baseline models: zero return, last return, moving average, momentum direction, majority class.
- [x] Tabular models: Ridge regression, Logistic regression, LightGBM (if installed).
- [x] Modeling dataset helpers (feature column selection, target column lookup, NaN-aware preparation).
- [x] Model evaluation workflow with chronological splitting.
- [x] Prediction Parquet storage with deterministic paths.
- [x] Model Evaluation tab in Streamlit dashboard.
- [x] Unit tests for splits, metrics, baselines, dataset, and evaluation.
- [ ] Macro indicators, sequence models, and deep learning (not yet implemented).

## Model Evaluation

### Baselines

Simple models that any useful predictive model must beat:

- **Regression**: Zero Return (always 0), Last Return (most recent), Moving Average (training mean).
- **Classification**: Momentum Direction (sign of last return), Majority Class (training mode).

### Tabular Models

- **Ridge Regression** for `target_return_h` — scaled features, L2 penalty.
- **Logistic Regression** for `target_direction_h` — scaled features, balanced class weights.
- **LightGBM Regressor / Classifier** — gradient-boosted trees (if installed).

All models use scikit-learn Pipelines where applicable. Scalers are fit only on
the training set and applied to the test set — no future leakage.

### Chronological Validation

All evaluation uses chronological splits:

- `chronological_train_test_split`: train on earlier data, test on later data.
- `walk_forward_split`: expanding or rolling window folds.

Random shuffling is never used. Future rows never leak into training.

### How to Use the Model Evaluation Tab

1. Build features in the Feature Preview tab (or load saved features).
2. Navigate to the Model Evaluation tab.
3. Select:
   - Task type: regression (predicts return magnitude) or classification (predicts direction).
   - Horizon: 1, 4, or 24 bars ahead.
   - Test size: percentage of data held out.
   - Whether to include tabular models.
4. Click Run Evaluation.
5. Inspect the metrics comparison table and y_true vs y_pred chart.
6. Optionally save predictions to local Parquet.

### Prediction Storage

Prediction results are saved to:

```
data/predictions/exchange=<exchange>/symbol=<symbol>/timeframe=<timeframe>/model=<model_name>/target=<target_column>/predictions.parquet
```

### Full Evaluation Report

The Model Evaluation tab includes a **Full Evaluation Report** section that runs
all supported task types, horizons, and models in one batch:

- **Task types**: regression (target_return_h), classification (target_direction_h)
- **Horizons**: 1, 4, 24
- **Models**: all regression baselines + Ridge + LightGBM; all classification baselines + Logistic Regression + LightGBM

The report produces:

- **Metrics table**: all task × horizon × model combinations with regression and
  classification metrics.
- **Best model summary**: best MAE/RMSE for regression, best accuracy/F1 for
  classification, per horizon.
- **Skipped combinations**: every failed combination is logged with a clear reason
  (missing target, insufficient data, LightGBM not installed).
- **CSV downloads**: metrics and skipped combinations can be downloaded.

Combinations that cannot run (e.g. missing target column, too few valid rows) are
skipped gracefully with a reason recorded — they never crash the whole report.

All results are experimental model evaluation outputs. They do not constitute
financial advice. Macro indicators, sequence models (Darts, NeuralForecast),
and deep learning are not implemented.

### Non-Trading Warning

All model outputs are **experimental research signals**. They are not investment
advice. Crypto markets are noisy, non-stationary, and high-risk. Past performance
in backtests does not guarantee future results.

## Milestone 3.5 Scope

- [x] Large recent-bars fetch with pagination (up to 50000 bars).
- [x] Data coverage summary showing requested vs actual rows.
- [x] Improved Full Evaluation Report with visualizations and cautious analysis.
- [x] Forward Forecast: single-point and forecast path.
- [x] Forecast path: sparse direct-horizon chart with actual OHLCV + predicted close.
- [x] Forecast Parquet storage with deterministic paths.
- [x] Forecast Path Chart timestamp generation fix (Timedelta arithmetic, expanded timeframe mapping).
- [x] OHLCV display aggregation (preserves OHLCV semantics, no random sampling, no averaging).
- [x] Chart time range controls and max candles selector.
- [x] Full dataset protection: df_chart/df_preview are rendering-only, never used for modeling.
- [x] Unit tests for pagination, forecast path, timestamp generation, display aggregation, and data integrity.

### Large Fetch Behavior

When requesting more than 1000 bars in Recent Bars mode, the app automatically
uses the paginated `fetch_ohlcv_range()` function instead of a single
`fetch_ohlcv()` call (which exchanges typically cap at ~1000 bars).

- A start timestamp is computed from `now - limit × timeframe_duration`.
- The paginated range fetch retrieves all available bars in the window.
- If more than *limit* bars are returned, only the most recent *limit* are kept.
- If fewer than requested are returned, a warning is shown with the shortfall.
- After fetching, the Data Coverage section shows: start, end, duration,
  requested vs actual rows, and data sizing warnings.

### Data Quality — Active Data Source

The Data Quality tab now clearly shows which data is being inspected:

- **Freshly fetched**: number of bars requested vs received.
- **Loaded from local Parquet**: file path context.
- Row count, symbol, and timeframe are displayed alongside quality metrics.

### Full Evaluation Report Improvements

The Full Evaluation Report now includes:

- **Metric bar charts**: grouped bars for each metric by model and horizon,
  separately for regression and classification.
- **Best Model Summary table**: for each task type × horizon, identifies the
  best model by MAE, RMSE, directional accuracy, balanced accuracy, F1, and
  AUC (where available). "Best" refers only to the listed metric under this
  historical test split.
- **Detailed Model View**: select a task type, horizon, and model to inspect
  y_true vs y_pred line chart, actual vs predicted scatter plot, residual plot
  (regression), confusion matrix (classification), and probability chart.
- **Cautious Textual Analysis**: rule-based analysis covering dataset size,
  model-vs-baseline comparison, horizon consistency, and limitations. Uses
  cautious language (no buy/sell/trading advice).
- **Downloads**: metrics CSV, skipped CSV, best model summary CSV, and
  analysis markdown.

### Forward Forecast vs Historical Evaluation

**Historical Evaluation** (Full Evaluation Report):
- Train on a chronological split of the data.
- Evaluate on a held-out test window.
- Compare models with metrics.
- Answers: "Did this model work on past unseen data?"

**Forward Forecast**:
- Train on ALL available labeled historical rows.
- Predict from the latest feature row (whose target is unknown — it's the
  future).
- Answers: "What does this model say about the immediate future?"
- The output is an **experimental forecast**, NOT a trading signal.

### Forward Forecast — Single Point

The single-point forecast:
1. Select task type (regression/classification), horizon (1/4/24), and model.
2. The model trains on all rows with valid features and known targets.
3. Predicts from the latest row with valid features (target may be NaN).
4. Regression output: predicted log return, latest close, implied future close,
   historical MAE/RMSE for context.
5. Classification output: predicted direction, probability (if available),
   historical balanced accuracy for context.
6. Results can be saved to `data/forecasts/...`.

### Forward Forecast — Forecast Path

The forecast path produces a visual chart of actual OHLCV candles (left)
and predicted future close-price points (right):

- **Method**: Sparse direct-horizon forecast using supported regression
  targets: `target_return_1`, `target_return_4`, `target_return_24`.
- **Path length**: user-selectable (6, 12, 24, 48, 72, 168 bars).
- Only horizons ≤ path_length are included in the forecast.
- Predicted future close at horizon *h*: `latest_close × exp(predicted_log_return_h)`.
- The future line starts from the latest observed close.
- A vertical marker separates observed data from the forecast.
- Intermediate points between predicted horizons are connected by
  interpolation for visualization — clearly labeled as such.
- Classification does not produce a close-price forecast path.

**Limitations**:
- Only 3 sparse horizons are directly predicted (1, 4, 24).
- Technical features are fixed at their latest observed values for all
  forecast steps — no feature dynamics are modeled.
- The forecast path is model-estimated and may be unstable with small datasets.
- This is NOT a trading signal or investment advice.

### Forecast Storage

Forecast results are saved to:

```
data/forecasts/exchange=<exchange>/symbol=<symbol>/timeframe=<timeframe>/model=<model_name>/target=<target_column>/forecast.parquet
```

### Forecast Path Chart — Timestamp Generation

Future forecast timestamps are generated using `pd.Timedelta` arithmetic,
never integer addition. The reusable `timeframe_to_timedelta(timeframe)`
helper converts any supported timeframe to a `pd.Timedelta`:

```python
future_ts = latest_ts + delta * step  # delta = pd.Timedelta, step = int
```

Supported timeframes: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`,
`6h`, `8h`, `12h`, `1d`, `1w`.

Direct `Timestamp + int` is never used — this avoids the pandas
`TypeError: Addition/subtraction of integers and integer-arrays with
Timestamp is no longer supported` in modern pandas versions.

### OHLCV Display Aggregation

Large datasets (e.g. 50000 bars) are slow to render in Plotly candlestick charts
and huge table previews. The display aggregation module solves this without
affecting the full dataset used for modeling.

**Key invariant**: The full dataset is always preserved for Parquet storage,
data quality checks, feature generation, model evaluation, and forecast fitting.

**Definitions**:
- `df_full` — the complete OHLCV DataFrame (used for storage, features, modeling).
- `df_view` — a time-range-filtered subset of `df_full` (UI only).
- `df_chart` — display-level aggregated OHLCV candles for Plotly rendering.
- `df_preview` — display-level table preview (tail rows only).

**Aggregation method**:
- Groups consecutive rows into approximately *target_bars* groups.
- Each output row preserves OHLCV semantics:
  - `open` = first open in the group
  - `high` = max high in the group
  - `low` = min low in the group
  - `close` = last close in the group
  - `volume` = sum of volume in the group
- Prices are **never averaged**.
- Random sampling and every-N-row sampling are deliberately not used because
  they destroy OHLCV relationships (a random open paired with a random close
  is not meaningful for candlestick analysis).

**Chart controls** (available in Fetch & Chart and Local Storage tabs):
- Time range: Full range, Last 7/30/90 days, or custom.
- Max candles: 500, 1000, 2000, 5000.
- Display summary shows: full rows, view rows, displayed candles, display mode
  (raw or aggregated), and approximate bars per displayed candle.

**Table preview**: Shows the latest 500 raw rows by default. The full dataset
row count is always shown. Aggregated overview is available as an alternative.

**Modeling protection**:
- `build_features()` always receives `df_full`.
- `check_ohlcv_quality()` always receives `df_full`.
- Model evaluation and forward forecast always use full features.
- `df_chart` and `df_preview` are rendering-only artifacts.

### Performance Recommendations for Large Datasets

- Fetch up to 50000 bars for thorough model evaluation (1h timeframe = ~5.7 years).
- Use chart time range filtering to inspect specific periods quickly.
- Limit displayed candles to 1000 for responsive chart interaction.
- Plotly performance degrades above ~5000 rendered candles.
- Feature generation and model evaluation throughput depend on row count, not
  chart aggregation — display settings do not affect modeling speed.

## Data Sources (Read-Only)

- CCXT public market data (Binance and other exchanges).
- Future milestones will add CoinGecko, FRED, yfinance, and others.

---

*Crypto markets are noisy, non-stationary, and high-risk. All outputs from this tool
are experimental research signals, not investment advice.*
