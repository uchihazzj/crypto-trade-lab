"""Full evaluation report across all task types, horizons, and models.

Runs every supported combination on a feature DataFrame and returns a
consolidated metrics table plus a skip log for anything that failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto_trend_lab.evaluation.report import compare_baselines_and_models
from src.crypto_trend_lab.models.dataset import get_default_feature_columns, get_target_column
from src.crypto_trend_lab.models.tabular import _HAS_CATBOOST, _HAS_LIGHTGBM, _HAS_XGBOOST

_SUPPORTED_HORIZONS = (1, 4, 24)
_SUPPORTED_TASK_TYPES = ("regression", "classification")

# Columns expected in the consolidated metrics DataFrame.
_METRICS_COLUMNS = [
    "task_type", "horizon", "target_column", "model_name",
    # regression
    "mae", "rmse", "directional_accuracy", "spearman_r",
    # classification
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc",
]


def _build_skipped_row(
    task_type: str, horizon: int, model_name: str, reason: str,
) -> dict:
    return {
        "task_type": task_type,
        "horizon": horizon,
        "model_name": model_name,
        "reason": reason,
    }


def run_full_evaluation_report(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = _SUPPORTED_HORIZONS,
    task_types: tuple[str, ...] = _SUPPORTED_TASK_TYPES,
    test_size: int | float = 0.2,
    feature_columns: list[str] | None = None,
    include_trees: bool = False,
) -> dict:
    """Run baselines and tabular models for every task × horizon combination.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from ``build_features()``.
    horizons : tuple of int
        Forecast horizons to evaluate.
    task_types : tuple of str
        Task types: ``"regression"``, ``"classification"``.
    test_size : int or float
        Test hold-out size passed to ``chronological_train_test_split``.
    feature_columns : list[str] or None
        Feature columns. If None, auto-detected from *df*.
    include_trees : bool
        If True, include tree ensembles and optional external models
        (XGBoost, CatBoost if installed). Default False for speed.

    Returns
    -------
    dict
        Keys: metrics_df (pd.DataFrame), skipped_df (pd.DataFrame),
        summary (dict with dataset info and run counts).
    """
    all_rows: list[dict] = []
    skipped: list[dict] = []

    if feature_columns is None:
        feature_columns = get_default_feature_columns(df)

    total_combinations = 0
    successful = 0

    for task_type in task_types:
        for horizon in horizons:
            target_column = get_target_column(task_type, horizon)

            # --- Check target column exists ---
            if target_column not in df.columns:
                reason = (
                    f"Target column {target_column!r} not in DataFrame. "
                    f"Run build_features() first."
                )
                for model_name in _expected_model_names(task_type, include_trees):
                    skipped.append(
                        _build_skipped_row(task_type, horizon, model_name, reason)
                    )
                continue

            total_combinations += 1

            try:
                result = compare_baselines_and_models(
                    df,
                    task_type=task_type,
                    horizon=horizon,
                    test_size=test_size,
                    feature_columns=feature_columns,
                    include_tabular=True,
                    include_trees=include_trees,
                )
            except ValueError as exc:
                # Missing target or insufficient data — skip all models
                reason = str(exc)
                for model_name in _expected_model_names(task_type, include_trees):
                    skipped.append(
                        _build_skipped_row(task_type, horizon, model_name, reason)
                    )
                continue
            except ImportError:
                # LightGBM unavailable (shouldn't happen since _HAS_LIGHTGBM guards it,
                # but catch defensively)
                reason = "LightGBM is not installed"
                # The baselines will have run; tabular models are skipped
                # Re-run without tabular to get baseline results
                try:
                    result = compare_baselines_and_models(
                        df,
                        task_type=task_type,
                        horizon=horizon,
                        test_size=test_size,
                        feature_columns=feature_columns,
                        include_tabular=False,
                        include_trees=False,
                    )
                except Exception:
                    for model_name in _expected_model_names(task_type, include_trees):
                        skipped.append(
                            _build_skipped_row(task_type, horizon, model_name, reason)
                        )
                    continue

                for model_name in _expected_tabular_names(task_type, include_trees):
                    skipped.append(
                        _build_skipped_row(task_type, horizon, model_name, reason)
                    )

            successful += 1

            # --- Collect per-model skipped entries from this run ---
            for s in result.get("skipped", []):
                skipped.append(s)

            # --- Collect results ---
            metrics_df = result["metrics_table"]
            for _, row in metrics_df.iterrows():
                combined = {
                    "task_type": task_type,
                    "horizon": horizon,
                    "target_column": target_column,
                    "model_name": row["model_name"],
                }
                # Merge all metrics
                for col in row.index:
                    if col != "model_name":
                        combined[col] = row[col]
                all_rows.append(combined)

    # Build output DataFrames
    metrics_df = pd.DataFrame(all_rows, columns=_METRICS_COLUMNS) if all_rows else pd.DataFrame(columns=_METRICS_COLUMNS)
    skipped_df = pd.DataFrame(skipped, columns=["task_type", "horizon", "model_name", "reason"]) if skipped else pd.DataFrame(columns=["task_type", "horizon", "model_name", "reason"])

    # Dataset summary
    t_min = df["timestamp"].min() if "timestamp" in df.columns else None
    t_max = df["timestamp"].max() if "timestamp" in df.columns else None

    summary = {
        "exchange": df["exchange"].iloc[0] if "exchange" in df.columns else None,
        "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else None,
        "timeframe": df["timeframe"].iloc[0] if "timeframe" in df.columns else None,
        "row_count": len(df),
        "min_timestamp": t_min,
        "max_timestamp": t_max,
        "feature_count": len(feature_columns),
        "total_combinations": total_combinations,
        "successful": successful,
        "skipped": len(skipped),
        "lightgbm_available": _HAS_LIGHTGBM,
        "xgboost_available": _HAS_XGBOOST,
        "catboost_available": _HAS_CATBOOST,
        "include_trees": include_trees,
    }

    return {
        "metrics_df": metrics_df,
        "skipped_df": skipped_df,
        "summary": summary,
    }


def _expected_model_names(task_type: str, include_trees: bool = False) -> list[str]:
    """Model names that would be run for a task type (baselines + tabular)."""
    if task_type == "regression":
        names = ["Zero Return", "Last Return", "Moving Average",
                 "Historical Mean Return", "Ridge", "ElasticNet"]
    else:
        names = ["Momentum Direction", "Majority Class", "Logistic Regression"]
    if _HAS_LIGHTGBM:
        names.append("LightGBM")
    if include_trees:
        if task_type == "regression":
            names.extend(["Random Forest", "Extra Trees", "HistGradientBoosting"])
        else:
            names.extend(["Random Forest", "Extra Trees", "HistGradientBoosting"])
        if _HAS_XGBOOST:
            names.append("XGBoost")
        if _HAS_CATBOOST:
            names.append("CatBoost")
    return names


def _expected_tabular_names(task_type: str, include_trees: bool = False) -> list[str]:
    """Only the tabular model names for a task type."""
    names = ["Ridge", "ElasticNet"] if task_type == "regression" else ["Logistic Regression"]
    if _HAS_LIGHTGBM:
        names.append("LightGBM")
    if include_trees:
        if task_type == "regression":
            names.extend(["Random Forest", "Extra Trees", "HistGradientBoosting"])
        else:
            names.extend(["Random Forest", "Extra Trees", "HistGradientBoosting"])
        if _HAS_XGBOOST:
            names.append("XGBoost")
        if _HAS_CATBOOST:
            names.append("CatBoost")
    return names


def build_best_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build a best-model-per-metric table from the full metrics DataFrame.

    For each task_type × horizon, finds the best model per relevant metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full evaluation metrics table from ``run_full_evaluation_report``.

    Returns
    -------
    pd.DataFrame
        Columns: task_type, horizon,
        best_mae_model, best_mae, best_rmse_model, best_rmse,
        best_dir_acc_model, best_dir_acc,
        best_bal_acc_model, best_bal_acc, best_f1_model, best_f1,
        best_auc_model, best_auc.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    rows = []
    for tt in ("regression", "classification"):
        tt_df = metrics_df[metrics_df["task_type"] == tt]
        if tt_df.empty:
            continue
        for h in sorted(tt_df["horizon"].unique()):
            h_df = tt_df[tt_df["horizon"] == h]
            row: dict = {"task_type": tt, "horizon": h}

            if tt == "regression":
                for metric in ["mae", "rmse", "directional_accuracy"]:
                    col = h_df[metric]
                    if col.notna().any():
                        if metric == "directional_accuracy":
                            idx = col.idxmax()
                        else:
                            idx = col.idxmin()
                        if pd.notna(idx):
                            row[f"best_{metric}_model"] = h_df.loc[idx, "model_name"]
                            row[f"best_{metric}"] = h_df.loc[idx, metric]
            else:
                for metric in ["balanced_accuracy", "f1", "auc"]:
                    if metric in h_df.columns and h_df[metric].notna().any():
                        idx = h_df[metric].idxmax()
                        if pd.notna(idx):
                            row[f"best_{metric}_model"] = h_df.loc[idx, "model_name"]
                            row[f"best_{metric}"] = h_df.loc[idx, metric]

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def generate_report_analysis(
    metrics_df: pd.DataFrame,
    summary: dict,
    skipped_df: pd.DataFrame | None = None,
) -> str:
    """Generate a cautious, rule-based textual analysis of the evaluation report.

    Uses only the data in *metrics_df* and *summary*. No LLM calls.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full evaluation metrics.
    summary : dict
        Dataset summary from ``run_full_evaluation_report``.
    skipped_df : pd.DataFrame or None
        Skipped combinations log.

    Returns
    -------
    str
        Markdown-formatted analysis text.
    """
    lines: list[str] = []
    lines.append("## Analysis Notes\n")

    # --- Dataset overview ---
    row_count = summary.get("row_count", 0)
    timeframe = summary.get("timeframe", "unknown")
    t_min = summary.get("min_timestamp")
    t_max = summary.get("max_timestamp")

    lines.append("### Dataset\n")
    lines.append(f"- **Rows**: {row_count}")
    lines.append(f"- **Timeframe**: {timeframe}")

    if t_min is not None and t_max is not None:
        duration = pd.Timestamp(t_max) - pd.Timestamp(t_min)
        lines.append(f"- **Date range**: {t_min} → {t_max} ({duration})")

    # Dataset size assessment
    if row_count < 1000:
        lines.append(
            f"- **Warning**: Only {row_count} rows — this dataset is suitable for "
            "UI and pipeline testing but likely too small for reliable model evaluation."
        )
    elif timeframe == "1h" and row_count < 3000:
        lines.append(
            f"- **Note**: {row_count} 1h bars — a larger dataset would strengthen "
            "conclusions."
        )
    elif timeframe in ("4h", "1d") and row_count < 1000:
        lines.append(
            f"- **Note**: {row_count} {timeframe} bars — limited for model evaluation."
        )

    lines.append("")

    # --- Test range ---
    test_pct = summary.get("test_size_pct", None)
    if test_pct is not None:
        lines.append(f"**Test split**: held out the most recent {test_pct:.0f}% of data.\n")

    # --- Skipped ---
    skipped_count = summary.get("skipped", 0)
    if skipped_count > 0:
        lines.append(f"### Skipped Combinations ({skipped_count})\n")
        lines.append(
            f"{skipped_count} model/horizon combination(s) were skipped. "
            "Common reasons include missing target columns, insufficient valid rows "
            "after NaN removal, or LightGBM not being installed. "
            "See the Skipped Combinations table for details.\n"
        )

    # --- Model performance analysis ---
    if metrics_df.empty:
        lines.append(
            "### Results\n\n"
            "No metrics were produced. All combinations were skipped. "
            "Run `build_features()` first, or fetch more data to ensure "
            "sufficient valid rows per target horizon.\n"
        )
        return "\n".join(lines)

    lines.append("### Model Performance\n")

    # Check if any tabular model beats ALL baselines
    baseline_names = {"Zero Return", "Last Return", "Moving Average",
                      "Momentum Direction", "Majority Class"}

    for tt in ("regression", "classification"):
        subset = metrics_df[metrics_df["task_type"] == tt]
        if subset.empty:
            continue

        lines.append(f"#### {tt.capitalize()}\n")

        for h in sorted(subset["horizon"].unique()):
            h_df = subset[subset["horizon"] == h]
            if h_df.empty:
                continue

            lines.append(f"**Horizon {h}**:")
            model_names = h_df["model_name"].tolist()
            baselines_in = [n for n in model_names if n in baseline_names]
            tabular_in = [n for n in model_names if n not in baseline_names]

            # Compare best tabular vs best baseline
            if tt == "regression" and "mae" in h_df.columns:
                best_baseline_mae = None
                best_tabular_mae = None
                for _, row in h_df.iterrows():
                    if pd.isna(row.get("mae")):
                        continue
                    if row["model_name"] in baseline_names:
                        if best_baseline_mae is None or row["mae"] < best_baseline_mae:
                            best_baseline_mae = row["mae"]
                    else:
                        if best_tabular_mae is None or row["mae"] < best_tabular_mae:
                            best_tabular_mae = row["mae"]

                if best_tabular_mae is not None and best_baseline_mae is not None:
                    if best_tabular_mae < best_baseline_mae:
                        lines.append(
                            f"  - A tabular model achieves lower MAE than baselines "
                            f"under this test split."
                        )
                    else:
                        lines.append(
                            f"  - No tabular model beats the best baseline MAE "
                            f"under this test split."
                        )
                # Directional accuracy
                if "directional_accuracy" in h_df.columns:
                    da_vals = h_df["directional_accuracy"].dropna()
                    if not da_vals.empty:
                        best_da = da_vals.max()
                        lines.append(f"  - Best directional accuracy: {best_da:.3f}")

            elif tt == "classification":
                if "balanced_accuracy" in h_df.columns:
                    ba_vals = h_df["balanced_accuracy"].dropna()
                    if not ba_vals.empty:
                        best_ba = ba_vals.max()
                        best_ba_model = h_df.loc[ba_vals.idxmax(), "model_name"]
                        if best_ba > 0.55:
                            lines.append(
                                f"  - Best balanced accuracy: {best_ba:.3f} "
                                f"({best_ba_model}) — above chance."
                            )
                        else:
                            lines.append(
                                f"  - Best balanced accuracy: {best_ba:.3f} "
                                f"({best_ba_model}) — near or below chance level."
                            )

            lines.append("")

    # --- Consistency across horizons ---
    lines.append("### Horizon Comparison\n")
    success_horizons = set()
    for h in (1, 4, 24):
        h_df = metrics_df[metrics_df["horizon"] == h]
        if not h_df.empty:
            success_horizons.add(h)

    if len(success_horizons) >= 2:
        # Check if metrics degrade with longer horizons
        for tt in ("regression", "classification"):
            tt_df = metrics_df[metrics_df["task_type"] == tt]
            if tt_df.empty or len(tt_df["horizon"].unique()) < 2:
                continue
            if tt == "regression" and "mae" in tt_df.columns:
                maes_by_h = {}
                for h in sorted(success_horizons):
                    h_df = tt_df[tt_df["horizon"] == h]
                    maes = h_df["mae"].dropna()
                    if not maes.empty:
                        maes_by_h[h] = maes.min()
                if len(maes_by_h) >= 2:
                    sorted_h = sorted(maes_by_h.keys())
                    if all(
                        maes_by_h[sorted_h[i]] <= maes_by_h[sorted_h[i + 1]]
                        for i in range(len(sorted_h) - 1)
                    ):
                        lines.append(
                            "- Regression MAE consistently increases with horizon length "
                            "— longer-horizon predictions are harder, as expected."
                        )
                    else:
                        lines.append(
                            "- Regression MAE does not increase monotonically with "
                            "horizon — results may be noisy given the dataset size."
                        )
            elif tt == "classification" and "balanced_accuracy" in tt_df.columns:
                bas_by_h = {}
                for h in sorted(success_horizons):
                    h_df = tt_df[tt_df["horizon"] == h]
                    bas = h_df["balanced_accuracy"].dropna()
                    if not bas.empty:
                        bas_by_h[h] = bas.max()
                if len(bas_by_h) >= 2:
                    sorted_h = sorted(bas_by_h.keys())
                    if all(bas_by_h[h] <= 0.55 for h in sorted_h):
                        lines.append(
                            "- Classification balanced accuracy is near or below "
                            "chance for all horizons — directional prediction is "
                            "difficult with these features."
                        )

    lines.append("")

    # --- Cautions ---
    lines.append("### Cautions\n")
    lines.append(
        "- All results are **historical evaluations** on a fixed chronological split. "
        "They do not imply future profitability."
    )
    lines.append(
        "- Crypto markets are **non-stationary** — patterns in past data may not "
        "persist in future periods."
    )
    lines.append(
        "- **Model comparison is metric-specific** — a model that ranks best on MAE "
        "may not be best on directional accuracy."
    )
    lines.append(
        "- Results are **preliminary** and should be validated with additional data, "
        "different market regimes, and alternative feature sets."
    )
    if row_count < 3000:
        lines.append(
            "- This dataset is **small** for model evaluation. Larger datasets "
            "would provide more robust conclusions."
        )
    lines.append(
        "- These are **experimental research signals**, not investment advice."
    )

    return "\n".join(lines)
