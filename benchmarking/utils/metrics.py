# benchmarking/utils/metrics.py
"""
Shared Evaluation Metrics for All Benchmark Models.

This module is intentionally kept independent of any specific model type so
that the same metric computation logic is reused across classical ML,
statistical, and deep learning baselines, as well as the proposed NiOA-DRNN.

Revision notes (v2)
--------------------
A `log_transform` parameter has been added to `compute_metrics` and to
`save_benchmark_results`.  When set to True, np.expm1 is applied to both
y_true and y_pred before computing any metric, thereby restoring original
kWh units after the log1p target transformation introduced in v2 of the
preprocessing pipeline.

This ensures that all benchmark models — regardless of architecture — are
evaluated in the same physical unit space, making cross-model comparisons
fair and directly interpretable.

The default is `False` so that existing callers that have already inverted
the transform themselves are not affected.  The benchmark notebooks (02, 03,
05, 06) should pass `log_transform=True` when calling `save_benchmark_results`.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true        : np.ndarray,
    y_pred        : np.ndarray,
    log_transform : bool = False,
) -> dict:
    """
    Compute MAE, RMSE, R², and sMAPE.

    Parameters
    ----------
    y_true        : np.ndarray   Ground-truth ΔE values.
    y_pred        : np.ndarray   Predicted ΔE values.
    log_transform : bool         If True, apply np.expm1 to both arrays before
                                 computing metrics.  Use when the pipeline has
                                 applied log1p to the target variable.

    Returns
    -------
    dict with keys: 'MAE', 'RMSE', 'R2', 'sMAPE'
    """
    # ------------------------------------------------------------------
    # Inverse transform  [NEW in v2]
    # ------------------------------------------------------------------
    if log_transform:
        y_true = np.expm1(np.clip(y_true.astype(np.float64), -10.0, 30.0))
        y_pred = np.expm1(np.clip(y_pred.astype(np.float64), -10.0, 30.0))

    # Clip negative predictions to zero (energy increments cannot be negative)
    y_pred = np.maximum(y_pred, 0.0)

    eps   = 1e-8
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))
    smape = float(
        np.mean(
            np.abs(y_true - y_pred)
            / ((np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps)
        ) * 100
    )
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE": smape}


def save_benchmark_results(
    model_name    : str,
    horizon       : int,
    y_true        : np.ndarray,
    y_pred        : np.ndarray,
    benchmark_root: str,
    log_transform : bool = False,
    extra_meta    : dict = None,
) -> dict:
    """
    Persist benchmark results in a structured, self-documenting format.

    Parameters
    ----------
    model_name     : str         Short model identifier (e.g. 'xgboost').
    horizon        : int         Forecast horizon in seconds.
    y_true         : np.ndarray  Ground-truth array (in log1p space if the
                                 pipeline applied the transform).
    y_pred         : np.ndarray  Predicted array (in log1p space if the model
                                 was trained on log1p targets).
    benchmark_root : str         Top-level benchmark results directory.
    log_transform  : bool        If True, apply np.expm1 before computing
                                 metrics.  Saved arrays are also converted to
                                 original kWh space for interpretability.
    extra_meta     : dict        Optional additional metadata to store.

    Returns
    -------
    dict   Computed metrics (in original kWh units when log_transform=True).
    """
    save_dir = os.path.join(benchmark_root, f"horizon_{horizon}", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Compute metrics (with inverse transform if required)
    metrics = compute_metrics(y_true, y_pred, log_transform=log_transform)

    # ------------------------------------------------------------------
    # Save arrays in original kWh space for interpretability
    # ------------------------------------------------------------------
    if log_transform:
        y_true_save = np.expm1(
            np.clip(y_true.astype(np.float64), -10.0, 30.0)
        )
        y_pred_save = np.maximum(
            np.expm1(np.clip(y_pred.astype(np.float64), -10.0, 30.0)),
            0.0,
        )
    else:
        y_true_save = y_true
        y_pred_save = np.maximum(y_pred, 0.0)

    np.save(os.path.join(save_dir, "y_test_pred.npy"), y_pred_save)
    np.save(os.path.join(save_dir, "y_test_true.npy"), y_true_save)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    summary = {
        "model"          : model_name,
        "horizon"        : horizon,
        "metrics"        : metrics,
        "log_transform"  : log_transform,
    }
    if extra_meta:
        summary["meta"] = extra_meta

    with open(os.path.join(save_dir, "result_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return metrics


def build_comparison_table(benchmark_root: str, horizon: int) -> pd.DataFrame:
    """
    Aggregate saved metrics from all benchmark models for a given horizon
    into a single comparison DataFrame.

    Parameters
    ----------
    benchmark_root : str   Top-level benchmark results directory.
    horizon        : int   Forecast horizon in seconds.

    Returns
    -------
    pd.DataFrame   One row per model, columns: Model, MAE, RMSE, R², sMAPE.
    """
    horizon_dir = os.path.join(benchmark_root, f"horizon_{horizon}")
    if not os.path.isdir(horizon_dir):
        raise FileNotFoundError(
            f"No benchmark results found at '{horizon_dir}'."
        )

    rows = []
    for model_name in sorted(os.listdir(horizon_dir)):
        metrics_path = os.path.join(horizon_dir, model_name, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            rows.append({"Model": model_name, **m})

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
