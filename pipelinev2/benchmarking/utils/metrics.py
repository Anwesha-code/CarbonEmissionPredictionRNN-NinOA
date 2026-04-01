# benchmarking/utils/metrics.py
"""
Shared Evaluation Metrics for All Benchmark Models.

This module is intentionally kept independent of any specific model type so
that the same metric computation logic is reused across classical ML,
statistical, and deep learning baselines, as well as the proposed NiOA-DRNN.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, R², and sMAPE.

    Parameters
    ----------
    y_true : np.ndarray   Ground-truth ΔE values.
    y_pred : np.ndarray   Predicted ΔE values.

    Returns
    -------
    dict with keys: 'MAE', 'RMSE', 'R2', 'sMAPE'
    """
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
    model_name : str,
    horizon    : int,
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    benchmark_root: str,
    extra_meta : dict = None,
) -> dict:
    """
    Persist benchmark results in a structured, self-documenting format.

    For each (model_name, horizon) combination the following are saved:
      - ``metrics.json``       : MAE, RMSE, R², sMAPE
      - ``y_test_pred.npy``    : Raw prediction array
      - ``result_summary.json``: Metrics + metadata for aggregation scripts

    Parameters
    ----------
    model_name     : str         Short model identifier (e.g. 'xgboost').
    horizon        : int         Forecast horizon in seconds.
    y_true         : np.ndarray  Ground-truth array.
    y_pred         : np.ndarray  Predicted array.
    benchmark_root : str         Top-level benchmark results directory.
    extra_meta     : dict        Optional additional metadata to store.

    Returns
    -------
    dict   Computed metrics.
    """
    save_dir = os.path.join(benchmark_root, f"horizon_{horizon}", model_name)
    os.makedirs(save_dir, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred)

    np.save(os.path.join(save_dir, "y_test_pred.npy"), y_pred)
    np.save(os.path.join(save_dir, "y_test_true.npy"), y_true)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    summary = {
        "model"   : model_name,
        "horizon" : horizon,
        "metrics" : metrics,
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
        raise FileNotFoundError(f"No benchmark results found at '{horizon_dir}'.")

    rows = []
    for model_name in sorted(os.listdir(horizon_dir)):
        metrics_path = os.path.join(horizon_dir, model_name, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            rows.append({"Model": model_name, **m})

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
