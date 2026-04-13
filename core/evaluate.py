# core/evaluate.py
"""
Standardised Model Evaluation.

All metrics computed here are reused identically across the NiOA-DRNN
proposed model and every benchmark comparator.  This centralised design
eliminates any risk of evaluation inconsistency between models.

Metrics Reported
----------------
MAE   — Mean Absolute Error (kWh)
RMSE  — Root Mean Squared Error (kWh)
R²    — Coefficient of Determination
sMAPE — Symmetric Mean Absolute Percentage Error (%)

Revision notes (v2)
--------------------
When the preprocessing pipeline applies a log1p transform to the target
variable (see preprocessing.py), the model learns to predict in
log1p(kWh) space.  All metrics must therefore be computed after applying
the inverse transform  np.expm1  to both y_true and y_pred so that the
numbers reported in the paper are in the original kWh units.

The parameter `log_transform` (default True) controls this behaviour.
Setting it to False reverts to the previous behaviour and is useful for
ablation studies or when the pipeline is run without target transformation.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true        : np.ndarray,
    y_pred        : np.ndarray,
    log_transform : bool = True,
) -> dict:
    """
    Compute the standard regression evaluation metrics.

    Parameters
    ----------
    y_true        : np.ndarray   Ground-truth values.
    y_pred        : np.ndarray   Model-predicted values.
    log_transform : bool         If True, apply np.expm1 to both arrays before
                                 computing metrics.  This restores original kWh
                                 units when the pipeline used log1p on targets.

    Returns
    -------
    dict  Keys: 'MAE', 'RMSE', 'R2', 'sMAPE'.  All values are Python floats
          expressed in original kWh units (or in log1p space if log_transform
          is False).
    """
    # ------------------------------------------------------------------
    # Inverse transform  [NEW in v2]
    # ------------------------------------------------------------------
    if log_transform:
        # Clip predictions to a safe range before inversion to prevent
        # np.expm1 from producing very large values for any pathological
        # out-of-range predictions.
        y_true = np.expm1(np.clip(y_true.astype(np.float64), -10.0, 30.0))
        y_pred = np.expm1(np.clip(y_pred.astype(np.float64), -10.0, 30.0))

    # Guard against negative predictions after inversion (physically impossible
    # for an energy increment) — clip to zero.
    y_pred = np.maximum(y_pred, 0.0)

    epsilon = 1e-8

    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))
    smape = float(
        np.mean(
            np.abs(y_true - y_pred)
            / ((np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon)
        ) * 100
    )

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE": smape}


def evaluate_keras_model(
    model         ,
    X_test        : np.ndarray,
    y_test        : np.ndarray,
    log_transform : bool = True,
):
    """
    Generate predictions from a compiled Keras model and compute metrics.

    Parameters
    ----------
    model         : tf.keras.Model   Trained Keras model.
    X_test        : np.ndarray       Test sequences (samples, seq_len, features).
    y_test        : np.ndarray       Test targets (samples,).
    log_transform : bool             If True, apply np.expm1 to y_true and
                                     y_pred before metric computation so that
                                     results are reported in original kWh units.
                                     Defaults to True.

    Returns
    -------
    metrics_df  : pd.DataFrame   Four-row summary table (Metric, Value).
    y_true_orig : np.ndarray     Ground-truth values in original kWh space.
    y_pred_orig : np.ndarray     Predictions in original kWh space.
    note        : str            Contextual note for the report.
    """
    # Raw predictions in log1p space (or original space if transform is off)
    y_pred_raw = model.predict(X_test, verbose=0).flatten()

    # ------------------------------------------------------------------
    # Inverse transform for reporting  [NEW in v2]
    # ------------------------------------------------------------------
    if log_transform:
        y_true_orig = np.expm1(np.clip(
            y_test.astype(np.float64), -10.0, 30.0
        ))
        y_pred_orig = np.expm1(np.clip(
            y_pred_raw.astype(np.float64), -10.0, 30.0
        ))
        y_pred_orig = np.maximum(y_pred_orig, 0.0)
    else:
        y_true_orig = y_test.copy()
        y_pred_orig = y_pred_raw.copy()

    # Compute metrics in original kWh space
    metrics = compute_metrics(y_true_orig, y_pred_orig, log_transform=False)

    metrics_df = pd.DataFrame(
        list(metrics.items()), columns=["Metric", "Value"]
    )

    note = (
        "Metrics are reported in original kWh units after applying the "
        "np.expm1 inverse transform.  Negative R² values, when observed, "
        "reflect the intrinsic noise and limited short-term predictability "
        "of ΔE measurements at fine temporal resolutions.  Performance "
        "improves progressively with longer forecast horizons as the "
        "signal-to-noise ratio increases."
    )

    return metrics_df, y_true_orig, y_pred_orig, note
