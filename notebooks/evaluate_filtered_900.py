# evaluate_filtered.py
"""
Post-hoc Physical Plausibility Filter and Re-evaluation
========================================================

This script re-evaluates an already-trained and saved NiOA-DRNN model
on a filtered subset of the test set, without any retraining.

What this does
--------------selecting AshCEP but it is still not showing on 
The saved y_test.npy arrays contain log1p-transformed energy_delta_k values.
When converted back to kWh via np.expm1, approximately 10.6% of test samples
contain physically implausible values of ~417 kWh for a 60-second window.
These are meter counter reset artefacts that passed the gap filter.

For a workstation-scale sensor, the maximum physically possible energy
increment over k seconds at 1000 W continuous load is:
    max_plausible_kwh = 1000 * k / 3600

This script removes those artefact rows from the evaluation set and
recomputes MAE, RMSE, R², and sMAPE on the clean subset.

This does NOT retrain the model. It only changes which test samples
are included in the evaluation. Both the filtered and unfiltered metrics
should be reported for full transparency.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import os
import sys
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── Adjust these paths to match your local setup ──────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.config  import SPLITS_DIR, NIOA_RESULTS_DIR, RESULTS_DIR
from core.utils   import configure_gpu, SequenceDataGenerator


# =============================================================================
# CONFIGURATION — edit these two values if needed
# =============================================================================

HORIZON     = 900     # seconds — must match the saved model
MAX_POWER_W = 1000   # conservative upper bound for workstation power in watts

# =============================================================================
# DERIVED SETTINGS
# =============================================================================

MAX_PLAUSIBLE_KWH    = MAX_POWER_W * HORIZON / 3600.0
MAX_PLAUSIBLE_LOG1P  = float(np.log1p(MAX_PLAUSIBLE_KWH))
BATCH_SIZE           = 16    # use same batch size as training


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², sMAPE. Inputs must already be in kWh space."""
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


def main():
    configure_gpu()

    # ── 1. Locate the saved splits ────────────────────────────────────────────
    split_dir = os.path.join(SPLITS_DIR, f"horizon_{HORIZON}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"No split directory found at {split_dir}. "
            f"Run notebook 01 first for k={HORIZON}."
        )

    print(f"\n{'='*65}")
    print(f"  Post-hoc filtered evaluation — k = {HORIZON} s")
    print(f"  Max plausible ΔE : {MAX_PLAUSIBLE_KWH:.4f} kWh  "
          f"(log1p = {MAX_PLAUSIBLE_LOG1P:.4f})")
    print(f"{'='*65}\n")

    # ── 2. Load test arrays (in log1p space) ──────────────────────────────────
    print("Loading test arrays ...")
    X_test = np.load(os.path.join(split_dir, "X_test.npy"), mmap_mode='r')
    y_test = np.load(os.path.join(split_dir, "y_test.npy"), mmap_mode='r')
    print(f"  X_test shape : {X_test.shape}")
    print(f"  y_test shape : {y_test.shape}")
    print(f"  y_test range : [{y_test.min():.4f}, {y_test.max():.4f}]  (log1p space)")

    # ── 3. Build the physical plausibility mask ───────────────────────────────
    # y_test is in log1p(kWh) space.
    # A value of log1p(417) ≈ 6.035 corresponds to the artefact samples.
    # We keep only rows where log1p(y) ≤ log1p(max_plausible_kwh).
    mask_clean = y_test <= MAX_PLAUSIBLE_LOG1P

    n_total   = len(y_test)
    n_clean   = int(mask_clean.sum())
    n_removed = n_total - n_clean
    frac_removed = n_removed / n_total * 100

    print(f"\n  Artefact filter (|ΔE| ≤ {MAX_PLAUSIBLE_KWH:.2f} kWh):")
    print(f"    Total test samples     : {n_total:>9,}")
    print(f"    Clean samples retained : {n_clean:>9,}  ({100-frac_removed:.2f}%)")
    print(f"    Artefacts removed      : {n_removed:>9,}  ({frac_removed:.2f}%)")

    #X_test_clean = X_test[mask_clean]
    #y_test_clean = y_test[mask_clean]

    #print(f"\n  X_test_clean shape : {X_test_clean.shape}")

    # ── 4. Find the saved model ───────────────────────────────────────────────
    # Look for the most recent experiment directory for this horizon
    if not os.path.isdir(NIOA_RESULTS_DIR):
        raise FileNotFoundError(
            f"No NiOA results directory found at {NIOA_RESULTS_DIR}."
        )

    candidates = [
        d for d in os.listdir(NIOA_RESULTS_DIR)
        if d.startswith(f"k{HORIZON}_") and
        os.path.isfile(os.path.join(
            NIOA_RESULTS_DIR, d, "model", f"NiOA_DRNN_k{HORIZON}.h5"
        ))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No saved model found for k={HORIZON} in {NIOA_RESULTS_DIR}. "
            f"Run notebook 01 first."
        )

    latest_exp = sorted(candidates)[-1]
    model_path = os.path.join(
        NIOA_RESULTS_DIR, latest_exp, "model", f"NiOA_DRNN_k{HORIZON}.h5"
    )
    print(f"\nLoading model from : {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

   # ── 5. Predict on FULL test set (original, unfiltered) ───────────────────
    print("\nPredicting on FULL test set (original, unfiltered) ...")
    full_gen      = SequenceDataGenerator(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Predict everything once (mmap handles batching safely from disk)
    y_pred_full_log1p = model.predict(full_gen, verbose=1).flatten()

    # Align lengths due to batch size remainder
    valid_len = len(y_pred_full_log1p)
    y_test_valid = y_test[:valid_len]
    mask_clean_valid = mask_clean[:valid_len]

    # Convert full predictions back to kWh
    y_true_full_kwh = np.expm1(np.clip(y_test_valid.astype(np.float64), -10., 30.))
    y_pred_full_kwh = np.maximum(
        np.expm1(np.clip(y_pred_full_log1p.astype(np.float64), -10., 30.)),
        0.0
    )
    metrics_full = compute_metrics(y_true_full_kwh, y_pred_full_kwh)

    # ── 6. Extract CLEAN test set predictions (artefacts removed) ────────────
    print("\nExtracting CLEAN test set predictions...")
    
    # Apply the mask directly to the 1D arrays we already calculated!
    y_true_clean_kwh = y_true_full_kwh[mask_clean_valid]
    y_pred_clean_kwh = y_pred_full_kwh[mask_clean_valid]

    # Recompute N based on valid truncated batch
    n_clean_actual = len(y_true_clean_kwh)
    print(f"  Clean predictions extracted (n={n_clean_actual:,})")

    metrics_clean = compute_metrics(y_true_clean_kwh, y_pred_clean_kwh)
    # ── 7. Print comparison ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS COMPARISON — k = {HORIZON} s")
    print(f"{'='*65}")
    print(f"\n  {'Metric':<10}  {'Full test set':>18}  {'Clean subset':>18}")
    print(f"  {'':─<10}  {'(n='+str(n_total)+')':>18}  {'(n='+str(n_clean)+')':>18}")
    for metric in ["MAE", "RMSE", "R2", "sMAPE"]:
        unit = "kWh" if metric in ("MAE", "RMSE") else ("" if metric == "R2" else "%")
        print(f"  {metric:<10}  {metrics_full[metric]:>14.4f} {unit:>3}  "
              f"{metrics_clean[metric]:>14.4f} {unit:>3}")

    # ── 8. Save filtered predictions and metrics ──────────────────────────────
    out_dir = os.path.join(
        NIOA_RESULTS_DIR, latest_exp, "predictions"
    )
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_test_true_clean.npy"),  y_true_clean_kwh)
    np.save(os.path.join(out_dir, "y_test_pred_clean.npy"),  y_pred_clean_kwh)
    np.save(os.path.join(out_dir, "y_test_true_full.npy"),   y_true_full_kwh)
    np.save(os.path.join(out_dir, "y_test_pred_full.npy"),   y_pred_full_kwh)

    results_summary = {
        "horizon_k"             : HORIZON,
        "max_power_watts"       : MAX_POWER_W,
        "max_plausible_kwh"     : MAX_PLAUSIBLE_KWH,
        "n_test_total"          : n_total,
        "n_test_clean"          : n_clean,
        "n_artefacts_removed"   : n_removed,
        "pct_artefacts"         : round(frac_removed, 4),
        "metrics_full_test"     : metrics_full,
        "metrics_clean_subset"  : metrics_clean,
        "model_path"            : model_path,
        "note": (
            "metrics_clean_subset reports performance on the physically "
            "plausible subset of the test set (energy increment ≤ "
            f"{MAX_PLAUSIBLE_KWH:.2f} kWh for k={HORIZON}s at 1000W max). "
            "metrics_full_test includes all samples including counter-reset "
            "artefacts that dominate the error. Both are reported for transparency."
        )
    }

    metrics_path = os.path.join(
        NIOA_RESULTS_DIR, latest_exp, "metrics_filtered.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\n  Saved to : {metrics_path}")
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  Full test set  : MAE = {metrics_full['MAE']:.4f} kWh | "
          f"R² = {metrics_full['R2']:.4f}")
    print(f"  Clean subset   : MAE = {metrics_clean['MAE']:.4f} kWh | "
          f"R² = {metrics_clean['R2']:.4f}")
    print(f"\n  Improvement in MAE : "
          f"{metrics_full['MAE'] - metrics_clean['MAE']:.4f} kWh reduction")
    print(f"  The clean subset represents {100-frac_removed:.1f}% of the test set.")
    print(f"{'='*65}\n")

    # ── 9. Generate quick comparison plots ───────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"NiOA-DRNN — k={HORIZON}s | Post-hoc Filtered Evaluation",
            fontsize=13, fontweight="bold"
        )

        # Left: scatter — full test set
        axes[0].scatter(
            y_true_full_kwh, y_pred_full_kwh,
            alpha=0.3, s=3, color="coral"
        )
        _lim = [0, max(y_true_full_kwh.max(), y_pred_full_kwh.max())]
        axes[0].plot(_lim, _lim, "b--", lw=1.5, label="Perfect")
        axes[0].set_title(f"Full Test Set (n={n_total:,})")
        axes[0].set_xlabel("Actual ΔE (kWh)")
        axes[0].set_ylabel("Predicted ΔE (kWh)")
        axes[0].legend(fontsize=8)
        axes[0].annotate(
            f"MAE={metrics_full['MAE']:.2f}\nR²={metrics_full['R2']:.3f}",
            xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8,
            color="coral"
        )

        # Middle: scatter — clean subset
        axes[1].scatter(
            y_true_clean_kwh, y_pred_clean_kwh,
            alpha=0.3, s=3, color="steelblue"
        )
        _lim2 = [0, max(y_true_clean_kwh.max(), y_pred_clean_kwh.max())]
        axes[1].plot(_lim2, _lim2, "r--", lw=1.5, label="Perfect")
        axes[1].set_title(f"Clean Subset (n={n_clean:,}, {100-frac_removed:.1f}%)")
        axes[1].set_xlabel("Actual ΔE (kWh)")
        axes[1].set_ylabel("Predicted ΔE (kWh)")
        axes[1].legend(fontsize=8)
        axes[1].annotate(
            f"MAE={metrics_clean['MAE']:.4f}\nR²={metrics_clean['R2']:.4f}",
            xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8,
            color="steelblue"
        )

        # Right: residual distributions
        res_full  = y_true_full_kwh  - y_pred_full_kwh
        res_clean = y_true_clean_kwh - y_pred_clean_kwh
        sns.histplot(
            res_clean, bins=60, kde=True, ax=axes[2],
            color="steelblue", label="Clean subset"
        )
        axes[2].axvline(0, color="red", lw=1.2, linestyle="--")
        axes[2].set_xlabel("Residual (kWh)")
        axes[2].set_title("Residual Distribution — Clean Subset")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(
            NIOA_RESULTS_DIR, latest_exp, "predictions",
            "comparison_filtered_vs_full.png"
        )
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"  Comparison plot saved to : {plot_path}\n")

    except ImportError:
        print("  (matplotlib/seaborn not available — plots skipped)\n")


if __name__ == "__main__":
    main()
