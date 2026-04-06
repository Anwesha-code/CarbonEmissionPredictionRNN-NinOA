# scripts/run_all_horizons.py
"""
Automated Multi-Horizon NiOA-DRNN Training Script.

This script replicates the logic of notebook 01_NiOA_DRNN_Training.ipynb
in a headless, non-interactive form so that all five forecast horizons can
be trained in a single unattended session — particularly useful when leaving
the GPU to run overnight.

Usage
-----
    python scripts/run_all_horizons.py

    # Run a specific subset of horizons:
    python scripts/run_all_horizons.py --horizons 60 300 900

    # Skip optimisation and use supplied hyperparameters:
    python scripts/run_all_horizons.py --skip-nioa --params-file path/to/best_params.json

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so that imports work regardless of where this
# script is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

from core.config        import (
    DATA_PATH, RESULTS_DIR, SPLITS_DIR, NIOA_RESULTS_DIR,
    FORECAST_HORIZONS, SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO,
    RANDOM_SEED, N_AGENTS, MAX_ITERATIONS, EXPLORATION_FACTOR,
    EXPLOITATION_FACTOR, OPT_SUBSET_RATIO, OPT_EPOCHS, OPT_PATIENCE,
    OPT_TIME_LIMIT, FINAL_EPOCHS, FINAL_PATIENCE, HYPERPARAMETER_BOUNDS,
)
from core.evaluate      import evaluate_keras_model
from core.models        import NinjaOptimizationAlgorithm, create_lstm_model
from core.preprocessing import load_and_prepare_data, split_by_timestamp
from core.train         import objective_function_lstm
from core.utils         import (
    configure_gpu, make_tf_dataset, scale_numeric_features,
    create_sequences, set_seeds, to_python_types,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _save_splits(horizon, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                 X_test_seq, y_test_seq, scaler, feature_cols,
                 train_df, val_df, test_df):
    """Persist canonical splits once per horizon."""
    split_dir = os.path.join(SPLITS_DIR, f"horizon_{horizon}")
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "X_train.npy"), X_train_seq)
    np.save(os.path.join(split_dir, "y_train.npy"), y_train_seq)
    np.save(os.path.join(split_dir, "X_val.npy"),   X_val_seq)
    np.save(os.path.join(split_dir, "y_val.npy"),   y_val_seq)
    np.save(os.path.join(split_dir, "X_test.npy"),  X_test_seq)
    np.save(os.path.join(split_dir, "y_test.npy"),  y_test_seq)
    joblib.dump(scaler, os.path.join(split_dir, "scaler.pkl"))

    with open(os.path.join(split_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    meta = {
        "horizon_k"         : horizon,
        "sequence_length"   : SEQUENCE_LENGTH,
        "train_ratio"       : TRAIN_RATIO,
        "val_ratio"         : VAL_RATIO,
        "train_start"       : str(train_df["server_timestamp"].min()),
        "train_end"         : str(train_df["server_timestamp"].max()),
        "val_start"         : str(val_df["server_timestamp"].min()),
        "val_end"           : str(val_df["server_timestamp"].max()),
        "test_start"        : str(test_df["server_timestamp"].min()),
        "test_end"          : str(test_df["server_timestamp"].max()),
        "n_train_sequences" : int(len(X_train_seq)),
        "n_val_sequences"   : int(len(X_val_seq)),
        "n_test_sequences"  : int(len(X_test_seq)),
        "n_features"        : int(X_train_seq.shape[2]),
        "random_seed"       : RANDOM_SEED,
        "generated_at"      : datetime.now().isoformat(),
    }
    with open(os.path.join(split_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [splits] Saved to {split_dir}")


# ===========================================================================
# Per-Horizon Training Function
# ===========================================================================

def train_horizon(horizon: int, skip_nioa: bool = False, fixed_params: dict = None):
    """
    Complete NiOA optimisation + final training pipeline for one horizon.

    Parameters
    ----------
    horizon      : int   Forecast horizon k (seconds).
    skip_nioa    : bool  If True, use `fixed_params` instead of running NiOA.
    fixed_params : dict  Manual hyperparameter dict (used when skip_nioa=True).
    """
    print("\n" + "=" * 65)
    print(f"  HORIZON k = {horizon} s  |  seq_len = {SEQUENCE_LENGTH}")
    print("=" * 65)

    # ── 1. Data loading and preprocessing ───────────────────────────────────
    print("  Loading and preprocessing data ...")
    df = load_and_prepare_data(DATA_PATH, k=horizon)
    print(f"  Dataset shape after preprocessing : {df.shape}")

    t_train_end = df["server_timestamp"].quantile(TRAIN_RATIO)
    t_val_end   = df["server_timestamp"].quantile(TRAIN_RATIO + VAL_RATIO)
    train_df, val_df, test_df = split_by_timestamp(df, t_train_end, t_val_end)

    assert train_df["server_timestamp"].max() < val_df["server_timestamp"].min()
    assert val_df["server_timestamp"].max()   < test_df["server_timestamp"].min()
    print("  Split verified — no leakage.")

    # ── 2. Scaling and sequence generation ──────────────────────────────────
    TARGET      = "energy_delta_k"
    feature_cols = [
        c for c in train_df.columns
        if c not in ["server_timestamp", "energy", TARGET]
    ]

    X_train_sc, X_val_sc, X_test_sc, scaler = scale_numeric_features(
        train_df[feature_cols].copy(),
        val_df[feature_cols].copy(),
        test_df[feature_cols].copy(),
        feature_cols,
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train_sc.values, train_df[TARGET].values, SEQUENCE_LENGTH
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val_sc.values, val_df[TARGET].values, SEQUENCE_LENGTH
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_sc.values, test_df[TARGET].values, SEQUENCE_LENGTH
    )

    SEQ_LEN   = X_train_seq.shape[1]
    NUM_FEATS = X_train_seq.shape[2]
    print(f"  Train seq: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")

    # ── 3. Save canonical splits ─────────────────────────────────────────────
    _save_splits(
        horizon, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq, scaler, feature_cols,
        train_df, val_df, test_df
    )

    # ── 4. NiOA / fixed hyperparameter selection ─────────────────────────────
    if skip_nioa and fixed_params:
        best_params   = fixed_params
        best_loss     = float("nan")
        convergence   = []
        print("  NiOA skipped — using supplied hyperparameters.")
    else:
        print("  Starting NiOA optimisation ...")
        n_tr_opt = int(len(X_train_seq) * OPT_SUBSET_RATIO)
        n_va_opt = int(len(X_val_seq)   * OPT_SUBSET_RATIO)

        ninja = NinjaOptimizationAlgorithm(
            objective_function = lambda p: objective_function_lstm(
                p,
                X_train_seq[:n_tr_opt], y_train_seq[:n_tr_opt],
                X_val_seq[:n_va_opt],   y_val_seq[:n_va_opt],
                opt_epochs   = OPT_EPOCHS,
                opt_patience = OPT_PATIENCE,
                time_limit   = OPT_TIME_LIMIT,
            ),
            bounds              = HYPERPARAMETER_BOUNDS,
            n_agents            = N_AGENTS,
            max_iterations      = MAX_ITERATIONS,
            exploration_factor  = EXPLORATION_FACTOR,
            exploitation_factor = EXPLOITATION_FACTOR,
            verbose             = True,
        )
        best_params, best_loss, convergence = ninja.optimize()
        print(f"  Best NiOA val MSE : {best_loss:.8f}")
        print(f"  Best params : {best_params}")

    # ── 5. Create experiment directory ───────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(NIOA_RESULTS_DIR, f"k{horizon}_seq{SEQUENCE_LENGTH}_{ts}")
    for sub in ["model", "predictions", "plots"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    with open(os.path.join(exp_dir, "best_params.json"), "w") as f:
        json.dump(to_python_types(best_params), f, indent=4)

    np.save(os.path.join(exp_dir, "convergence.npy"), np.array(convergence))

    # ── 6. Final model training ───────────────────────────────────────────────
    K.clear_session(); gc.collect()

    model = create_lstm_model(best_params, SEQ_LEN, NUM_FEATS)

    BATCH    = best_params["batch_size"]
    train_ds = make_tf_dataset(X_train_seq, y_train_seq, SEQ_LEN, NUM_FEATS, BATCH)
    val_ds   = make_tf_dataset(X_val_seq,   y_val_seq,   SEQ_LEN, NUM_FEATS, BATCH)

    history = model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = FINAL_EPOCHS,
        steps_per_epoch  = len(X_train_seq) // BATCH,
        validation_steps = len(X_val_seq)   // BATCH,
        callbacks        = [EarlyStopping(
            monitor="val_loss", patience=FINAL_PATIENCE,
            restore_best_weights=True, verbose=1,
        )],
        verbose=1,
    )

    # ── 7. Save model and artefacts ───────────────────────────────────────────
    model.save(os.path.join(exp_dir, "model", f"NiOA_DRNN_k{horizon}.h5"))
    joblib.dump(scaler, os.path.join(exp_dir, "scaler.pkl"))

    training_config = {
        "horizon_k"       : horizon,
        "sequence_length" : SEQUENCE_LENGTH,
        "best_params"     : to_python_types(best_params),
        "best_nioa_loss"  : float(best_loss) if not np.isnan(best_loss) else None,
        "completed_at"    : datetime.now().isoformat(),
    }
    with open(os.path.join(exp_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=4)

    # ── 8. Evaluation ────────────────────────────────────────────────────────
    metrics_df, y_true, y_pred, _ = evaluate_keras_model(model, X_test_seq, y_test_seq)
    print(metrics_df.to_string(index=False))

    np.save(os.path.join(exp_dir, "predictions", "y_test_true.npy"), y_true)
    np.save(os.path.join(exp_dir, "predictions", "y_test_pred.npy"), y_pred)

    metrics_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"  Experiment artefacts saved to : {exp_dir}")

    # Clean up GPU memory before the next horizon
    del model, X_train_seq, X_val_seq, X_test_seq
    K.clear_session(); gc.collect()


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NiOA-DRNN for multiple forecast horizons."
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=FORECAST_HORIZONS,
        help=f"Horizons to train (default: all {FORECAST_HORIZONS})"
    )
    parser.add_argument(
        "--skip-nioa", action="store_true",
        help="Skip NiOA and use --params-file instead."
    )
    parser.add_argument(
        "--params-file", type=str, default=None,
        help="Path to a best_params.json to use when --skip-nioa is set."
    )
    args = parser.parse_args()

    configure_gpu()
    set_seeds(RANDOM_SEED)

    fixed_params = None
    if args.skip_nioa and args.params_file:
        with open(args.params_file) as f:
            fixed_params = json.load(f)
        print(f"Loaded fixed hyperparameters from : {args.params_file}")

    print(f"\nHorizons to train : {args.horizons}")
    print(f"NiOA skipped      : {args.skip_nioa}")
    print(f"RANDOM_SEED       : {RANDOM_SEED}")
    print(f"SEQUENCE_LENGTH   : {SEQUENCE_LENGTH}")

    for k in args.horizons:
        if k not in FORECAST_HORIZONS:
            print(f"WARNING: {k} is not in FORECAST_HORIZONS {FORECAST_HORIZONS} — skipping.")
            continue
        train_horizon(k, skip_nioa=args.skip_nioa, fixed_params=fixed_params)

    print("\n" + "=" * 65)
    print("  ALL HORIZONS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
