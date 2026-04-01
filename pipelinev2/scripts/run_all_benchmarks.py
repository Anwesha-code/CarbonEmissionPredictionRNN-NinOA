# scripts/run_all_benchmarks.py
"""
Automated Benchmark Runner — Classical and Deep Learning Baselines.

Runs Linear Regression, SVR, XGBoost, MLP, Vanilla LSTM, and CNN-LSTM
for every available frozen split horizon in a single session.

ARIMA and DRNN+Optuna are intentionally excluded here because:
  - ARIMA is slow (rolling forecast) and is best run interactively.
  - DRNN+Optuna involves a full Optuna study and benefits from monitoring.

Usage
-----
    python scripts/run_all_benchmarks.py
    python scripts/run_all_benchmarks.py --horizons 60 300 --models lr xgboost lstm

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import argparse, gc, json, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.linear_model    import LinearRegression
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
)
from tensorflow.keras.callbacks import EarlyStopping

try:
    from xgboost import XGBRegressor
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False
    print("WARNING: xgboost not installed — XGBoost benchmark will be skipped.")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.config  import SPLITS_DIR, BENCHMARK_DIR, FORECAST_HORIZONS, RANDOM_SEED
from core.utils   import configure_gpu, set_seeds, make_tf_dataset
from benchmarking.utils.data_loader import (
    load_splits, flatten_sequences, combine_train_val
)
from benchmarking.utils.metrics import save_benchmark_results


# ===========================================================================
# Individual model runners
# ===========================================================================

def run_linear_regression(X_tv, y_tv, X_test, y_test, horizon):
    mdl = LinearRegression()
    mdl.fit(X_tv, y_tv)
    y_pred = mdl.predict(X_test)
    return save_benchmark_results("linear_regression", horizon, y_test, y_pred, BENCHMARK_DIR)


def run_svr(X_tr, y_tr, X_test, y_test, horizon, max_train=50_000):
    X_tr_s = X_tr[:max_train]
    y_tr_s = y_tr[:max_train]
    mdl = SVR(kernel="rbf", C=10, epsilon=0.1)
    mdl.fit(X_tr_s, y_tr_s)
    y_pred = mdl.predict(X_test)
    return save_benchmark_results(
        "svr", horizon, y_test, y_pred, BENCHMARK_DIR,
        extra_meta={"training_subset": max_train}
    )


def run_xgboost(X_tr, y_tr, X_va, y_va, X_test, y_test, horizon):
    if not _XGBOOST_OK:
        print("  [xgboost] Skipped — package not installed.")
        return {}
    mdl = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=20, eval_metric="rmse",
        random_state=RANDOM_SEED, verbosity=0,
    )
    mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    y_pred = mdl.predict(X_test)
    return save_benchmark_results("xgboost", horizon, y_test, y_pred, BENCHMARK_DIR)


def run_mlp(X_tv, y_tv, X_test, y_test, horizon):
    mdl = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu",
        solver="adam", learning_rate_init=5e-4,
        max_iter=200, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15,
        random_state=RANDOM_SEED, verbose=False,
    )
    mdl.fit(X_tv, y_tv)
    y_pred = mdl.predict(X_test)
    return save_benchmark_results("mlp", horizon, y_test, y_pred, BENCHMARK_DIR)


def run_vanilla_lstm(X_train, y_train, X_val, y_val, X_test, y_test, horizon):
    K.clear_session(); gc.collect()

    SEQ = X_train.shape[1]
    F   = X_train.shape[2]
    B   = 32

    mdl = Sequential(name="Vanilla_LSTM")
    mdl.add(Input(shape=(SEQ, F)))
    mdl.add(LSTM(64, return_sequences=True))
    mdl.add(Dropout(0.3))
    mdl.add(LSTM(64))
    mdl.add(Dropout(0.3))
    mdl.add(Dense(32, activation="relu"))
    mdl.add(Dense(1))
    mdl.compile(loss="mse", optimizer="adam", metrics=["mae"])

    train_ds = make_tf_dataset(X_train, y_train, SEQ, F, B)
    val_ds   = make_tf_dataset(X_val,   y_val,   SEQ, F, B)

    mdl.fit(
        train_ds, validation_data=val_ds,
        epochs=40,
        steps_per_epoch  = len(X_train) // B,
        validation_steps = len(X_val)   // B,
        callbacks=[EarlyStopping(monitor="val_loss", patience=7,
                                 restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    y_pred = mdl.predict(X_test, verbose=0).flatten()
    metrics = save_benchmark_results("vanilla_lstm", horizon, y_test, y_pred, BENCHMARK_DIR)
    del mdl; K.clear_session(); gc.collect()
    return metrics


def run_cnn_lstm(X_train, y_train, X_val, y_val, X_test, y_test, horizon):
    K.clear_session(); gc.collect()

    SEQ = X_train.shape[1]
    F   = X_train.shape[2]
    B   = 32

    mdl = Sequential(name="CNN_LSTM")
    mdl.add(Input(shape=(SEQ, F)))
    mdl.add(Conv1D(64, kernel_size=3, activation="relu", padding="same"))
    mdl.add(MaxPooling1D(pool_size=2))
    mdl.add(Conv1D(32, kernel_size=3, activation="relu", padding="same"))
    mdl.add(LSTM(64))
    mdl.add(Dropout(0.3))
    mdl.add(Dense(32, activation="relu"))
    mdl.add(Dense(1))
    mdl.compile(loss="mse", optimizer="adam", metrics=["mae"])

    train_ds = make_tf_dataset(X_train, y_train, SEQ, F, B)
    val_ds   = make_tf_dataset(X_val,   y_val,   SEQ, F, B)

    mdl.fit(
        train_ds, validation_data=val_ds,
        epochs=40,
        steps_per_epoch  = len(X_train) // B,
        validation_steps = len(X_val)   // B,
        callbacks=[EarlyStopping(monitor="val_loss", patience=7,
                                 restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    y_pred = mdl.predict(X_test, verbose=0).flatten()
    metrics = save_benchmark_results("cnn_lstm", horizon, y_test, y_pred, BENCHMARK_DIR)
    del mdl; K.clear_session(); gc.collect()
    return metrics


# ===========================================================================
# Per-Horizon Runner
# ===========================================================================

MODEL_REGISTRY = {
    "lr"          : "linear_regression",
    "svr"         : "svr",
    "xgboost"     : "xgboost",
    "mlp"         : "mlp",
    "lstm"        : "vanilla_lstm",
    "cnn_lstm"    : "cnn_lstm",
}

def run_benchmarks_for_horizon(horizon: int, models_to_run: list):
    print(f"\n{'='*65}")
    print(f"  BENCHMARKS — k = {horizon} s")
    print(f"{'='*65}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, meta = \
        load_splits(SPLITS_DIR, horizon)

    X_tr_flat = flatten_sequences(X_train)
    X_va_flat = flatten_sequences(X_val)
    X_te_flat = flatten_sequences(X_test)
    X_tv_flat, y_tv = combine_train_val(X_tr_flat, y_train, X_va_flat, y_val)

    results = {}

    if "lr" in models_to_run:
        print("  [1/6] Linear Regression ...")
        results["linear_regression"] = run_linear_regression(
            X_tv_flat, y_tv, X_te_flat, y_test, horizon
        )

    if "svr" in models_to_run:
        print("  [2/6] SVR ...")
        results["svr"] = run_svr(
            X_tv_flat, y_tv, X_te_flat, y_test, horizon
        )

    if "xgboost" in models_to_run:
        print("  [3/6] XGBoost ...")
        results["xgboost"] = run_xgboost(
            X_tr_flat, y_train, X_va_flat, y_val, X_te_flat, y_test, horizon
        )

    if "mlp" in models_to_run:
        print("  [4/6] MLP ...")
        results["mlp"] = run_mlp(
            X_tv_flat, y_tv, X_te_flat, y_test, horizon
        )

    if "lstm" in models_to_run:
        print("  [5/6] Vanilla LSTM ...")
        results["vanilla_lstm"] = run_vanilla_lstm(
            X_train, y_train, X_val, y_val, X_test, y_test, horizon
        )

    if "cnn_lstm" in models_to_run:
        print("  [6/6] CNN-LSTM ...")
        results["cnn_lstm"] = run_cnn_lstm(
            X_train, y_train, X_val, y_val, X_test, y_test, horizon
        )

    # Print summary
    print(f"\n  Results for k = {horizon} s :")
    for name, m in results.items():
        if m:
            print(f"    {name:<22} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")

    return results


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run classical and DL benchmarks for all saved horizon splits."
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=None,
        help="Horizons to benchmark (default: all with saved splits)."
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to include (default: all)."
    )
    args = parser.parse_args()

    configure_gpu()
    set_seeds(RANDOM_SEED)

    # Discover which horizons have saved splits
    available_horizons = []
    for h in FORECAST_HORIZONS:
        split_dir = os.path.join(SPLITS_DIR, f"horizon_{h}")
        if os.path.isdir(split_dir) and os.path.isfile(
            os.path.join(split_dir, "X_train.npy")
        ):
            available_horizons.append(h)
        else:
            print(f"  WARNING: No saved splits for k={h} — skipping.")

    horizons = args.horizons if args.horizons else available_horizons
    print(f"\nHorizons to benchmark : {horizons}")
    print(f"Models to run         : {args.models}")

    for h in horizons:
        if h not in available_horizons:
            print(f"  WARNING: k={h} has no saved splits — skipping.")
            continue
        run_benchmarks_for_horizon(h, args.models)

    print("\n" + "=" * 65)
    print("  ALL BENCHMARKS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
