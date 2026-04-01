# benchmarking/utils/data_loader.py
"""
Benchmark Data Loader.

Loads the canonical splits that were generated and frozen by the NiOA-DRNN
training notebook (01_NiOA_DRNN_Training.ipynb).  Every benchmark model must
use these arrays exclusively to guarantee a fair, leakage-free comparison.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import os
import json
import numpy as np
import joblib


def load_splits(splits_root: str, horizon: int):
    """
    Load the frozen train / validation / test arrays for a given horizon.

    Parameters
    ----------
    splits_root : str   Path to the top-level splits directory
                        (e.g., ``results/splits``).
    horizon     : int   Forecast horizon in seconds (1, 60, 300, 900, 1800).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Sequence arrays in float32.  X arrays have shape
        (N, seq_len, n_features); y arrays have shape (N,).
    scaler : sklearn.preprocessing.StandardScaler
        The scaler fitted on the training split — reuse for any model that
        requires separately scaled flat features.
    meta   : dict
        Split metadata loaded from ``split_metadata.json``.
    """
    path = os.path.join(splits_root, f"horizon_{horizon}")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No split directory found at '{path}'.  "
            f"Please run 01_NiOA_DRNN_Training.ipynb for horizon k={horizon} first."
        )

    X_train = np.load(os.path.join(path, "X_train.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    X_val   = np.load(os.path.join(path, "X_val.npy"))
    y_val   = np.load(os.path.join(path, "y_val.npy"))
    X_test  = np.load(os.path.join(path, "X_test.npy"))
    y_test  = np.load(os.path.join(path, "y_test.npy"))
    scaler  = joblib.load(os.path.join(path, "scaler.pkl"))

    with open(os.path.join(path, "split_metadata.json")) as f:
        meta = json.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, meta


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3-D LSTM sequence arrays to 2-D for classical ML models.

    (N, seq_len, n_features) → (N, seq_len × n_features)

    This preserves all temporal information in a format compatible with
    sklearn estimators such as LinearRegression, SVR, and XGBoost.

    Parameters
    ----------
    X : np.ndarray   3-D array of shape (N, seq_len, n_features).

    Returns
    -------
    np.ndarray   2-D array of shape (N, seq_len × n_features).
    """
    return X.reshape(X.shape[0], -1)


def combine_train_val(
    X_train, y_train, X_val, y_val
):
    """
    Concatenate train and validation splits for final benchmark model fitting.

    For benchmark models that do not require a separate validation split
    (e.g., Linear Regression, SVR), the combined train+val set is used as
    the final training data to maximise the information available.

    Returns
    -------
    X_trainval : np.ndarray   Concatenated features.
    y_trainval : np.ndarray   Concatenated targets.
    """
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    return X_trainval, y_trainval
