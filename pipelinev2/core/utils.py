# core/utils.py
"""
General Utility Functions.

Covers:
  - Reproducibility seed setting across NumPy, Python random, and TensorFlow.
  - Train-only StandardScaler fitting and transformation.
  - Sliding-window sequence generation for LSTM input construction.
  - JSON-safe conversion of NumPy scalar types.
  - tf.data.Dataset pipeline construction for memory-efficient training.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seeds(seed: int = 42) -> None:
    """
    Fix all random number generators to ensure fully reproducible results
    across repeated runs on the same hardware configuration.

    Parameters
    ----------
    seed : int   Master random seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===========================================================================
# Feature Scaling
# ===========================================================================

def scale_numeric_features(train, val, test, feature_cols):
    """
    Fit a StandardScaler exclusively on the training split and apply the
    learned transformation to validation and test splits.

    Fitting only on training data is critical to prevent information leakage
    from future observations into the model's input representation.

    Parameters
    ----------
    train        : pd.DataFrame   Training split.
    val          : pd.DataFrame   Validation split.
    test         : pd.DataFrame   Test split.
    feature_cols : List[str]      Column names to be scaled.

    Returns
    -------
    (train, val, test, scaler)
        Modified dataframes and the fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    val[feature_cols]   = scaler.transform(val[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    return train, val, test, scaler


# ===========================================================================
# Sequence Generation
# ===========================================================================

def create_sequences(
    data   : np.ndarray,
    target : np.ndarray,
    seq_len: int,
):
    """
    Construct overlapping sliding-window sequences for LSTM training.

    For each valid starting index i, a sequence X[i : i+seq_len] is paired
    with target y[i+seq_len].  The resulting arrays conform to Keras LSTM
    input convention: (samples, timesteps, features).

    Parameters
    ----------
    data    : np.ndarray  2-D feature array of shape (N, F).
    target  : np.ndarray  1-D target array of length N.
    seq_len : int         Number of consecutive time steps per sequence.

    Returns
    -------
    X : np.ndarray  Shape (N − seq_len, seq_len, F), dtype float32.
    y : np.ndarray  Shape (N − seq_len,),            dtype float32.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ===========================================================================
# JSON Serialisation Helper
# ===========================================================================

def to_python_types(d: dict) -> dict:
    """
    Recursively convert NumPy scalar values within a dictionary to native
    Python int / float types so that the dictionary can be safely serialised
    with the standard json module.

    Parameters
    ----------
    d : dict   Dictionary potentially containing NumPy scalars.

    Returns
    -------
    dict   A new dictionary with all NumPy scalars replaced by Python natives.
    """
    clean = {}
    for key, val in d.items():
        if hasattr(val, "item"):          # catches np.int32, np.float64, etc.
            clean[key] = val.item()
        elif isinstance(val, np.str_):
            clean[key] = str(val)
        else:
            clean[key] = val
    return clean


# ===========================================================================
# tf.data.Dataset Generator (Memory-Efficient Training)
# ===========================================================================

def make_tf_dataset(
    X           : np.ndarray,
    y           : np.ndarray,
    seq_len     : int,
    num_feats   : int,
    batch_size  : int,
    shuffle     : bool = False,
    buffer_size : int  = 10_000,
):
    """
    Build a tf.data.Dataset from pre-computed sequence arrays.

    Prefetching is enabled unconditionally to overlap data loading with
    GPU computation, reducing the overall epoch wall-clock time.

    Parameters
    ----------
    X          : np.ndarray  Sequence array (samples, seq_len, features).
    y          : np.ndarray  Target array (samples,).
    seq_len    : int         Sequence length — used only in the output signature.
    num_feats  : int         Feature count — used only in the output signature.
    batch_size : int         Mini-batch size.
    shuffle    : bool        Whether to shuffle before batching (use for train).
    buffer_size: int         Shuffle buffer size.

    Returns
    -------
    tf.data.Dataset   Batched and prefetched dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================================================================
# GPU Configuration
# ===========================================================================

def configure_gpu() -> bool:
    """
    Enable per-process GPU memory growth to prevent TensorFlow from
    pre-allocating the entire GPU memory pool on session start.

    Returns
    -------
    bool   True if at least one GPU was detected and configured.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configured: {len(gpus)} device(s) detected.")
        return True
    print("No GPU detected — training will proceed on CPU.")
    return False
