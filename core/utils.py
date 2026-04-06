# core/utils.py
"""
General Utility Functions.

Why SequenceDataGenerator instead of tf.data or model.fit(X, y)?
-----------------------------------------------------------------
The training dataset for this project is (1,874,617 × 120 × 17) float32
≈ 15 GB.  The two approaches that fail at this size:

  model.fit(X_train, y_train, ...)
    → Keras calls tf.constant(X_train) internally, which copies the ENTIRE
      15 GB array into a TensorFlow EagerTensor on the GPU → OOM crash.

  tf.data.Dataset.from_tensor_slices((X, y))
    → Same problem — converts the full array into a graph constant before
      any batching occurs → same OOM crash.

The correct solution for large datasets on GPU:
  tf.keras.utils.Sequence subclass (SequenceDataGenerator).

  How it works:
    - The full numpy array stays in CPU RAM throughout training.
    - On each training step Keras calls __getitem__(batch_index), which
      returns a plain numpy batch of shape (batch_size, seq_len, feats).
    - Keras converts ONLY that one batch to a GPU tensor, runs the forward
      pass, discards the GPU tensor, and moves to the next batch.
    - Peak GPU memory = one batch only, regardless of dataset size.
    - Works for any dataset size as long as CPU RAM can hold the full array.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import math
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seeds(seed: int = 42) -> None:
    """Fix all RNGs for fully reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===========================================================================
# GPU Configuration
# ===========================================================================

def configure_gpu() -> bool:
    """
    Enable per-process memory growth on every available GPU.

    Without this TensorFlow pre-allocates the entire VRAM pool at session
    start.  For large datasets that can cause the GPU to OOM before the
    first batch is processed.

    Must be called BEFORE any TensorFlow computation is started and before
    any model or dataset object is created.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU ready — {len(gpus)} device(s), memory growth enabled.")
        return True
    print("No GPU detected — training will run on CPU.")
    return False


# ===========================================================================
# Feature Scaling
# ===========================================================================

def scale_numeric_features(train, val, test, feature_cols):
    """
    Fit StandardScaler on training data only and transform all three splits.

    Fitting on training data only prevents any future-data leakage into the
    feature normalisation.

    Parameters
    ----------
    train, val, test : pd.DataFrame  The three chronological splits.
    feature_cols     : List[str]     Columns to scale.

    Returns
    -------
    (train, val, test, scaler)
    """
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    train[feature_cols] = scaler.transform(train[feature_cols])
    val[feature_cols]   = scaler.transform(val[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])
    return train, val, test, scaler


# ===========================================================================
# Sliding-Window Sequence Generation
# ===========================================================================

def create_sequences(
    data   : np.ndarray,
    target : np.ndarray,
    seq_len: int,
) -> tuple:
    """
    Build overlapping sliding-window sequences using numpy stride tricks.

    For each valid start index i the sequence X[i : i+seq_len] is paired
    with target y[i+seq_len], giving Keras-compatible arrays of shape
    (N-seq_len, seq_len, features).

    Stride tricks avoid allocating a Python list of subarrays and are
    ~5-10× faster than the list-append approach for large N.

    Parameters
    ----------
    data    : np.ndarray  (N, F) feature array — must be C-contiguous.
    target  : np.ndarray  (N,)   target array.
    seq_len : int         Number of timesteps per sequence.

    Returns
    -------
    X : np.ndarray  float32, shape (N-seq_len, seq_len, F)
    y : np.ndarray  float32, shape (N-seq_len,)
    """
    # Ensure the base array is contiguous so stride arithmetic is valid
    data = np.ascontiguousarray(data, dtype=np.float32)

    N, F    = data.shape
    n_seqs  = N - seq_len

    # Stride trick: shape (n_seqs, seq_len, F) without intermediate copies
    from numpy.lib.stride_tricks import as_strided
    row_stride = data.strides[0]
    col_stride = data.strides[1]
    X = as_strided(
        data,
        shape   = (n_seqs, seq_len, F),
        strides = (row_stride, row_stride, col_stride),
    ).copy()   # .copy() makes it fully independent and C-contiguous

    y = target[seq_len:].astype(np.float32)
    return X, y


# ===========================================================================
# Batch Data Generator  (GPU-safe for large datasets)
# ===========================================================================

class SequenceDataGenerator(tf.keras.utils.Sequence):
    """
    A Keras-native batch generator that feeds large numpy arrays to the GPU
    one batch at a time, keeping the full dataset in CPU RAM.

    Design rationale
    ----------------
    For arrays of ~15 GB (this project's training set), calling
    ``model.fit(X, y)`` or ``tf.data.Dataset.from_tensor_slices((X, y))``
    causes TensorFlow to convert the entire array into a GPU-resident
    EagerTensor before batching — crashing with
    "Dst tensor is not initialized" or CUDA OOM.

    ``tf.keras.utils.Sequence`` avoids this because:
      1. Keras calls ``__getitem__(i)`` to fetch one batch of numpy arrays.
      2. The batch (batch_size × seq_len × features) is small and fits in VRAM.
      3. Keras converts ONLY that batch to GPU, runs the step, frees the tensor.
      4. The 15 GB array never touches the GPU as a whole.

    Usage
    -----
    >>> gen = SequenceDataGenerator(X_train, y_train, batch_size=32, shuffle=True)
    >>> model.fit(gen, validation_data=val_gen, epochs=40)

    Parameters
    ----------
    X          : np.ndarray  (N, seq_len, features)
    y          : np.ndarray  (N,)
    batch_size : int
    shuffle    : bool        Shuffle index order at the end of every epoch.
    seed       : int         Base random seed for reproducible shuffles.
    """

    def __init__(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        batch_size : int  = 32,
        shuffle    : bool = False,
        seed       : int  = 42,
    ):
        self.X          = X
        self.y          = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self._rng       = np.random.RandomState(seed)
        self.indices    = np.arange(len(X))
        if shuffle:
            self._rng.shuffle(self.indices)

    # ------------------------------------------------------------------ #
    # Required Sequence interface                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Number of complete batches per epoch."""
        return math.floor(len(self.X) / self.batch_size)

    def __getitem__(self, idx: int):
        """
        Return the idx-th batch as a pair of numpy arrays.
        Keras converts each batch to a GPU tensor individually.
        """
        batch_idx = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        return self.X[batch_idx], self.y[batch_idx]

    def on_epoch_end(self):
        """Re-shuffle indices at the end of each epoch (training only)."""
        if self.shuffle:
            self._rng.shuffle(self.indices)


# ===========================================================================
# JSON Serialisation Helper
# ===========================================================================

def to_python_types(d: dict) -> dict:
    """Convert numpy scalars in a dict to native Python types for json.dump."""
    clean = {}
    for key, val in d.items():
        if hasattr(val, "item"):
            clean[key] = val.item()
        elif isinstance(val, np.str_):
            clean[key] = str(val)
        else:
            clean[key] = val
    return clean
