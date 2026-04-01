# core/train.py
"""
Training Utilities for NiOA Hyperparameter Optimisation.

This module provides:
  - TimeLimitCallback : A Keras callback that halts training gracefully when a
    per-trial wall-clock limit is exceeded, preventing runaway NiOA trials from
    occupying the GPU indefinitely.
  - objective_function_lstm : The function that NiOA minimises.  It builds a
    fresh DRNN for each candidate hyperparameter configuration, trains it on
    the optimisation subset, and returns the minimum validation MSE achieved.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import gc
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from core.models import create_lstm_model


# ===========================================================================
# Time-Limit Callback
# ===========================================================================

class TimeLimitCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that stops training once a configurable wall-clock
    time limit has elapsed.

    This is particularly useful during NiOA optimisation where each
    trial is expected to complete within a bounded time window, so that
    the total search remains tractable even on large datasets.

    Parameters
    ----------
    max_seconds : int   Maximum allowed training duration in seconds.
    """

    def __init__(self, max_seconds: int = 420):
        super().__init__()
        self.max_seconds = max_seconds
        self._start_time = None

    def on_train_begin(self, logs=None):
        self._start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if time.time() - self._start_time > self.max_seconds:
            self.model.stop_training = True


# ===========================================================================
# NiOA Objective Function
# ===========================================================================

def objective_function_lstm(
    params  : dict,
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_val   : np.ndarray,
    y_val   : np.ndarray,
    opt_epochs   : int = 15,
    opt_patience : int = 3,
    time_limit   : int = 420,
) -> float:
    """
    Objective function minimised by NiOA during hyperparameter optimisation.

    For a given hyperparameter configuration, this function:
      1. Constructs a fresh DRNN model.
      2. Trains it on the optimisation subset with early stopping and a
         wall-clock time limit.
      3. Returns the minimum validation MSE observed across all epochs.

    If training fails for any reason (e.g., GPU OOM), a sentinel value of
    np.inf is returned so that NiOA simply discards the failed configuration.

    Parameters
    ----------
    params       : dict         Hyperparameter dictionary from NiOA.
    X_train      : np.ndarray   Training sequences (opt subset), shape
                                (N, seq_len, n_features).
    y_train      : np.ndarray   Training targets,  shape (N,).
    X_val        : np.ndarray   Validation sequences (opt subset).
    y_val        : np.ndarray   Validation targets.
    opt_epochs   : int          Maximum training epochs per NiOA trial.
    opt_patience : int          Early-stopping patience.
    time_limit   : int          Wall-clock limit per trial (seconds).

    Returns
    -------
    float   Minimum validation MSE (lower is better).
    """
    model = None

    try:
        model = create_lstm_model(
            params,
            seq_len  = X_train.shape[1],
            num_feats= X_train.shape[2],
        )

        callbacks = [
            EarlyStopping(
                monitor             = "val_loss",
                patience            = opt_patience,
                restore_best_weights= True,
                verbose             = 0,
            ),
            TimeLimitCallback(max_seconds=time_limit),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = opt_epochs,
            batch_size      = params["batch_size"],
            callbacks       = callbacks,
            verbose         = 0,
        )

        best_val_loss = float(np.min(history.history["val_loss"]))

    except Exception as exc:
        print(f"\n  [objective_function_lstm] Trial failed — {exc}")
        print(f"  Params: {params}")
        best_val_loss = np.inf

    finally:
        if model is not None:
            del model
        K.clear_session()
        gc.collect()

    return best_val_loss
