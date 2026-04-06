# core/train.py
"""
Training Utilities for NiOA Hyperparameter Optimisation.

This module provides:
  - TimeLimitCallback : Halts a NiOA trial gracefully after a wall-clock limit.
  - objective_function_lstm : The function NiOA minimises.  Uses
    SequenceDataGenerator so that even the 30%-subset optimisation arrays
    (≈ 4.5 GB for the largest horizons) are fed to GPU one batch at a time
    rather than being copied wholesale into VRAM.

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
from core.utils  import SequenceDataGenerator


# ===========================================================================
# Time-Limit Callback
# ===========================================================================

class TimeLimitCallback(tf.keras.callbacks.Callback):
    """
    Stop training after max_seconds wall-clock time.

    Used in NiOA trials to bound the cost of each hyperparameter evaluation
    and prevent a single slow configuration from stalling the search.
    """

    def __init__(self, max_seconds: int = 420):
        super().__init__()
        self.max_seconds = max_seconds
        self._t0 = None

    def on_train_begin(self, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Check at epoch granularity — avoids overhead of per-batch timing
        if time.time() - self._t0 > self.max_seconds:
            print(f"\n  [TimeLimitCallback] {self.max_seconds}s limit reached "
                  f"— stopping trial at epoch {epoch + 1}.")
            self.model.stop_training = True


# ===========================================================================
# NiOA Objective Function
# ===========================================================================

def objective_function_lstm(
    params     : dict,
    X_train    : np.ndarray,
    y_train    : np.ndarray,
    X_val      : np.ndarray,
    y_val      : np.ndarray,
    opt_epochs  : int = 15,
    opt_patience: int = 3,
    time_limit  : int = 420,
    batch_size  : int = 32,
) -> float:
    """
    Build, train, and evaluate one candidate DRNN configuration for NiOA.

    Uses SequenceDataGenerator for both training and validation so that the
    optimisation-subset arrays (which can be several GB for k=900/1800) are
    fed to the GPU one batch at a time instead of being copied in full.

    Parameters
    ----------
    params       : dict         Candidate hyperparameter configuration.
    X_train      : np.ndarray   Optimisation training sequences.
    y_train      : np.ndarray   Optimisation training targets.
    X_val        : np.ndarray   Optimisation validation sequences.
    y_val        : np.ndarray   Optimisation validation targets.
    opt_epochs   : int          Maximum epochs per trial.
    opt_patience : int          EarlyStopping patience.
    time_limit   : int          Wall-clock cap in seconds.
    batch_size   : int          Override batch size (ignores params['batch_size']
                                if set explicitly, useful for memory tuning).

    Returns
    -------
    float   Minimum validation MSE achieved across all epochs.
            Returns np.inf if the trial fails for any reason.
    """
    model = None
    bs    = params.get("batch_size", batch_size)

    try:
        model = create_lstm_model(
            params,
            seq_len   = X_train.shape[1],
            num_feats = X_train.shape[2],
        )

        # SequenceDataGenerator: full arrays stay in CPU RAM,
        # only one batch is sent to GPU per step.
        train_gen = SequenceDataGenerator(
            X_train, y_train, batch_size=bs, shuffle=True
        )
        val_gen = SequenceDataGenerator(
            X_val, y_val, batch_size=bs, shuffle=False
        )

        history = model.fit(
            train_gen,
            validation_data = val_gen,
            epochs          = opt_epochs,
            callbacks       = [
                EarlyStopping(
                    monitor              = "val_loss",
                    patience             = opt_patience,
                    restore_best_weights = True,
                    verbose              = 0,
                ),
                TimeLimitCallback(max_seconds=time_limit),
            ],
            verbose=0,
        )

        best_val_loss = float(np.min(history.history["val_loss"]))

    except Exception as exc:
        print(f"\n  [objective_function_lstm] Trial failed — {exc}")
        best_val_loss = np.inf

    finally:
        if model is not None:
            del model
        K.clear_session()
        gc.collect()

    return best_val_loss
