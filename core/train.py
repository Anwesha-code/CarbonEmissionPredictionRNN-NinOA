# core/train.py
"""
Training Utilities for NiOA Hyperparameter Optimisation.

This module provides:
  - TimeLimitCallback : Halts a NiOA trial gracefully after a wall-clock limit.
  - objective_function_lstm : The function NiOA minimises.  Uses
    SequenceDataGenerator so that even the 30%-subset optimisation arrays
    (approximately 4.5 GB for the largest horizons) are fed to the GPU one
    batch at a time rather than being copied wholesale into VRAM.

Revision notes (v2)
--------------------
objective_function_lstm now accepts loss_fn and huber_delta parameters and
forwards them to create_lstm_model.  This is necessary so that every NiOA
trial is evaluated under the same Huber loss objective as the final training
run.  If the search used a different loss internally, the hyperparameters it
identifies as optimal would not be optimal under the final objective.

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

    def __init__(self, max_seconds: int = 1800):
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
    params      : dict,
    X_train     : np.ndarray,
    y_train     : np.ndarray,
    X_val       : np.ndarray,
    y_val       : np.ndarray,
    opt_epochs  : int   = 15,
    opt_patience: int   = 3,
    time_limit  : int   = 1800,
    batch_size  : int   = 32,
    loss_fn     : str   = "huber",
    huber_delta : float = 1.0,
) -> float:
    """
    Build, train, and evaluate one candidate DRNN configuration for NiOA.

    Uses SequenceDataGenerator for both training and validation so that the
    optimisation-subset arrays are fed to the GPU one batch at a time rather
    than being copied in full into VRAM.

    Parameters
    ----------
    params       : dict         Candidate hyperparameter configuration from NiOA.
    X_train      : np.ndarray   Optimisation training sequences.
    y_train      : np.ndarray   Optimisation training targets (log1p space).
    X_val        : np.ndarray   Optimisation validation sequences.
    y_val        : np.ndarray   Optimisation validation targets (log1p space).
    opt_epochs   : int          Maximum epochs per trial.
    opt_patience : int          EarlyStopping patience during optimisation.
    time_limit   : int          Wall-clock cap in seconds per trial.
    batch_size   : int          Override batch size for the trial.
    loss_fn      : str          Loss function passed to create_lstm_model.
                                Accepted values: 'huber', 'mae', 'mse'.
                                Should match the value used in final training
                                so that NiOA evaluates candidates under the
                                same objective.
    huber_delta  : float        Delta parameter for Huber loss.

    Returns
    -------
    float   Minimum validation loss achieved across all epochs.
            Returns np.inf if the trial fails for any reason.
    """
    model = None
    bs    = params.get("batch_size", batch_size)

    try:
        # Pass loss configuration through to model construction.
        # This was the missing step in v1 — without it every NiOA trial used
        # MSE internally even though final training used Huber, meaning the
        # identified optimum was not guaranteed to be optimal under Huber loss.
        model = create_lstm_model(
            params,
            seq_len     = X_train.shape[1],
            num_feats   = X_train.shape[2],
            loss_fn     = loss_fn,
            huber_delta = huber_delta,
        )

        # SequenceDataGenerator: full arrays stay in CPU RAM;
        # only one batch is transferred to the GPU per training step.
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
