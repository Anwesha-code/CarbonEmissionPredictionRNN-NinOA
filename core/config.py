# core/config.py
"""
Master Configuration for the NiOA-DRNN Carbon Emission Prediction Pipeline.

All hyperparameters, file paths, and experimental settings are centralised
here to ensure full reproducibility across every experiment and benchmark
comparison conducted in this study.

Three corrective measures are introduced:
  1.  Target capping  — residual extreme values beyond TARGET_CAP_PERCENTILE
      are clipped to that percentile value before sequence generation.
  2.  log1p transform — after capping, np.log1p is applied to the target so
      that the model learns on a near-Gaussian distribution.  Predictions are
      inverted with np.expm1 at evaluation time.
  3.  Huber / MAE loss — the loss function is switched from MSE to Huber
      (delta = 1.0) which is linear for large residuals and therefore far more
      robust to the residual tail that persists even after capping.
  4.  Increased NiOA trial time — OPT_TIME_LIMIT is raised to 1800 s so that
      each trial can complete at least 3-4 epochs and provide a meaningful
      hyperparameter signal to the optimiser.
      
Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import os
import sys

# ===========================================================================
# Path Configuration
# ===========================================================================
# BASE_DIR  → the 'core/' directory containing this file
# PROJECT_ROOT → top-level project directory (one level above 'core/')
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Ensure both are importable from notebooks
for _p in [BASE_DIR, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "2agosto -dic 2021.csv")

# ---------------------------------------------------------------------------
# Results directories
# ---------------------------------------------------------------------------
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
SPLITS_DIR       = os.path.join(RESULTS_DIR, "splits")
NIOA_RESULTS_DIR = os.path.join(RESULTS_DIR, "nioa_drnn")
BENCHMARK_DIR    = os.path.join(RESULTS_DIR, "benchmark")

for _d in [RESULTS_DIR, SPLITS_DIR, NIOA_RESULTS_DIR, BENCHMARK_DIR]:
    os.makedirs(_d, exist_ok=True)


# ===========================================================================
# Multi-Horizon Configuration
# ===========================================================================
# Forecast horizons (k) in seconds.
# ΔEₖ(t) = E(t+k) − E(t) is the prediction target for each horizon.
FORECAST_HORIZONS = [1, 60, 300, 900, 1800]

# ===========================================================================
# Gap Filter Configuration
# ===========================================================================
# A target row is kept only when the real elapsed time between row t and
# row t+k is <= GAP_TOLERANCE_FACTOR * k seconds.
#
# The dataset contains gaps up to 65.8 days (5,683,034 s).  Without this
# filter, energy.shift(-k) computes multi-day deltas for rows near gaps,
# corrupting the target and causing validation MSE to be ~240,000x higher
# than training MSE (the root cause of the flat predictions seen previously).
#
# 2.0 tolerates brief sensor dropouts (≤ 2× the horizon) while firmly
# rejecting multi-hour or multi-day gap artefacts.  Adjust in range [1.5, 3.0]
# after running notebooks/00_Gap_Diagnostics.ipynb on your specific dataset.
GAP_TOLERANCE_FACTOR = 2.0


# ===========================================================================
# Target Preprocessing  (NEW in v2)
# ===========================================================================
# TARGET_CAP_PERCENTILE
#   Energy increments above this percentile of the full dataset are clipped
#   to the cap value before sequence generation.  A cap at the 99.5th
#   percentile removes the top 0.5% of extreme readings (likely residual
#   meter-counter artefacts that the gap filter could not eliminate) while
#   preserving all legitimate high-load events.
#   Set to 100 to disable capping entirely.
TARGET_CAP_PERCENTILE = 99.5

# TARGET_LOG_TRANSFORM
#   When True, np.log1p is applied to energy_delta_k after capping.
#   This stabilises the variance, compresses the dynamic range, and renders
#   the target distribution approximately symmetric — a necessary precondition
#   for MSE / Huber loss to function correctly on this dataset.
#   np.expm1 is applied automatically at evaluation time to restore original
#   kWh units for metric reporting.
TARGET_LOG_TRANSFORM = True


# ===========================================================================
# Loss Function Configuration  (NEW in v2)
# ===========================================================================
# LOSS_FUNCTION
#   Accepted values: 'huber', 'mae', 'mse'
#   Huber loss combines the squared penalty of MSE for small residuals with
#   the linear penalty of MAE for large residuals.  After log1p transform the
#   residuals are well-behaved, but Huber provides an additional safeguard
#   against any residual tail.  MAE is similarly robust but may train more
#   slowly.  MSE is retained as an option for ablation studies.
LOSS_FUNCTION = "huber"

# HUBER_DELTA
#   The threshold at which Huber loss transitions from quadratic to linear.
#   In log1p-transformed space, a delta of 1.0 corresponds to a real-valued
#   energy increment of approximately e^1 − 1 ≈ 1.72 kWh.  Below this
#   threshold the loss is squared; above it the loss is linear.
HUBER_DELTA = 1.0

# ===========================================================================
# Train / Validation / Test Split Ratios
# ===========================================================================
TRAIN_RATIO = 0.70   # 70 % — used to fit the model
VAL_RATIO   = 0.15   # 15 % — used for early stopping and NiOA evaluation
# Test ratio = 1 − TRAIN_RATIO − VAL_RATIO = 0.15 (15 %) — held out entirely


# ===========================================================================
# Sequence (Sliding-Window) Configuration
# ===========================================================================
SEQUENCE_LENGTH = 120   # 120 rows ≈ 2 minutes of historical context


# ===========================================================================
# Reproducibility
# ===========================================================================
RANDOM_SEED = 42


# ===========================================================================
# NiOA Hyperparameter Optimisation Settings
# ===========================================================================
N_AGENTS             = 6      # Population size (number of ninja agents)
MAX_ITERATIONS       = 6      # Refinement iterations after initial evaluation
EXPLORATION_FACTOR   = 2.0    # Controls breadth of random perturbations
EXPLOITATION_FACTOR  = 0.5    # Attraction strength towards the best agent
OPT_SUBSET_RATIO     = 0.30   # Fraction of training data used in NiOA trials
OPT_EPOCHS           = 15     # Maximum epochs per NiOA trial
OPT_PATIENCE         = 3      # Early-stopping patience during NiOA
OPT_TIME_LIMIT       = 1800    
# OPT_TIME_LIMIT  (INCREASED in v2: 420 → 1800 s)
# The previous limit of 420 s allowed only 1–2 epochs per trial on a 15 GB
# dataset, making hyperparameter comparison effectively meaningless.  With
# 1800 s per trial each configuration can complete 3–5 epochs and exhibit
# genuine convergence behaviour before the timer fires.

# ===========================================================================
# Final Training Configuration
# ===========================================================================
FINAL_EPOCHS   = 40
FINAL_PATIENCE = 7


# ===========================================================================
# NiOA Hyperparameter Search Space
# ===========================================================================
# Each entry: parameter_name → ([bounds_or_choices], type_string)
# Types:
#   'int'         — uniform integer between bounds[0] and bounds[1] (inclusive)
#   'float'       — uniform float between bounds[0] and bounds[1]
#   'float_log'   — log-uniform float (perturbations applied in log₁₀ space)
#   'categorical' — one of the listed choices
HYPERPARAMETER_BOUNDS = {
    "lstm_layers"   : ([2, 3],       "int"),
    "units"         : ([64, 128],    "categorical"),
    "dropout"       : ([0.3, 0.6],   "float"),
    "optimizer"     : (["adamw"],    "categorical"),
    "learning_rate" : ([5e-5, 5e-4], "float_log"),
    "batch_size"    : ([32],         "categorical"),
}
