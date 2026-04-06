# core/config.py
"""
Master Configuration for the NiOA-DRNN Carbon Emission Prediction Pipeline.

All hyperparameters, file paths, and experimental settings are centralised
here to ensure full reproducibility across every experiment and benchmark
comparison conducted in this study.

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
OPT_TIME_LIMIT       = 420    # Wall-clock time limit per trial (7 minutes)


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
