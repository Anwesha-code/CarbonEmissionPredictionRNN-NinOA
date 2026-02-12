# config.py
import os

# =========================
# Optimisation configuration
# =========================
OPT_SUBSET_RATIO = 0.30
OPT_EPOCHS = 15
OPT_PATIENCE = 3
OPT_TIME_LIMIT = 540

# Final training configuration

FINAL_EPOCHS = 40
FINAL_PATIENCE = 7


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "2agosto -dic 2021.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= TARGET =================
TARGET_COLUMN = "energy_delta"

# ================= SEQUENCE =================
SEQUENCE_LENGTH = 10

# ================= SPLIT =================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ================= REPRODUCIBILITY =================
RANDOM_SEED = 42

# ================= NINJA OA =================
N_AGENTS = 6
MAX_ITERATIONS = 6
EXPLORATION_FACTOR = 2.0
EXPLOITATION_FACTOR = 0.5


# NiOA Hyperparameter Search Space

HYPERPARAMETER_BOUNDS = {
    "lstm_layers": ([2, 3], "int"),
    "units": ([64, 128], "categorical"),
    "dropout": ([0.3, 0.6], "float"),
    "optimizer": (["adamw"], "categorical"),
    "learning_rate": ([5e-5, 5e-4], "float_log"),
    "batch_size": ([32], "categorical"),
}