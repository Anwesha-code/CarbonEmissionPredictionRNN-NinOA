# config.py
import os

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
N_AGENTS = 12
MAX_ITERATIONS = 12
EXPLORATION_FACTOR = 2.0
EXPLOITATION_FACTOR = 0.5
