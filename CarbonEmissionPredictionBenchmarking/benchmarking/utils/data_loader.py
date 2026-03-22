import numpy as np
import os
from benchmarking.config.global_config import SPLIT_PATH

def load_splits(horizon):

    path = os.path.join(SPLIT_PATH, f"horizon_{horizon}")

    X_train = np.load(os.path.join(path, "X_train.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    X_test = np.load(os.path.join(path, "X_test.npy"))
    y_test = np.load(os.path.join(path, "y_test.npy"))

    return X_train, y_train, X_test, y_test