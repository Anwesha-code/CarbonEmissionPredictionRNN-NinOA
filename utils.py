# utils.py

import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import StandardScaler

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_sequences(data, target, seq_length):
    """
    Sliding window sequence generation
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def scale_features(train, val, test, columns):
    scaler = StandardScaler()
    scaler.fit(train[columns])

    train[columns] = scaler.transform(train[columns])
    val[columns] = scaler.transform(val[columns])
    test[columns] = scaler.transform(test[columns])

    return train, val, test, scaler
