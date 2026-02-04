# utils.py
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import StandardScaler

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def scale_numeric_features(train, val, test, feature_cols):
    """
    Scale numeric features using a single scaler
    fitted on the training set only.
    """

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    return train, val, test, scaler


def create_sequences(data, target, seq_len):
    """
    Sliding-window sequence generation for RNN/LSTM.
    """

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
