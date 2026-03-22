import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

from benchmarking.config.global_config import TARGET_COLUMN


def load_and_prepare_data(path, k):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    column_mapping = {
        "MAC": "mac_address",
        "fecha_servidor": "server_timestamp",
        "fecha_esp32": "sensor_timestamp",
        "voltaje": "voltage",
        "corriente": "current",
        "potencia": "power",
        "frecuencia": "frequency",
        "energia": "energy",
        "fp": "power_factor",
        "ESP32_temp": "sensor_temperature",
        "ESP32_ter": "sensor_temperature",
        "WORKSTATION_CPU": "cpu_usage_percent",
        "WORKSTATION_RAM_POWER": "ram_power_watts",
    }

    df.rename(
        columns={k: v for k, v in column_mapping.items() if k in df.columns},
        inplace=True
    )

    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)

    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(columns=["mac_address", "sensor_timestamp"], errors="ignore", inplace=True)

    df["hour"] = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    df[TARGET_COLUMN] = df["energy"].shift(-k) - df["energy"]

    df.dropna(inplace=True)

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != TARGET_COLUMN
    ]

    z = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    df = df[(z < 3).all(axis=1)].reset_index(drop=True)
    
    df.reset_index(drop=True, inplace=True)

    return df


def split_by_ratio(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    return train, val, test


def get_feature_columns(df, target_col=TARGET_COLUMN):
    return [
        col for col in df.columns
        if col not in ["server_timestamp", target_col]
    ]


def scale_numeric_features(train, val, test, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    return train, val, test, scaler


def create_sequences(data, target, seq_len):
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)