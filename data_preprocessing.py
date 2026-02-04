# data_preprocessing.py
import pandas as pd
import numpy as np
from scipy import stats
from config import *

def load_and_prepare_data(path):
    """
    Robust, time-aware preprocessing with English column names
    and Δenergy target for non-stationary mitigation.
    """

    df = pd.read_csv(
    path,
    on_bad_lines='skip',
    engine='python'
    )



    # ================= RENAME COLUMNS =================
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
        "WORKSTATION_RAM_POWER": "ram_power_watts"
    }

    df.rename(
        columns={k: v for k, v in column_mapping.items() if k in df.columns},
        inplace=True
    )

    # ================= TIMESTAMP =================
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)

    # ================= MISSING VALUES =================
    df.ffill(inplace=True)

    # ================= DUPLICATES =================
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ================= DROP IDENTIFIERS =================
    df.drop(columns=["mac_address", "sensor_timestamp"],
            errors="ignore", inplace=True)

    # ================= TEMPORAL FEATURES =================
    df["hour"] = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    # ================= OUTLIER REMOVAL =================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

    # ================= LAG FEATURE =================
    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    # ================= TARGET (Δenergy) =================
    df["energy_delta"] = df["energy"].diff()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def split_data(df):
    """
    Strict chronological split: Train → Validation → Test
    """

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    return train, val, test

