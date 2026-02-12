# data_preprocessing.py
import pandas as pd
import numpy as np
from scipy import stats
from config import *

def load_and_prepare_data(path, k):
    df = pd.read_csv(
        path,
        on_bad_lines="skip",
        engine="python"
    )

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

    # Missing values
    df.ffill(inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop identifiers
    df.drop(columns=["mac_address", "sensor_timestamp"],
            errors="ignore", inplace=True)

    # Temporal features
    df["hour"] = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    # Outlier removal (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    df = df[(z < 3).all(axis=1)].reset_index(drop=True)

    # Lag feature
    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    ####################TARGET: k-step cumulative energy
    df["energy_delta_k"] = df["energy"].shift(-k) - df["energy"]

    # Drop horizon-induced NaNs BEFORE splitting
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def split_by_timestamp(df, t_train_end, t_val_end):
    """
    Fixed timestamp split (canonical)
    """
    train = df[df["server_timestamp"] < t_train_end].reset_index(drop=True)
    val = df[
        (df["server_timestamp"] >= t_train_end) &
        (df["server_timestamp"] < t_val_end)
    ].reset_index(drop=True)
    test = df[df["server_timestamp"] >= t_val_end].reset_index(drop=True)

    return train, val, test
