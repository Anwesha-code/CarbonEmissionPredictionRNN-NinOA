# data_preprocessing.py
import pandas as pd
from config import *

def load_and_prepare_data(path):
    """
    Loads raw data, standardises column names, performs
    time-aware preprocessing, and computes energy increments (Δenergy).
    """

    df = pd.read_excel(path)

    # ================= RENAME COLUMNS TO ENGLISH =================
    column_mapping = {
        "MAC": "mac_address",
        "weekday": "weekday_raw",
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

    # ================= TIMESTAMP HANDLING =================
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)

    # ================= DROP IDENTIFIERS =================
    df.drop(columns=["mac_address", "sensor_timestamp"],
            errors="ignore", inplace=True)

    # ================= TEMPORAL FEATURES =================
    df["hour"] = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    # ================= TARGET TRANSFORMATION =================
    df["energy_delta"] = df["energy"].diff()
    df.dropna(inplace=True)

    return df


def split_data(df):
    """
    Chronological split: Train → Validation → Test
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    return train, val, test
