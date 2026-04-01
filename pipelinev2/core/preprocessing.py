# core/preprocessing.py
"""
Data Loading and Preprocessing Utilities.

This module handles the complete data ingestion pipeline:
  - Loading the raw CSV with fault tolerance for malformed lines.
  - Renaming Spanish-language column names to standardised English equivalents.
  - Timestamp parsing and strict chronological sorting.
  - Forward-fill imputation and duplicate removal.
  - Z-score based outlier removal (|z| < 3).
  - Temporal feature engineering (hour, weekday, voltage lag).
  - Multi-horizon cumulative energy increment target computation:
        ΔEₖ(t) = E(t+k) − E(t)
  - Strictly chronological train / validation / test splitting.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Data Loading
# ===========================================================================

def load_and_prepare_data(path: str, k: int) -> pd.DataFrame:
    """
    Load the raw sensor CSV, apply the full preprocessing pipeline, and
    compute the k-step cumulative energy increment target.

    Parameters
    ----------
    path : str
        Absolute or relative path to the raw CSV file.
    k    : int
        Forecast horizon in seconds.
        The target column 'energy_delta_k' = E(t+k) − E(t).

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered dataframe ready for train/val/test split.
        Contains the target column 'energy_delta_k' with no remaining NaN rows.
    """
    # ------------------------------------------------------------------
    # 1. Load CSV — skip malformed lines to handle minor data artefacts
    # ------------------------------------------------------------------
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    # ------------------------------------------------------------------
    # 2. Standardise column names (Spanish → English)
    # ------------------------------------------------------------------
    column_mapping = {
        "MAC"                  : "mac_address",
        "fecha_servidor"       : "server_timestamp",
        "fecha_esp32"          : "sensor_timestamp",
        "voltaje"              : "voltage",
        "corriente"            : "current",
        "potencia"             : "power",
        "frecuencia"           : "frequency",
        "energia"              : "energy",
        "fp"                   : "power_factor",
        "ESP32_temp"           : "sensor_temperature",
        "ESP32_ter"            : "sensor_temperature",
        "WORKSTATION_CPU"      : "cpu_usage_percent",
        "WORKSTATION_RAM_POWER": "ram_power_watts",
    }
    df.rename(
        columns={src: tgt for src, tgt in column_mapping.items()
                 if src in df.columns},
        inplace=True,
    )

    # ------------------------------------------------------------------
    # 3. Parse timestamp and sort in strict chronological order
    # ------------------------------------------------------------------
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Forward-fill missing values (handles brief sensor dropouts)
    # ------------------------------------------------------------------
    df.ffill(inplace=True)

    # ------------------------------------------------------------------
    # 5. Drop duplicate rows arising from sensor re-transmissions
    # ------------------------------------------------------------------
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 6. Remove identifier columns that carry no predictive information
    # ------------------------------------------------------------------
    df.drop(
        columns=["mac_address", "sensor_timestamp"],
        errors="ignore",
        inplace=True,
    )

    # ------------------------------------------------------------------
    # 7. Temporal feature engineering
    # ------------------------------------------------------------------
    df["hour"]    = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    # One-step voltage lag captures short-term electrical momentum
    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    # ------------------------------------------------------------------
    # 8. Multi-horizon cumulative energy increment target
    #    ΔEₖ(t) = E(t+k) − E(t)
    # ------------------------------------------------------------------
    df["energy_delta_k"] = df["energy"].shift(-k) - df["energy"]

    # ------------------------------------------------------------------
    # 9. Drop rows containing NaN introduced by shift operations
    # ------------------------------------------------------------------
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 10. Outlier removal via Z-score (|z| < 3) on numeric feature
    #     columns.  The target is excluded intentionally — extreme
    #     energy events are valid and should not be discarded.
    # ------------------------------------------------------------------
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != "energy_delta_k"
    ]
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

    return df


# ===========================================================================
# Data Splitting
# ===========================================================================

def split_by_timestamp(
    df         : pd.DataFrame,
    t_train_end: pd.Timestamp,
    t_val_end  : pd.Timestamp,
):
    """
    Perform a strictly chronological train / validation / test split
    using pre-computed timestamp boundaries.

    This approach guarantees deployment realism: the model is never
    trained on data that post-dates any validation or test sample.

    Parameters
    ----------
    df          : pd.DataFrame   Cleaned dataframe with 'server_timestamp'.
    t_train_end : pd.Timestamp   Upper bound (exclusive) for the training set.
    t_val_end   : pd.Timestamp   Upper bound (exclusive) for the validation set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) — each with reset_index applied.
    """
    train = df[df["server_timestamp"] < t_train_end].reset_index(drop=True)
    val   = df[
        (df["server_timestamp"] >= t_train_end) &
        (df["server_timestamp"] <  t_val_end)
    ].reset_index(drop=True)
    test  = df[df["server_timestamp"] >= t_val_end].reset_index(drop=True)

    return train, val, test
