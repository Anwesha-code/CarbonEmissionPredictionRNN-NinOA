# core/preprocessing.py
"""
Data Loading and Preprocessing Utilities.

This module handles the complete data ingestion pipeline:
  - Loading the raw CSV with fault tolerance for malformed lines.
  - Renaming Spanish-language column names to standardised English equivalents.
  - Timestamp parsing and strict chronological sorting.
  - Forward-fill imputation and duplicate removal.
  - Z-score based outlier removal (|z| < 3) on features only.
  - Temporal feature engineering (hour, weekday, voltage lag).
  - Gap-aware multi-horizon cumulative energy increment target computation.
  - Target capping at a configurable percentile.         [NEW in v2]
  - log1p variance-stabilising transform on the target.  [NEW in v2]

Gap-Awareness
--------------
The raw dataset contains recording gaps of up to 65.8 days between consecutive
rows.  The naive formula  delta_E_k(t) = energy.shift(-k) - energy  shifts by
row index rather than by time.  When the k-th future row lies across a gap,
the computed delta equals the total energy consumed during the gap, producing
corrupted targets that are orders of magnitude larger than legitimate values.

Target Distribution Collapse (v1 issue)
-----------------------------------------
Even after the gap filter was applied, the raw energy_delta_k variable remained
extremely right-skewed:
  - k = 60 s  : mean = 6.64 kWh,  std = 52.20 kWh,  max = 417.78 kWh
  - k = 300 s : mean = 9.65 kWh,  std = 62.69 kWh,  max = 417.78 kWh
  - k = 900 s : mean = 10.09 kWh, std = 64.02 kWh,  max = 417.79 kWh

The standard deviation was 7–8 times the mean, and the maximum was 40–60
times the mean.  When MSE is minimised on such a distribution, the gradient
update mathematically drives the model towards predicting the training mean
(near zero for the bulk of samples), causing complete collapse on high-energy
events.  All three completed horizons exhibited this pattern: negative R²,
near-constant time-series predictions, and bimodal residual distributions.

Two remedies are applied here:
  1.  Percentile cap   — energy_delta_k values above TARGET_CAP_PERCENTILE
      (default 99.5th percentile) are clipped.  Any delta exceeding roughly
      50 kWh in a 60-second window is physically implausible for a workstation
      sensor and is almost certainly a residual counter-reset artefact.
  2.  log1p transform  — after capping, np.log1p is applied.  This maps the
      skewed distribution onto a near-Gaussian one, enabling MSE / Huber loss
      to function correctly.  The inverse transform (np.expm1) is applied in
      evaluate.py before metric computation.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Default gap-tolerance (can be overridden via load_and_prepare_data arguments)
# ---------------------------------------------------------------------------
GAP_TOLERANCE_FACTOR = 2.0


# ===========================================================================
# Data Loading
# ===========================================================================

def load_and_prepare_data(
    path                  : str,
    k                     : int,
    gap_factor            : float = GAP_TOLERANCE_FACTOR,
    target_cap_percentile : float = 99.5,
    log_transform         : bool  = True,
    verbose               : bool  = True,
) -> pd.DataFrame:
    """
    Load the raw sensor CSV, apply the full preprocessing pipeline, and
    compute a gap-aware k-step cumulative energy increment target.

    Parameters
    ----------
    path                  : str    Path to the raw CSV file.
    k                     : int    Forecast horizon in seconds.
    gap_factor            : float  Tolerance multiplier for the gap filter.
    target_cap_percentile : float  Percentile at which to cap extreme targets.
                                   Pass 100 to disable capping entirely.
    log_transform         : bool   If True, apply np.log1p to the target after
                                   capping.  Defaults to True.
    verbose               : bool   Print a preprocessing summary when True.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with target column 'energy_delta_k'.
        If log_transform=True the target is in log1p(kWh) space.
    """

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)

    # ------------------------------------------------------------------
    # 2. Standardise column names
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
    # 3. Parse timestamp and sort chronologically
    # ------------------------------------------------------------------
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Forward-fill missing values and drop duplicates
    # ------------------------------------------------------------------
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 5. Remove identifier columns
    # ------------------------------------------------------------------
    df.drop(
        columns=["mac_address", "sensor_timestamp"],
        errors="ignore",
        inplace=True,
    )

    # ------------------------------------------------------------------
    # 6. Temporal feature engineering
    # ------------------------------------------------------------------
    df["hour"]    = df["server_timestamp"].dt.hour
    df["weekday"] = df["server_timestamp"].dt.dayofweek

    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    # ------------------------------------------------------------------
    # 7. Gap-aware target computation
    # ------------------------------------------------------------------
    df["energy_delta_k"] = df["energy"].shift(-k) - df["energy"]

    ts_future       = df["server_timestamp"].shift(-k)
    elapsed_seconds = (ts_future - df["server_timestamp"]).dt.total_seconds()

    max_allowed_s   = gap_factor * k
    gap_mask        = elapsed_seconds.isna() | (elapsed_seconds > max_allowed_s)
    reset_mask      = df["energy_delta_k"] < 0

    n_gap_corrupted   = int(gap_mask.sum())
    n_reset_corrupted = int((reset_mask & ~gap_mask).sum())

    df.loc[gap_mask | reset_mask, "energy_delta_k"] = np.nan

    # ------------------------------------------------------------------
    # 8. Drop all rows with remaining NaN values
    # ------------------------------------------------------------------
    n_before_drop = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_after_drop = len(df)

    # ------------------------------------------------------------------
    # 9. Z-score outlier removal on FEATURE columns only
    #    (The target is handled separately via capping below.)
    # ------------------------------------------------------------------
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != "energy_delta_k"
    ]
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    n_before_zscore = len(df)
    df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)
    n_zscore_removed = n_before_zscore - len(df)

    # ------------------------------------------------------------------
    # 10. Target capping  [NEW in v2]
    #
    # Even after the gap filter removes negative deltas and time-elapsed
    # outliers, extreme positive values in the range of 400+ kWh persist
    # in the dataset.  For a 1-second-resolution workstation sensor these
    # magnitudes are physically implausible and represent residual counter-
    # reset artefacts.  Clipping at the 99.5th percentile removes the top
    # 0.5% of values without affecting legitimate high-load events.
    # ------------------------------------------------------------------
    n_capped = 0
    cap_value = np.inf

    if target_cap_percentile < 100:
        cap_value = float(
            df["energy_delta_k"].quantile(target_cap_percentile / 100.0)
        )
        df["energy_delta_k"] = df["energy_delta_k"].clip(upper=cap_value)
        if verbose:
            print(f"  Target capped at {target_cap_percentile}th pctile "
                  f"= {cap_value:.4f} kWh")

    if log_transform:
        df["energy_delta_k"] = np.log1p(df["energy_delta_k"])
        if verbose:
            print(f"  log1p transform applied  "
                  f"(new range: {df['energy_delta_k'].min():.4f} "
                  f"to {df['energy_delta_k'].max():.4f})")
            
            
    # ------------------------------------------------------------------
    # 12. Preprocessing summary
    # ------------------------------------------------------------------
    if verbose:
        t_stats = df["energy_delta_k"]
        _space  = "log1p(kWh)" if log_transform else "kWh"
        print(f"\n{'='*62}")
        print(f"  Preprocessing summary  (k = {k} s, tolerance = {gap_factor}x)")
        print(f"{'='*62}")
        print(f"  Rows in raw CSV          : {n_raw:>10,}")
        print(f"  Gap-corrupted targets    : {n_gap_corrupted:>10,}  "
              f"(elapsed > {max_allowed_s:.0f} s)")
        print(f"  Counter-reset targets    : {n_reset_corrupted:>10,}  "
              f"(delta_E < 0)")
        print(f"  Dropped for NaN          : {n_before_drop - n_after_drop:>10,}")
        print(f"  Removed by Z-score       : {n_zscore_removed:>10,}  "
              f"(features only)")
        if target_cap_percentile < 100.0:
            print(f"  Capped at {target_cap_percentile:.1f}th pctile : {n_capped:>10,}  "
                  f"(cap = {cap_value:.4f} kWh)")
        print(f"  log1p transform applied  : {'yes' if log_transform else 'no':>10}")
        print(f"  Final usable rows        : {len(df):>10,}")
        print(f"\n  Target  energy_delta_k  [{_space}]:")
        print(f"    mean  = {t_stats.mean():>12.6f}")
        print(f"    std   = {t_stats.std():>12.6f}")
        print(f"    min   = {t_stats.min():>12.6f}")
        print(f"    max   = {t_stats.max():>12.6f}")
        print(f"{'='*62}\n")

    return df


# ===========================================================================
# Gap Diagnostics
# ===========================================================================

def diagnose_gaps(
    path       : str,
    k          : int,
    gap_factor : float = GAP_TOLERANCE_FACTOR,
) -> pd.DataFrame:
    """
    Load timestamps and energy only, then report gap-by-gap statistics.

    Parameters
    ----------
    path       : str    Path to the raw CSV.
    k          : int    Forecast horizon in seconds.
    gap_factor : float  Same tolerance as load_and_prepare_data.

    Returns
    -------
    pd.DataFrame  Top 10 gaps sorted by duration.
    """
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    col_map = {
        "fecha_servidor": "server_timestamp",
        "energia"       : "energy",
    }
    df.rename(
        columns={c: n for c, n in col_map.items() if c in df.columns},
        inplace=True,
    )
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"])
    df = df.sort_values("server_timestamp").reset_index(drop=True)
    df.drop_duplicates(subset=["server_timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ts_future       = df["server_timestamp"].shift(-k)
    elapsed_seconds = (ts_future - df["server_timestamp"]).dt.total_seconds()
    max_allowed_s   = gap_factor * k

    corrupted_mask = elapsed_seconds.isna() | (elapsed_seconds > max_allowed_s)
    n_corrupted    = int(corrupted_mask.sum())
    n_total        = len(df)

    print(f"\nGap diagnostic  (k = {k} s, tolerance = {gap_factor}x = {max_allowed_s:.0f} s)")
    print(f"  Total rows        : {n_total:>10,}")
    print(f"  Corrupted targets : {n_corrupted:>10,}  "
          f"({100 * n_corrupted / n_total:.2f} %)")
    print(f"  Clean targets     : {n_total - n_corrupted:>10,}  "
          f"({100 * (n_total - n_corrupted) / n_total:.2f} %)")

    dt = df["server_timestamp"].diff().dt.total_seconds().fillna(0)
    gap_df = pd.DataFrame({
        "gap_start"      : df["server_timestamp"].values,
        "gap_duration_s" : dt.values,
    })
    gap_df["gap_duration_h"] = gap_df["gap_duration_s"] / 3600
    gap_df = (gap_df[gap_df["gap_duration_s"] > max_allowed_s]
              .sort_values("gap_duration_s", ascending=False)
              .head(10)
              .reset_index(drop=True))

    if len(gap_df) > 0:
        print(f"\n  Top gaps (duration > {max_allowed_s:.0f} s):")
        for _, row in gap_df.iterrows():
            print(f"    {row['gap_start']}   "
                  f"{row['gap_duration_s']:>12,.0f} s  "
                  f"({row['gap_duration_h']:>8.1f} h)")
    else:
        print(f"  No gaps exceeding {max_allowed_s:.0f} s found.")

    return gap_df


# ===========================================================================
# Data Splitting
# ===========================================================================

def split_by_timestamp(
    df         : pd.DataFrame,
    t_train_end: pd.Timestamp,
    t_val_end  : pd.Timestamp,
):
    """
    Strictly chronological train / validation / test split.

    Parameters
    ----------
    df          : pd.DataFrame   Cleaned dataframe with 'server_timestamp'.
    t_train_end : pd.Timestamp   Exclusive upper bound for training.
    t_val_end   : pd.Timestamp   Exclusive upper bound for validation.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train = df[df["server_timestamp"] < t_train_end].reset_index(drop=True)
    val   = df[
        (df["server_timestamp"] >= t_train_end) &
        (df["server_timestamp"] <  t_val_end)
    ].reset_index(drop=True)
    test  = df[df["server_timestamp"] >= t_val_end].reset_index(drop=True)

    return train, val, test
