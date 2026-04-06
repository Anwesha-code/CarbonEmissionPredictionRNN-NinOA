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

Gap-Awareness — Why It Matters
--------------------------------
The raw dataset contains recording gaps of up to 65.8 days between consecutive
rows (confirmed via timestampcheck.ipynb, max delta_t = 5,683,034 s).

The naive target formula  delta_E_k(t) = energy.shift(-k) - energy  shifts by
row index, not by time.  When the k-th future row lies across a recording gap,
the computed delta equals the total energy consumed during the gap — potentially
hundreds of kWh — rather than the intended k-second increment.

These corrupted values caused:
  · Predictions clustering at 0  (model learned to predict training mean)
  · Validation MSE 1672 vs training MSE 0.007  (240,000x ratio)
  · Scatter plot showing three clusters at (-400, 0), (0, 0), (+400, 0)

The fix applied here:
  1. Compute elapsed real time between row t and row t+k via timestamp shift.
  2. Invalidate target if elapsed > GAP_TOLERANCE_FACTOR * k seconds.
  3. Also invalidate if the energy counter decremented (meter reset/reboot).
  4. All invalid targets become NaN and are dropped before modelling.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Gap-tolerance configuration
# ===========================================================================

# A target is accepted only when elapsed_seconds <= GAP_TOLERANCE_FACTOR * k.
# Value of 2.0 means: for k=60 s, accept only rows where the 60th future row
# is at most 120 real seconds away.  This tolerates brief sensor dropouts while
# firmly rejecting rows that straddle multi-hour or multi-day gaps.
GAP_TOLERANCE_FACTOR = 2.0


# ===========================================================================
# Data Loading
# ===========================================================================

def load_and_prepare_data(
    path       : str,
    k          : int,
    gap_factor : float = GAP_TOLERANCE_FACTOR,
    verbose    : bool  = True,
) -> pd.DataFrame:
    """
    Load the raw sensor CSV, apply the full preprocessing pipeline, and
    compute a gap-aware k-step cumulative energy increment target.

    Parameters
    ----------
    path       : str    Absolute or relative path to the raw CSV file.
    k          : int    Forecast horizon in seconds.
    gap_factor : float  Tolerance multiplier for the gap filter (default 2.0).
    verbose    : bool   Print a preprocessing summary when True.

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered dataframe with target column
        'energy_delta_k'.  All NaN and gap-corrupted rows are removed.
    """

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)

    # ------------------------------------------------------------------
    # 2. Standardise column names (Spanish to English)
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

    # Voltage lag captures short-term electrical momentum
    if "voltage" in df.columns:
        df["voltage_lag1"] = df["voltage"].shift(1)

    # ------------------------------------------------------------------
    # 7. Gap-aware target computation
    #
    # PROBLEM (old code):
    #   energy.shift(-k) - energy  moves k rows forward in the index.
    #   If row[t+k] is separated from row[t] by a 65-day gap, the delta
    #   equals 65 days of energy consumption, not k seconds of it.
    #   This produced corrupted targets of +-400 kWh that destroyed
    #   validation performance while training loss appeared fine.
    #
    # FIX:
    #   (a) Compute the raw row-based delta as before.
    #   (b) Also shift the timestamp column by k rows to get the real
    #       elapsed time for each (t, t+k) pair.
    #   (c) Invalidate the target wherever elapsed_seconds > gap_factor * k.
    #   (d) Invalidate wherever the energy counter went negative
    #       (indicates meter replacement or system reboot).
    # ------------------------------------------------------------------

    # (a) Row-based energy delta — same formula as always
    df["energy_delta_k"] = df["energy"].shift(-k) - df["energy"]

    # (b) Real elapsed time between row t and row t+k
    ts_future       = df["server_timestamp"].shift(-k)
    elapsed_seconds = (ts_future - df["server_timestamp"]).dt.total_seconds()

    # (c) Gap filter mask
    max_allowed_s = gap_factor * k
    gap_mask = elapsed_seconds.isna() | (elapsed_seconds > max_allowed_s)

    # (d) Counter-reset mask (energy column is cumulative and must not decrease)
    reset_mask = df["energy_delta_k"] < 0

    # Count before invalidating so we can report numbers
    n_gap_corrupted   = int(gap_mask.sum())
    n_reset_corrupted = int((reset_mask & ~gap_mask).sum())

    # Set corrupted targets to NaN — they will be dropped in step 8
    df.loc[gap_mask | reset_mask, "energy_delta_k"] = np.nan

    # ------------------------------------------------------------------
    # 8. Drop all rows with any remaining NaN
    #    (from lag shifts, ffill boundary, and gap filter above)
    # ------------------------------------------------------------------
    n_before_drop = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_after_drop = len(df)

    # ------------------------------------------------------------------
    # 9. Z-score outlier removal on FEATURE columns only
    #
    #    The target is deliberately excluded:
    #      - Legitimate high-load events produce large but real delta values.
    #      - The gap filter in step 7 already handles pathological extremes.
    #      - Removing extreme targets would bias the model toward low-load
    #        periods and underestimate peak consumption.
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
    # 10. Preprocessing summary
    # ------------------------------------------------------------------
    if verbose:
        t_stats = df["energy_delta_k"]
        print(f"\n{'='*58}")
        print(f"  Preprocessing summary  (k = {k} s, tolerance = {gap_factor}x)")
        print(f"{'='*58}")
        print(f"  Rows in raw CSV          : {n_raw:>10,}")
        print(f"  Gap-corrupted targets    : {n_gap_corrupted:>10,}  "
              f"(elapsed > {max_allowed_s:.0f} s)")
        print(f"  Counter-reset targets    : {n_reset_corrupted:>10,}  "
              f"(delta_E < 0)")
        print(f"  Dropped for NaN          : {n_before_drop - n_after_drop:>10,}")
        print(f"  Removed by Z-score       : {n_zscore_removed:>10,}  "
              f"(features only)")
        print(f"  Final usable rows        : {len(df):>10,}")
        print(f"\n  Target  energy_delta_k  statistics:")
        print(f"    mean  = {t_stats.mean():>12.6f} kWh")
        print(f"    std   = {t_stats.std():>12.6f} kWh")
        print(f"    min   = {t_stats.min():>12.6f} kWh")
        print(f"    max   = {t_stats.max():>12.6f} kWh")
        print(f"    |max| = {t_stats.abs().max():>12.6f} kWh")
        print(f"{'='*58}\n")

    return df


# ===========================================================================
# Gap Diagnostics  (optional — run once in a notebook to inspect your data)
# ===========================================================================

def diagnose_gaps(
    path       : str,
    k          : int,
    gap_factor : float = GAP_TOLERANCE_FACTOR,
) -> pd.DataFrame:
    """
    Load timestamps and energy only, then show a gap-by-gap breakdown.

    Run this once in a notebook cell before the main pipeline to understand
    exactly how many rows would be invalidated by the gap filter, and where
    the largest gaps fall in the recording.

    Parameters
    ----------
    path       : str    Path to the raw CSV.
    k          : int    Forecast horizon in seconds (same as training).
    gap_factor : float  Same tolerance as load_and_prepare_data.

    Returns
    -------
    pd.DataFrame  Top 10 gaps sorted by duration, with columns:
                  gap_start, gap_duration_s, gap_duration_h.
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

    # Elapsed time for each (t, t+k) pair
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

    # Build a top-10 table of the largest individual gaps
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
    (train_df, val_df, test_df) — each with reset_index applied.
    """
    train = df[df["server_timestamp"] < t_train_end].reset_index(drop=True)
    val   = df[
        (df["server_timestamp"] >= t_train_end) &
        (df["server_timestamp"] <  t_val_end)
    ].reset_index(drop=True)
    test  = df[df["server_timestamp"] >= t_val_end].reset_index(drop=True)

    return train, val, test
