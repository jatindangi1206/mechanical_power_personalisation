"""
Data preprocessing and feature engineering module.
Handles: outlier removal, imputation, derived variables, 
         mechanical power calculation, and hourly resampling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import PLAUSIBILITY_BOUNDS, MODE_MAP


# ═══════════════════════════════════════════════════════════
# 1. PHYSIOLOGICAL PLAUSIBILITY FILTER
# ═══════════════════════════════════════════════════════════

def remove_outliers(df: pd.DataFrame, col: str, bounds: Tuple[float, float]
                    ) -> pd.DataFrame:
    """Remove values outside physiological plausibility bounds."""
    lo, hi = bounds
    mask = df[col].between(lo, hi) | df[col].isna()
    n_removed = (~mask).sum()
    if n_removed > 0:
        df = df[mask].copy()
    return df, n_removed


def apply_plausibility_filters(df: pd.DataFrame,
                               verbose: bool = True) -> pd.DataFrame:
    """Apply all plausibility bounds to a wide-format DataFrame."""
    total_removed = 0
    for col, bounds in PLAUSIBILITY_BOUNDS.items():
        if col in df.columns:
            df, n = remove_outliers(df, col, bounds)
            total_removed += n
            if verbose and n > 0:
                print(f"  Removed {n} outliers from {col} "
                      f"(bounds: {bounds})")
    if verbose:
        print(f"  Total outliers removed: {total_removed}")
    return df


# ═══════════════════════════════════════════════════════════
# 2. PIVOT CHARTEVENTS TO WIDE FORMAT
# ═══════════════════════════════════════════════════════════

def pivot_to_wide(events: pd.DataFrame,
                  index_cols: list = ["icustay_id", "charttime"],
                  parameter_col: str = "parameter",
                  value_col: str = "val") -> pd.DataFrame:
    """
    Pivot long-format events to wide-format.
    Handles duplicates by taking the mean of numeric values.
    """
    # Ensure val is numeric
    events = events.copy()
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    
    # Drop rows where value is NaN
    events = events.dropna(subset=[value_col])
    
    # Group and take mean for duplicates
    wide = events.pivot_table(
        index=index_cols,
        columns=parameter_col,
        values=value_col,
        aggfunc="mean"
    ).reset_index()
    
    wide.columns.name = None
    return wide


# ═══════════════════════════════════════════════════════════
# 3. FIO2 STANDARDISATION
# ═══════════════════════════════════════════════════════════

def standardise_fio2(df: pd.DataFrame, col: str = "fio2") -> pd.DataFrame:
    """Convert FiO2 from percentage (21-100) to fraction (0.21-1.0)."""
    if col not in df.columns:
        return df
    df = df.copy()
    # If values are >1, assume percentage
    mask_pct = df[col] > 1.0
    df.loc[mask_pct, col] = df.loc[mask_pct, col] / 100.0
    # Clamp
    df[col] = df[col].clip(0.21, 1.0)
    return df


# ═══════════════════════════════════════════════════════════
# 4. TEMPERATURE STANDARDISATION
# ═══════════════════════════════════════════════════════════

def standardise_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all temperature to Celsius."""
    df = df.copy()
    # Merge Fahrenheit → Celsius
    if "temperature_f" in df.columns:
        if "temperature_c" not in df.columns:
            df["temperature_c"] = np.nan
        mask = df["temperature_c"].isna() & df["temperature_f"].notna()
        df.loc[mask, "temperature_c"] = (df.loc[mask, "temperature_f"] - 32) * 5 / 9
        df.drop(columns=["temperature_f"], inplace=True, errors="ignore")
    if "temperature_c" in df.columns:
        df.rename(columns={"temperature_c": "temperature"}, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════
# 5. ANTHROPOMETRICS & PREDICTED BODY WEIGHT
# ═══════════════════════════════════════════════════════════

def compute_anthropometrics(cohort: pd.DataFrame,
                            anthro_events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute height, weight, BMI, and Predicted Body Weight (PBW).
    PBW is used for VT/kg calculations (ARDSNet standard).
    """
    cohort = cohort.copy()
    
    # Get first recorded height and weight per stay
    for param, col in [("height_cm", "height_cm"), ("height_in", "height_in"),
                       ("weight_kg", "weight_kg"), ("weight_lbs", "weight_lbs")]:
        subset = anthro_events[anthro_events["parameter"] == param]
        if len(subset) > 0:
            first_val = (subset.sort_values("charttime")
                         .groupby("icustay_id")["val"]
                         .first()
                         .reset_index()
                         .rename(columns={"val": col}))
            cohort = cohort.merge(first_val, on="icustay_id", how="left")
    
    # Unify height to cm
    if "height_cm" not in cohort.columns:
        cohort["height_cm"] = np.nan
    if "height_in" in cohort.columns:
        mask = cohort["height_cm"].isna() & cohort["height_in"].notna()
        cohort.loc[mask, "height_cm"] = cohort.loc[mask, "height_in"] * 2.54
    
    # Unify weight to kg
    if "weight_kg" not in cohort.columns:
        cohort["weight_kg"] = np.nan
    if "weight_lbs" in cohort.columns:
        mask = cohort["weight_kg"].isna() & cohort["weight_lbs"].notna()
        cohort.loc[mask, "weight_kg"] = cohort.loc[mask, "weight_lbs"] * 0.4536
    
    # Filter implausible values
    cohort.loc[~cohort["height_cm"].between(120, 230), "height_cm"] = np.nan
    cohort.loc[~cohort["weight_kg"].between(30, 300), "weight_kg"] = np.nan
    
    # Impute missing height/weight by sex median
    for sex in ["M", "F"]:
        mask = (cohort["gender"] == sex) & cohort["height_cm"].isna()
        median_h = cohort.loc[cohort["gender"] == sex, "height_cm"].median()
        if pd.notna(median_h):
            cohort.loc[mask, "height_cm"] = median_h
        else:
            cohort.loc[mask, "height_cm"] = 170 if sex == "M" else 160
        
        mask = (cohort["gender"] == sex) & cohort["weight_kg"].isna()
        median_w = cohort.loc[cohort["gender"] == sex, "weight_kg"].median()
        if pd.notna(median_w):
            cohort.loc[mask, "weight_kg"] = median_w
        else:
            cohort.loc[mask, "weight_kg"] = 80 if sex == "M" else 65
    
    # BMI
    cohort["bmi"] = cohort["weight_kg"] / (cohort["height_cm"] / 100) ** 2
    
    # Predicted Body Weight (ARDSNet formula)
    # Male:   PBW = 50.0 + 0.91 × (height_cm − 152.4)
    # Female: PBW = 45.5 + 0.91 × (height_cm − 152.4)
    is_male = cohort["gender"] == "M"
    cohort["pbw"] = np.where(
        is_male,
        50.0 + 0.91 * (cohort["height_cm"] - 152.4),
        45.5 + 0.91 * (cohort["height_cm"] - 152.4)
    )
    cohort["pbw"] = cohort["pbw"].clip(lower=30)
    
    return cohort


# ═══════════════════════════════════════════════════════════
# 6. MECHANICAL POWER CALCULATION
# ═══════════════════════════════════════════════════════════

def calculate_mechanical_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Mechanical Power using the simplified Gattinoni 2016 formula:
        MP = 0.098 × RR × VT × (Ppeak − 0.5 × ΔP)
    where ΔP = Ppeak − PEEP
    
    Also calculates driving pressure and compliance when Pplat is available.
    """
    df = df.copy()
    
    # Resolve tidal volume: prefer observed > set
    if "tidal_volume_observed" in df.columns and "tidal_volume_set" in df.columns:
        df["tidal_volume"] = df["tidal_volume_observed"].fillna(
            df["tidal_volume_set"])
    elif "tidal_volume_observed" in df.columns:
        df["tidal_volume"] = df["tidal_volume_observed"]
    elif "tidal_volume_set" in df.columns:
        df["tidal_volume"] = df["tidal_volume_set"]
    elif "tidal_volume" not in df.columns:
        df["tidal_volume"] = np.nan
    
    # Resolve respiratory rate: prefer total > set > base
    rr_cols = ["respiratory_rate_total", "respiratory_rate_set", "respiratory_rate"]
    df["rr"] = np.nan
    for c in rr_cols:
        if c in df.columns:
            df["rr"] = df["rr"].fillna(df[c])
    
    # Resolve PEEP: prefer set > total
    peep_cols = ["peep_set", "total_peep", "peep"]
    df["peep_final"] = np.nan
    for c in peep_cols:
        if c in df.columns:
            df["peep_final"] = df["peep_final"].fillna(df[c])
    
    # Peak pressure
    if "peak_pressure" not in df.columns:
        df["peak_pressure"] = np.nan
    
    # Plateau pressure
    if "plateau_pressure" not in df.columns:
        df["plateau_pressure"] = np.nan
    
    # ── Driving Pressure ──
    # ΔP = Plateau_pressure − PEEP (when Pplat available)
    # Fallback: ΔP = Peak_pressure − PEEP (includes airway resistance)
    df["driving_pressure"] = np.nan
    mask_pplat = df["plateau_pressure"].notna() & df["peep_final"].notna()
    df.loc[mask_pplat, "driving_pressure"] = (
        df.loc[mask_pplat, "plateau_pressure"] - df.loc[mask_pplat, "peep_final"]
    )
    mask_ppeak = df["driving_pressure"].isna() & df["peak_pressure"].notna() & df["peep_final"].notna()
    df.loc[mask_ppeak, "driving_pressure"] = (
        df.loc[mask_ppeak, "peak_pressure"] - df.loc[mask_ppeak, "peep_final"]
    )
    
    # ── Static Compliance ──
    mask_compliance = (df["tidal_volume"].notna() &
                       df["driving_pressure"].notna() &
                       (df["driving_pressure"] > 0))
    df["compliance_static"] = np.nan
    df.loc[mask_compliance, "compliance_static"] = (
        df.loc[mask_compliance, "tidal_volume"] /
        df.loc[mask_compliance, "driving_pressure"]
    )
    
    # ── Mechanical Power (simplified) ──
    # MP = 0.098 × RR × VT(L) × (Ppeak − 0.5 × (Ppeak − PEEP))
    # Simplifies to: MP = 0.098 × RR × VT(L) × (0.5 × Ppeak + 0.5 × PEEP)
    # But more commonly: MP = 0.098 × RR × VT × (Ppeak − ΔP/2)
    mask_mp = (df["rr"].notna() & df["tidal_volume"].notna() &
               df["peak_pressure"].notna() & df["peep_final"].notna())
    
    vt_liters = df["tidal_volume"] / 1000.0  # mL → L
    delta_p = df["peak_pressure"] - df["peep_final"]
    
    df["mechanical_power"] = np.nan
    df.loc[mask_mp, "mechanical_power"] = (
        0.098 * df.loc[mask_mp, "rr"] *
        vt_liters.loc[mask_mp] *
        (df.loc[mask_mp, "peak_pressure"] - 0.5 * delta_p.loc[mask_mp])
    )
    
    # Filter negative/implausible MP
    df.loc[df["mechanical_power"] < 0, "mechanical_power"] = np.nan
    df.loc[df["mechanical_power"] > 100, "mechanical_power"] = np.nan
    
    return df


# ═══════════════════════════════════════════════════════════
# 7. DERIVE CLINICAL VARIABLES
# ═══════════════════════════════════════════════════════════

def derive_clinical_variables(df: pd.DataFrame,
                              cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived clinical variables:
    - P/F ratio, S/F ratio
    - VT per PBW
    - Normalised MP (per kg PBW)
    - ARDS severity
    - GCS total
    """
    df = df.copy()
    
    # Merge PBW from cohort
    if "pbw" not in df.columns and "icustay_id" in df.columns:
        pbw_map = cohort.set_index("icustay_id")["pbw"].to_dict()
        df["pbw"] = df["icustay_id"].map(pbw_map)
    
    # VT per PBW (mL/kg)
    if "tidal_volume" in df.columns and "pbw" in df.columns:
        df["vt_per_pbw"] = df["tidal_volume"] / df["pbw"]
    
    # Normalised MP (J/min/kg PBW)
    if "mechanical_power" in df.columns and "pbw" in df.columns:
        df["mp_per_pbw"] = df["mechanical_power"] / df["pbw"]
    
    # P/F ratio
    if "pao2" in df.columns and "fio2" in df.columns:
        mask = df["fio2"] > 0
        df["pf_ratio"] = np.nan
        df.loc[mask, "pf_ratio"] = df.loc[mask, "pao2"] / df.loc[mask, "fio2"]
    
    # S/F ratio
    if "spo2" in df.columns and "fio2" in df.columns:
        mask = df["fio2"] > 0
        df["sf_ratio"] = np.nan
        df.loc[mask, "sf_ratio"] = df.loc[mask, "spo2"] / df.loc[mask, "fio2"]
    
    # ARDS severity (Berlin Definition on P/F ratio)
    if "pf_ratio" in df.columns:
        conditions = [
            df["pf_ratio"] <= 100,
            df["pf_ratio"] <= 200,
            df["pf_ratio"] <= 300,
            df["pf_ratio"] > 300
        ]
        labels = [3, 2, 1, 0]  # severe=3, moderate=2, mild=1, none=0
        df["ards_severity"] = np.select(conditions, labels, default=np.nan)
    
    # GCS total (sum of eye + verbal + motor)
    gcs_cols = [c for c in df.columns if c.startswith("gcs_")]
    if len(gcs_cols) >= 3:
        df["gcs_total"] = df[gcs_cols].sum(axis=1, min_count=3)
    
    return df


# ═══════════════════════════════════════════════════════════
# 8. VENTILATOR MODE HARMONISATION
# ═══════════════════════════════════════════════════════════

def harmonise_vent_mode(vent_data: pd.DataFrame) -> pd.DataFrame:
    """Map raw ventilator mode strings to canonical categories."""
    df = vent_data.copy()
    if "ventilator_mode" in df.columns:
        df["vent_mode_canonical"] = df["ventilator_mode"].map(MODE_MAP).fillna("other")
    return df


# ═══════════════════════════════════════════════════════════
# 9. HOURLY RESAMPLING
# ═══════════════════════════════════════════════════════════

def resample_hourly(df: pd.DataFrame,
                    icustay_col: str = "icustay_id",
                    time_col: str = "charttime",
                    numeric_agg: str = "mean",
                    forward_fill_hours: int = 6) -> pd.DataFrame:
    """
    Resample time-series data to hourly intervals per ICU stay.
    Forward-fills missing values up to `forward_fill_hours`.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Round to nearest hour
    df["hour"] = df[time_col].dt.floor("h")
    
    # Get numeric columns
    exclude = {icustay_col, time_col, "hour", "subject_id", "hadm_id",
               "itemid", "parameter", "valueuom"}
    numeric_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Aggregate by stay + hour
    hourly = (df.groupby([icustay_col, "hour"])[numeric_cols]
              .agg(numeric_agg)
              .reset_index())
    
    # Create complete hourly grid per stay
    result_frames = []
    for stay_id, group in hourly.groupby(icustay_col):
        min_hour = group["hour"].min()
        max_hour = group["hour"].max()
        full_range = pd.date_range(min_hour, max_hour, freq="h")
        full_df = pd.DataFrame({"hour": full_range, icustay_col: stay_id})
        merged = full_df.merge(group, on=[icustay_col, "hour"], how="left")
        
        # Forward fill (limited)
        for col in numeric_cols:
            merged[col] = merged[col].ffill(limit=forward_fill_hours)
        
        result_frames.append(merged)
    
    if len(result_frames) == 0:
        return pd.DataFrame()
    
    result = pd.concat(result_frames, ignore_index=True)
    return result


# ═══════════════════════════════════════════════════════════
# 10. MASTER PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

def preprocess_pipeline(extracted: Dict[str, pd.DataFrame],
                        cohort: pd.DataFrame = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        extracted: dict from data_extraction.extract_all()
        cohort: optional cohort DataFrame (if not in extracted)
    
    Returns:
        hourly_features DataFrame
    """
    if cohort is None:
        cohort = extracted["cohort"]
    vent = extracted["vent"]
    vitals = extracted["vitals"]
    labs = extracted["labs"]
    anthro = extracted["anthropometrics"]

    if verbose:
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

    # ── Step 1: Anthropometrics ──────────────────────────
    if verbose:
        print("\n[1/7] Computing anthropometrics (height, weight, PBW)...")
    cohort = compute_anthropometrics(cohort, anthro)

    # ── Step 2: Pivot vent data to wide format ───────────
    if verbose:
        print("[2/7] Pivoting ventilator data to wide format...")
    vent_numeric = vent[vent["parameter"] != "ventilator_mode"].copy()
    vent_numeric["val"] = pd.to_numeric(vent_numeric["val"], errors="coerce")
    vent_wide = pivot_to_wide(vent_numeric)

    # Handle vent mode separately
    vent_mode = vent[vent["parameter"] == "ventilator_mode"].copy()
    if len(vent_mode) > 0:
        vent_mode_first = (vent_mode.sort_values("charttime")
                           .groupby(["icustay_id", pd.Grouper(key="charttime", freq="h")])
                           ["val"].first()
                           .reset_index()
                           .rename(columns={"val": "ventilator_mode"}))
        vent_wide = vent_wide.merge(
            vent_mode_first, on=["icustay_id", "charttime"], how="left"
        )

    if verbose:
        print(f"  Vent wide shape: {vent_wide.shape}")

    # ── Step 3: Pivot vitals to wide format ──────────────
    if verbose:
        print("[3/7] Pivoting vitals to wide format...")
    vitals_wide = pivot_to_wide(vitals)
    if verbose:
        print(f"  Vitals wide shape: {vitals_wide.shape}")

    # ── Step 4: Merge vent + vitals ──────────────────────
    if verbose:
        print("[4/7] Merging ventilator + vitals on hourly grid...")
    
    # Round to hour for merging
    vent_wide["hour"] = pd.to_datetime(vent_wide["charttime"]).dt.floor("h")
    vitals_wide["hour"] = pd.to_datetime(vitals_wide["charttime"]).dt.floor("h")
    
    # Aggregate to hourly level
    exclude_cols = {"icustay_id", "charttime", "hour"}
    vent_num_cols = [c for c in vent_wide.columns
                     if c not in exclude_cols and c != "ventilator_mode"]
    vitals_num_cols = [c for c in vitals_wide.columns if c not in exclude_cols]
    
    vent_hourly = (vent_wide.groupby(["icustay_id", "hour"])
                   [vent_num_cols].mean().reset_index())
    vitals_hourly = (vitals_wide.groupby(["icustay_id", "hour"])
                     [vitals_num_cols].mean().reset_index())
    
    # Merge
    combined = vent_hourly.merge(
        vitals_hourly, on=["icustay_id", "hour"], how="outer"
    ).sort_values(["icustay_id", "hour"])
    
    if verbose:
        print(f"  Combined shape: {combined.shape}")

    # ── Step 5: FiO2 and Temperature standardisation ─────
    if verbose:
        print("[5/7] Standardising FiO2 and temperature...")
    combined = standardise_fio2(combined)
    combined = standardise_temperature(combined)

    # ── Step 6: Calculate Mechanical Power ───────────────
    if verbose:
        print("[6/7] Calculating mechanical power and derived variables...")
    combined = calculate_mechanical_power(combined)
    combined = derive_clinical_variables(combined, cohort)

    # ── Step 7: Plausibility filter + forward fill ───────
    if verbose:
        print("[7/7] Applying plausibility filters and forward-filling...")
    combined = apply_plausibility_filters(combined, verbose=verbose)
    
    # Forward fill per stay (limited)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "icustay_id"]
    
    for col in numeric_cols:
        combined[col] = combined.groupby("icustay_id")[col].transform(
            lambda x: x.ffill(limit=6)
        )

    if verbose:
        print(f"\n{'='*50}")
        print(f"PREPROCESSED DATA")
        print(f"  Stays: {combined['icustay_id'].nunique()}")
        print(f"  Hourly observations: {len(combined)}")
        print(f"  Features: {len(combined.columns)}")
        mp_avail = combined["mechanical_power"].notna().sum()
        print(f"  MP available: {mp_avail} observations "
              f"({mp_avail/len(combined)*100:.1f}%)")
        print(f"{'='*50}")

    return combined
