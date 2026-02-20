"""
Data extraction module for MIMIC-III CSV files.
Loads and joins core tables for ventilator analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import sys

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import (
    DATA_DIR, VENT_ITEMIDS, VENT_ITEMIDS_CV,
    VITAL_ITEMIDS, ANTHRO_ITEMIDS, LAB_ITEMIDS,
    TBI_ICD9, NMD_ICD9, ECMO_ICD9, CARDIAC_SURGERY_ICD9,
)

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ═══════════════════════════════════════════════════════════
# 1. RAW TABLE LOADERS
# ═══════════════════════════════════════════════════════════

def load_table(name: str, data_dir: Path = DATA_DIR, **kwargs) -> pd.DataFrame:
    """Load a MIMIC-III CSV table."""
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    df = pd.read_csv(path, **kwargs)
    # Lowercase columns for consistency
    df.columns = [c.lower() for c in df.columns]
    return df


def load_patients(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and preprocess patients table."""
    pts = load_table("PATIENTS", data_dir)
    pts["dob"] = pd.to_datetime(pts["dob"])
    pts["dod"] = pd.to_datetime(pts["dod"], errors="coerce")
    pts["dod_hosp"] = pd.to_datetime(pts["dod_hosp"], errors="coerce")
    return pts


def load_admissions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and preprocess admissions table."""
    adm = load_table("ADMISSIONS", data_dir)
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in adm.columns:
            adm[col] = pd.to_datetime(adm[col], errors="coerce")
    return adm


def load_icustays(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and preprocess ICU stays table."""
    icu = load_table("ICUSTAYS", data_dir)
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    return icu


def load_chartevents(data_dir: Path = DATA_DIR,
                     itemids: Optional[List[int]] = None) -> pd.DataFrame:
    """Load chartevents, optionally filtered by itemids."""
    ce = load_table("CHARTEVENTS", data_dir,
                    dtype={"value": str, "valuenum": float})
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    if itemids is not None:
        ce = ce[ce["itemid"].isin(itemids)]
    return ce


def load_labevents(data_dir: Path = DATA_DIR,
                   itemids: Optional[List[int]] = None) -> pd.DataFrame:
    """Load lab events, optionally filtered by itemids."""
    le = load_table("LABEVENTS", data_dir,
                    dtype={"value": str, "valuenum": float})
    le["charttime"] = pd.to_datetime(le["charttime"])
    if itemids is not None:
        le = le[le["itemid"].isin(itemids)]
    return le


def load_diagnoses(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load diagnoses ICD table."""
    return load_table("DIAGNOSES_ICD", data_dir)


def load_d_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load item dictionary."""
    return load_table("D_ITEMS", data_dir)


def load_d_labitems(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load lab item dictionary."""
    return load_table("D_LABITEMS", data_dir)


def load_procedureevents(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load procedure events (MetaVision)."""
    pe = load_table("PROCEDUREEVENTS_MV", data_dir)
    for col in ["starttime", "endtime"]:
        if col in pe.columns:
            pe[col] = pd.to_datetime(pe[col], errors="coerce")
    return pe


def load_outputevents(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load output events (urine output etc.)."""
    oe = load_table("OUTPUTEVENTS", data_dir)
    oe["charttime"] = pd.to_datetime(oe["charttime"])
    return oe


def load_inputevents_mv(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load MetaVision input events (vasopressors, fluids)."""
    ie = load_table("INPUTEVENTS_MV", data_dir)
    for col in ["starttime", "endtime"]:
        if col in ie.columns:
            ie[col] = pd.to_datetime(ie[col], errors="coerce")
    return ie


def load_inputevents_cv(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load CareVue input events."""
    ie = load_table("INPUTEVENTS_CV", data_dir)
    ie["charttime"] = pd.to_datetime(ie["charttime"], errors="coerce")
    return ie


# ═══════════════════════════════════════════════════════════
# 2. ITEM ID FLATTENING
# ═══════════════════════════════════════════════════════════

def _flatten_itemids(mapping: Dict[str, List[int]]) -> List[int]:
    """Flatten a nested itemid dict into a single list."""
    ids = []
    for v in mapping.values():
        ids.extend(v)
    return ids


def _itemid_to_label(mapping: Dict[str, List[int]]) -> Dict[int, str]:
    """Invert itemid mapping: itemid -> label."""
    inv = {}
    for label, ids in mapping.items():
        for i in ids:
            inv[i] = label
    return inv


# ═══════════════════════════════════════════════════════════
# 3. COHORT CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def build_cohort(data_dir: Path = DATA_DIR,
                 min_age: int = 18,
                 min_vent_hours: float = 6,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Build the study cohort by applying inclusion/exclusion criteria.
    
    Returns a DataFrame with one row per qualifying ICU stay.
    """
    # Load core tables
    pts = load_patients(data_dir)
    adm = load_admissions(data_dir)
    icu = load_icustays(data_dir)
    diag = load_diagnoses(data_dir)

    if verbose:
        print(f"Loaded: {len(pts)} patients, {len(adm)} admissions, "
              f"{len(icu)} ICU stays")

    # ── Merge ICU stays with patient + admission info ────
    cohort = icu.merge(pts[["subject_id", "gender", "dob", "dod", "dod_hosp",
                            "expire_flag"]], on="subject_id", how="left")
    cohort = cohort.merge(adm[["hadm_id", "admittime", "dischtime", "deathtime",
                                "admission_type", "admission_location",
                                "ethnicity", "diagnosis",
                                "hospital_expire_flag"]],
                          on="hadm_id", how="left")
    if verbose:
        print(f"After merge: {len(cohort)} ICU stays")

    # ── Calculate age at admission ───────────────────────
    # Use year difference to avoid int64 overflow on MIMIC-III shifted DOBs
    cohort["age"] = (
        cohort["intime"].dt.year - cohort["dob"].dt.year
    ).astype(float)
    # MIMIC-III: patients ≥89 have shifted DOB → age >300; cap at 91.4
    cohort.loc[cohort["age"] > 200, "age"] = 91.4

    # ── Inclusion: age ≥ 18 ──────────────────────────────
    n_before = len(cohort)
    cohort = cohort[cohort["age"] >= min_age].copy()
    if verbose:
        print(f"Age ≥ {min_age}: {len(cohort)} ({n_before - len(cohort)} excluded)")

    # ── First ICU stay only ──────────────────────────────
    cohort = cohort.sort_values(["subject_id", "intime"])
    cohort["stay_rank"] = cohort.groupby("subject_id").cumcount() + 1
    n_before = len(cohort)
    cohort = cohort[cohort["stay_rank"] == 1].copy()
    if verbose:
        print(f"First ICU stay: {len(cohort)} ({n_before - len(cohort)} excluded)")

    # ── Exclusion: TBI, NMD, ECMO, cardiac surgery ──────
    if len(diag) > 0:
        excl_codes = TBI_ICD9 + NMD_ICD9 + ECMO_ICD9
        # Match prefix for broad ICD-9 categories
        excl_hadm = set()
        for _, row in diag.iterrows():
            code = str(row.get("icd9_code", ""))
            for excl in excl_codes:
                if code.startswith(excl):
                    excl_hadm.add(row["hadm_id"])
                    break
            for cs in CARDIAC_SURGERY_ICD9:
                if code.startswith(cs):
                    excl_hadm.add(row["hadm_id"])
                    break

        n_before = len(cohort)
        cohort = cohort[~cohort["hadm_id"].isin(excl_hadm)].copy()
        if verbose:
            print(f"After ICD exclusions: {len(cohort)} "
                  f"({n_before - len(cohort)} excluded: TBI/NMD/ECMO/cardiac)")

    # ── ICU LOS ──────────────────────────────────────────
    cohort["icu_los_hours"] = (
        (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
    )

    # ── Mortality outcome ────────────────────────────────
    cohort["hospital_mortality"] = cohort["hospital_expire_flag"].astype(int)
    cohort["icu_mortality"] = (
        cohort["deathtime"].notna() &
        (cohort["deathtime"] <= cohort["outtime"])
    ).astype(int)

    # ── 90-day mortality (if dod available) ──────────────
    cohort["mortality_90d"] = (
        cohort["dod"].notna() &
        ((cohort["dod"] - cohort["dischtime"]).dt.days <= 90)
    ).astype(int)

    cohort = cohort.reset_index(drop=True)
    if verbose:
        print(f"\n{'='*50}")
        print(f"FINAL COHORT: {len(cohort)} ICU stays")
        print(f"  Hospital mortality: {cohort['hospital_mortality'].mean():.1%}")
        print(f"  ICU mortality:      {cohort['icu_mortality'].mean():.1%}")
        print(f"  Median age:         {cohort['age'].median():.0f}")
        print(f"  Median ICU LOS:     {cohort['icu_los_hours'].median():.1f} hours")
        print(f"{'='*50}")

    return cohort


# ═══════════════════════════════════════════════════════════
# 4. EXTRACT VENTILATOR DATA
# ═══════════════════════════════════════════════════════════

def extract_ventilator_data(cohort: pd.DataFrame,
                            data_dir: Path = DATA_DIR,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Extract all ventilator-related chartevents for the cohort.
    Returns a tidy DataFrame with columns:
        icustay_id, charttime, parameter, value
    """
    # Combine MV + CV itemids
    all_vent_ids = (_flatten_itemids(VENT_ITEMIDS) +
                    _flatten_itemids(VENT_ITEMIDS_CV))

    # Build reverse lookup
    label_map = {**_itemid_to_label(VENT_ITEMIDS),
                 **_itemid_to_label(VENT_ITEMIDS_CV)}

    if verbose:
        print(f"Loading chartevents for {len(all_vent_ids)} ventilator itemids...")

    ce = load_chartevents(data_dir, itemids=all_vent_ids)

    # Filter to cohort
    valid_stays = set(cohort["icustay_id"].values)
    ce = ce[ce["icustay_id"].isin(valid_stays)].copy()

    # Map to canonical parameter names
    ce["parameter"] = ce["itemid"].map(label_map)

    # Use valuenum where available; parse value otherwise
    ce["val"] = ce["valuenum"]
    mask_text = ce["val"].isna() & ce["value"].notna()
    ce.loc[mask_text, "val"] = ce.loc[mask_text, "value"]

    if verbose:
        print(f"  Extracted {len(ce)} ventilator observations "
              f"for {ce['icustay_id'].nunique()} stays")
        print(f"  Parameters: {ce['parameter'].value_counts().to_dict()}")

    return ce[["icustay_id", "subject_id", "hadm_id",
               "charttime", "itemid", "parameter", "val", "valueuom"]].copy()


def extract_vitals(cohort: pd.DataFrame,
                   data_dir: Path = DATA_DIR,
                   verbose: bool = True) -> pd.DataFrame:
    """Extract vital sign chartevents for the cohort."""
    all_vital_ids = _flatten_itemids(VITAL_ITEMIDS)
    label_map = _itemid_to_label(VITAL_ITEMIDS)

    ce = load_chartevents(data_dir, itemids=all_vital_ids)
    valid_stays = set(cohort["icustay_id"].values)
    ce = ce[ce["icustay_id"].isin(valid_stays)].copy()
    ce["parameter"] = ce["itemid"].map(label_map)
    ce["val"] = ce["valuenum"]

    if verbose:
        print(f"  Extracted {len(ce)} vital sign observations "
              f"for {ce['icustay_id'].nunique()} stays")

    return ce[["icustay_id", "subject_id", "hadm_id",
               "charttime", "itemid", "parameter", "val", "valueuom"]].copy()


def extract_anthropometrics(cohort: pd.DataFrame,
                            data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Extract height and weight measurements."""
    all_ids = _flatten_itemids(ANTHRO_ITEMIDS)
    label_map = _itemid_to_label(ANTHRO_ITEMIDS)

    ce = load_chartevents(data_dir, itemids=all_ids)
    valid_stays = set(cohort["icustay_id"].values)
    ce = ce[ce["icustay_id"].isin(valid_stays)].copy()
    ce["parameter"] = ce["itemid"].map(label_map)
    ce["val"] = ce["valuenum"]
    return ce[["icustay_id", "charttime", "parameter", "val"]].copy()


def extract_labs(cohort: pd.DataFrame,
                 data_dir: Path = DATA_DIR,
                 verbose: bool = True) -> pd.DataFrame:
    """Extract laboratory values for the cohort."""
    all_lab_ids = _flatten_itemids(LAB_ITEMIDS)
    label_map = _itemid_to_label(LAB_ITEMIDS)

    le = load_labevents(data_dir, itemids=all_lab_ids)
    valid_subjects = set(cohort["subject_id"].values)
    le = le[le["subject_id"].isin(valid_subjects)].copy()
    le["parameter"] = le["itemid"].map(label_map)
    le["val"] = le["valuenum"]

    if verbose:
        print(f"  Extracted {len(le)} lab observations "
              f"for {le['subject_id'].nunique()} patients")

    return le[["subject_id", "hadm_id",
               "charttime", "itemid", "parameter", "val", "valueuom"]].copy()


# ═══════════════════════════════════════════════════════════
# 5. MASTER EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════

def extract_all(cohort: pd.DataFrame = None,
                data_dir: Path = DATA_DIR,
                min_age: int = 18,
                verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the complete data extraction pipeline.
    
    Args:
        cohort: Pre-built cohort DataFrame. If None, builds one via build_cohort().
        data_dir: Path to MIMIC-III data directory.
    
    Returns dict with keys: cohort, vent, vitals, labs, anthropometrics
    """
    print("=" * 60)
    print("MECHANICAL POWER PERSONALISATION — DATA EXTRACTION")
    print("=" * 60)

    if cohort is None:
        cohort = build_cohort(data_dir, min_age=min_age, verbose=verbose)
    else:
        if verbose:
            print(f"Using pre-built cohort: {len(cohort)} ICU stays")
    
    print("\n--- Ventilator Data ---")
    vent = extract_ventilator_data(cohort, data_dir, verbose=verbose)
    
    print("\n--- Vital Signs ---")
    vitals = extract_vitals(cohort, data_dir, verbose=verbose)
    
    print("\n--- Anthropometrics ---")
    anthro = extract_anthropometrics(cohort, data_dir)
    
    print("\n--- Laboratory Data ---")
    labs = extract_labs(cohort, data_dir, verbose=verbose)

    return {
        "cohort": cohort,
        "vent": vent,
        "vitals": vitals,
        "labs": labs,
        "anthropometrics": anthro,
    }
