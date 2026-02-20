"""
Configuration for Mechanical Power Personalisation Project.
Adapted for MIMIC-III Demo Database.
"""
import os
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get(
    "MIMIC_DATA_DIR",
    "/Users/jatindangi/Desktop/TSB:KCDH/mp1/mechanical_power/mimic-iii-clinical-database-demo-1.4"
))
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
ARTEFACTS_DIR.mkdir(exist_ok=True)

# ─── MIMIC-III ItemID Mappings ─────────────────────────────
# MetaVision ItemIDs (MIMIC-III / IV)
VENT_ITEMIDS = {
    "tidal_volume_set":       [224684],
    "tidal_volume_observed":  [224685],
    "tidal_volume_spont":     [224686],
    "respiratory_rate":       [220210],
    "respiratory_rate_set":   [224688],
    "respiratory_rate_total": [224690],
    "peep_set":               [220339],
    "total_peep":             [224700],
    "fio2":                   [223835],
    "peak_pressure":          [224695],
    "plateau_pressure":       [224696],
    "mean_airway_pressure":   [224697],
    "minute_ventilation":     [224687],
    "ventilator_mode":        [223849],
}

# CareVue ItemIDs (older MIMIC-III patients)
VENT_ITEMIDS_CV = {
    "tidal_volume":           [681, 682],
    "tidal_volume_set":       [683],
    "respiratory_rate":       [618],
    "respiratory_rate_set":   [619],
    "peep":                   [505, 506],
    "fio2":                   [190, 191, 3420],
    "peak_pressure":          [535],
    "plateau_pressure":       [543],
    "mean_airway_pressure":   [444],
    "minute_ventilation":     [445, 448, 450],
    "ventilator_mode":        [720],
}

VITAL_ITEMIDS = {
    "heart_rate":             [220045, 211],
    "sbp":                    [220050, 220179, 51],
    "dbp":                    [220051, 220180, 8368],
    "map":                    [220052, 220181, 52],
    "spo2":                   [220277, 646],
    "temperature_f":          [223761, 678],
    "temperature_c":          [223762, 676],
    "gcs_eye":                [220739, 184],
    "gcs_verbal":             [223900, 723],
    "gcs_motor":              [223901, 454],
}

ANTHRO_ITEMIDS = {
    "height_cm":              [226730],
    "height_in":              [226707],
    "weight_kg":              [226512, 224639],
    "weight_lbs":             [226531],
}

# Lab Item IDs (from D_LABITEMS)
LAB_ITEMIDS = {
    "ph":           [50820],
    "pao2":         [50821],
    "paco2":        [50818],
    "hco3":         [50882],
    "base_excess":  [50802],
    "lactate":      [50813],
    "creatinine":   [50912],
    "bun":          [51006],
    "bilirubin":    [50885],
    "platelet":     [51265],
    "wbc":          [51301],
    "hemoglobin":   [51222],
    "sodium":       [50983],
    "potassium":    [50971],
    "albumin":      [50862],
}

# ─── Action Space ─────────────────────────────────────────
ACTIONS = {
    0: "large_decrease",    # -5 J/min
    1: "small_decrease",    # -2 J/min
    2: "maintain",          #  0 J/min
    3: "small_increase",    # +2 J/min
    4: "large_increase",    # +5 J/min
}
ACTION_DELTAS = {0: -5.0, 1: -2.0, 2: 0.0, 3: +2.0, 4: +5.0}
N_ACTIONS = len(ACTIONS)

# ─── Physiological Plausibility Bounds ────────────────────
PLAUSIBILITY_BOUNDS = {
    "tidal_volume":       (50, 1500),
    "respiratory_rate":   (5, 50),
    "peep":               (0, 30),
    "plateau_pressure":   (5, 60),
    "peak_pressure":      (5, 80),
    "spo2":               (50, 100),
    "fio2":               (0.21, 1.0),
    "heart_rate":         (20, 250),
    "map":                (20, 200),
    "mechanical_power":   (0, 100),
    "ph":                 (6.5, 7.8),
    "pao2":               (20, 600),
    "paco2":              (10, 150),
    "temperature":        (30, 42),
}

# ─── Ventilator Mode Harmonisation ────────────────────────
MODE_MAP = {
    "CMV":                              "AC_VC",
    "CMV/ASSIST":                       "AC_VC",
    "CMV/ASSIST/AutoFlow":              "AC_VC",
    "Assist Control-Volume Control":    "AC_VC",
    "SIMV":                             "SIMV",
    "SIMV/P":                           "SIMV",
    "SIMV/PSV":                         "SIMV",
    "SIMV/PSV/AutoFlow":                "SIMV",
    "PRVC/AC":                          "AC_PRVC",
    "Pressure Regulated Volume Control":"AC_PRVC",
    "PCV+Assist":                       "AC_PC",
    "PCV+":                             "AC_PC",
    "Assist Control-Pressure Control":  "AC_PC",
    "CPAP":                             "PS_CPAP",
    "CPAP/PSV":                         "PS_CPAP",
    "Pressure Support/CPAP":            "PS_CPAP",
    "PSV/SBT":                          "PS_CPAP",
    "APRV":                             "APRV",
    "Airway Pressure Release Ventilation":"APRV",
    "Standby":                          "STANDBY",
    "(S) CMV":                          "AC_VC",
    "(S) CMV/AutoFlow":                 "AC_VC",
}

# ─── RL / Training Config ─────────────────────────────────
RL_CONFIG = {
    "gamma":             0.99,
    "lr":                3e-4,
    "learning_rate":     3e-4,      # alias
    "batch_size":        256,       # Smaller for demo data
    "n_epochs":          100,
    "cql_alpha":         1.0,       # CQL conservative penalty
    "alpha":             1.0,       # alias
    "hidden_dim":        256,
    "hidden_dims":       [256, 256],
    "dropout":           0.3,
    "tau":               0.005,
    "lookback_hours":    24,
    "min_vent_hours":    6,         # Reduced for demo (was 24)
    "state_dim":         None,      # Set dynamically
    "n_actions":         N_ACTIONS,
}

# ─── Reward Function Weights ──────────────────────────────
REWARD_CONFIG = {
    "terminal_survival":       +100.0,
    "terminal_death_icu":      -100.0,
    "terminal_death_hospital": -80.0,
    "vfd_bonus_per_day":       0.5,
    "spo2_target_low":         92,
    "spo2_target_high":        98,
    "pplat_warn":              28,
    "pplat_hard":              30,
    "pplat_critical":          35,
    "dp_warn":                 13,
    "dp_hard":                 15,
    "dp_critical":             20,
    "mp_warn":                 20,
    "mp_critical":             25,
    "vt_pbw_high":             8.0,
    "vt_pbw_critical":         9.0,
    "vt_pbw_low":              4.0,
}

# ─── ICD-9 Exclusion Codes (MIMIC-III uses ICD-9) ────────
TBI_ICD9 = ["800", "801", "802", "803", "804", "850", "851", "852", "853", "854"]
NMD_ICD9 = ["335.2", "359.1", "357.0", "358.0"]  # ALS, MD, GBS, MG
ECMO_ICD9 = ["39.65"]
CARDIAC_SURGERY_ICD9 = ["35", "36", "37"]  # Valve, CABG, heart procedures

# Derived constants
STATE_DIM = len(
    ["age", "bmi", "pbw",
     "heart_rate", "map", "spo2", "temperature",
     "mechanical_power", "tidal_volume", "rr", "peep_final",
     "fio2", "peak_pressure", "plateau_pressure",
     "driving_pressure", "compliance_static",
     "vt_per_pbw", "mp_per_pbw",
     "ph", "pao2", "paco2", "pf_ratio", "sf_ratio",
     "lactate", "creatinine",
     "ards_severity", "hours_on_vent"]
)  # 27

RANDOM_SEED = 42
