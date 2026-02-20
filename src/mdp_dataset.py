"""
MDP Dataset Construction for Offline RL.
Converts preprocessed hourly data into (s, a, r, s', done) transitions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import (
    ACTIONS, ACTION_DELTAS, N_ACTIONS, REWARD_CONFIG, RL_CONFIG
)


# ═══════════════════════════════════════════════════════════
# 1. ACTION DISCRETISATION
# ═══════════════════════════════════════════════════════════

def discretise_mp_action(mp_current: float, mp_next: float) -> int:
    """
    Map the observed MP change to one of 5 discrete actions.
    
    Actions:
        0: large_decrease  (ΔMP ≤ -3.5)
        1: small_decrease  (-3.5 < ΔMP ≤ -1.0)
        2: maintain        (-1.0 < ΔMP < +1.0)
        3: small_increase  (+1.0 ≤ ΔMP < +3.5)
        4: large_increase  (ΔMP ≥ +3.5)
    """
    if np.isnan(mp_current) or np.isnan(mp_next):
        return 2  # Default to maintain if MP unknown
    
    delta = mp_next - mp_current
    
    if delta <= -3.5:
        return 0   # large decrease
    elif delta <= -1.0:
        return 1   # small decrease
    elif delta < 1.0:
        return 2   # maintain
    elif delta < 3.5:
        return 3   # small increase
    else:
        return 4   # large increase


# ═══════════════════════════════════════════════════════════
# 2. REWARD FUNCTION
# ═══════════════════════════════════════════════════════════

def calculate_reward(state_t: Dict, state_t1: Dict, action: int,
                     is_terminal: bool, episode_info: Dict) -> float:
    """
    Multi-objective reward function with clinical hierarchy:
    1. Survival (dominant terminal signal)
    2. Oxygenation adequacy
    3. VILI avoidance
    4. Haemodynamic stability
    5. Weaning encouragement
    """
    R = REWARD_CONFIG
    reward = 0.0

    # ════════════════════════════════════════════════
    # TERMINAL REWARD
    # ════════════════════════════════════════════════
    if is_terminal:
        if episode_info.get("hospital_mortality", 0) == 1:
            if episode_info.get("icu_mortality", 0) == 1:
                reward += R["terminal_death_icu"]
            else:
                reward += R["terminal_death_hospital"]
        else:
            reward += R["terminal_survival"]
        return float(reward)

    # ════════════════════════════════════════════════
    # INTERMEDIATE REWARDS
    # ════════════════════════════════════════════════

    # ── Oxygenation ──
    spo2_t = state_t.get("spo2", 95)
    spo2_t1 = state_t1.get("spo2", 95)
    
    if pd.notna(spo2_t1):
        if R["spo2_target_low"] <= spo2_t1 <= R["spo2_target_high"]:
            reward += 1.0
        
        if pd.notna(spo2_t):
            delta_spo2 = spo2_t1 - spo2_t
            if delta_spo2 > 0:
                reward += 1.5 * min(delta_spo2, 5)
            elif spo2_t1 < R["spo2_target_low"]:
                reward += 2.0 * max(delta_spo2, -10)

        # Hypoxaemia penalties
        if spo2_t1 < 80:
            reward -= 40.0
        elif spo2_t1 < 85:
            reward -= 25.0
        elif spo2_t1 < 88:
            reward -= 10.0

    # ── P/F ratio improvement ──
    pf_t = state_t.get("pf_ratio")
    pf_t1 = state_t1.get("pf_ratio")
    if pd.notna(pf_t) and pd.notna(pf_t1):
        delta_pf = pf_t1 - pf_t
        reward += 0.05 * np.clip(delta_pf, -50, 50)

    # ── Haemodynamic stability ──
    map_t1 = state_t1.get("map")
    map_t = state_t.get("map")
    if pd.notna(map_t1):
        if 65 <= map_t1 <= 100:
            reward += 0.5
        if map_t1 < 50:
            reward -= 30.0
        elif map_t1 < 60:
            reward -= 15.0
        if pd.notna(map_t):
            delta_map = map_t1 - map_t
            if delta_map > 0 and map_t1 < 65:
                reward += 0.3 * min(delta_map, 20)

    # ── Lactate clearance ──
    lac_t = state_t.get("lactate")
    lac_t1 = state_t1.get("lactate")
    if pd.notna(lac_t) and pd.notna(lac_t1):
        if lac_t1 < lac_t:
            reward += 2.0

    # ════════════════════════════════════════════════
    # SAFETY PENALTIES — VILI Prevention
    # ════════════════════════════════════════════════

    # Plateau pressure
    pplat = state_t1.get("plateau_pressure")
    if pd.notna(pplat):
        if pplat > R["pplat_critical"]:
            reward -= 30.0
        elif pplat > R["pplat_hard"]:
            reward -= 15.0
        elif pplat > R["pplat_warn"]:
            reward -= 5.0

    # Driving pressure
    dp = state_t1.get("driving_pressure")
    if pd.notna(dp):
        if dp > R["dp_critical"]:
            reward -= 20.0
        elif dp > R["dp_hard"]:
            reward -= 8.0
        elif dp > R["dp_warn"]:
            reward -= 3.0

    # Mechanical power
    mp = state_t1.get("mechanical_power")
    if pd.notna(mp):
        if mp > R["mp_critical"]:
            reward -= 5.0 * (mp - R["mp_critical"])
        elif mp > R["mp_warn"]:
            reward -= 3.0 * (mp - R["mp_warn"])

    # Tidal volume per PBW
    vt_pbw = state_t1.get("vt_per_pbw")
    if pd.notna(vt_pbw):
        if vt_pbw > R["vt_pbw_critical"]:
            reward -= 20.0
        elif vt_pbw > R["vt_pbw_high"]:
            reward -= 8.0
        elif vt_pbw < R["vt_pbw_low"]:
            reward -= 5.0

    # ════════════════════════════════════════════════
    # WEANING BONUS
    # ════════════════════════════════════════════════
    is_stable = (
        (pd.isna(spo2_t1) or spo2_t1 >= 92) and
        (pd.isna(map_t1) or map_t1 >= 65) and
        (pd.isna(lac_t1) or lac_t1 < 2.0)
    )
    if action in [0, 1] and is_stable:
        reward += 3.0

    # ════════════════════════════════════════════════
    # TIME PENALTY
    # ════════════════════════════════════════════════
    hours = state_t.get("hours_on_vent", 0)
    if hours < 24:
        reward -= 0.05
    elif hours < 72:
        reward -= 0.10
    else:
        reward -= 0.20

    return float(reward)


# ═══════════════════════════════════════════════════════════
# 3. STATE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════

# Features that form the state vector
STATIC_FEATURES = [
    "age", "bmi", "pbw",
]
SNAPSHOT_FEATURES = [
    "heart_rate", "map", "spo2", "temperature",
    "mechanical_power", "tidal_volume", "rr", "peep_final",
    "fio2", "peak_pressure", "plateau_pressure",
    "driving_pressure", "compliance_static",
    "vt_per_pbw", "mp_per_pbw",
]
LAB_FEATURES = [
    "ph", "pao2", "paco2", "pf_ratio", "sf_ratio",
    "lactate", "creatinine",
]
DERIVED_FEATURES = [
    "ards_severity", "hours_on_vent",
]

ALL_STATE_FEATURES = STATIC_FEATURES + SNAPSHOT_FEATURES + LAB_FEATURES + DERIVED_FEATURES


def flatten_state(row: pd.Series, feature_list: List[str] = None) -> np.ndarray:
    """Convert a row/dict to a flat numpy state vector."""
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    state = []
    for f in feature_list:
        val = row.get(f, np.nan)
        if pd.isna(val):
            state.append(0.0)  # Fill with 0 for missing
        else:
            state.append(float(val))
    return np.array(state, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# 4. EPISODE CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def build_episodes(hourly_data: pd.DataFrame,
                   cohort: pd.DataFrame,
                   min_steps: int = 6,
                   verbose: bool = True) -> List[Dict]:
    """
    Build patient episodes for offline RL.
    
    Each episode contains:
        - states: list of state dicts
        - actions: list of discretised MP actions
        - rewards: list of step rewards
        - patient_info: static patient info
    """
    episodes = []
    n_skipped = 0
    
    # Merge static info
    static_cols = ["icustay_id", "age", "gender", "bmi", "pbw",
                   "hospital_mortality", "icu_mortality",
                   "icu_los_hours", "admission_type"]
    static_info = cohort[
        [c for c in static_cols if c in cohort.columns]
    ].copy()
    
    hourly = hourly_data.merge(
        static_info, on="icustay_id", how="left"
    ).sort_values(["icustay_id", "hour"])
    
    for stay_id, group in hourly.groupby("icustay_id"):
        group = group.reset_index(drop=True)
        T = len(group)
        
        if T < min_steps:
            n_skipped += 1
            continue
        
        # Compute hours on ventilator
        if "hour" in group.columns:
            start_time = group["hour"].iloc[0]
            group["hours_on_vent"] = (
                (group["hour"] - start_time).dt.total_seconds() / 3600
            ).astype(int)
        else:
            group["hours_on_vent"] = range(T)
        
        # Patient-level info
        patient_info = {
            "icustay_id": stay_id,
            "hospital_mortality": int(group["hospital_mortality"].iloc[0])
                if "hospital_mortality" in group.columns else 0,
            "icu_mortality": int(group["icu_mortality"].iloc[0])
                if "icu_mortality" in group.columns else 0,
            "n_hours": T,
        }
        
        states = []
        actions = []
        rewards = []
        
        for t in range(T):
            row = group.iloc[t]
            state = row.to_dict()
            states.append(state)
            
            # Action: observed MP change to next step
            if t < T - 1:
                mp_t = row.get("mechanical_power", np.nan)
                mp_t1 = group.iloc[t + 1].get("mechanical_power", np.nan)
                action = discretise_mp_action(mp_t, mp_t1)
            else:
                action = 2  # Terminal step: maintain
            actions.append(action)
        
        # Compute rewards
        for t in range(T):
            is_terminal = (t == T - 1)
            state_t = states[t]
            state_t1 = states[min(t + 1, T - 1)]
            
            r = calculate_reward(
                state_t, state_t1, actions[t],
                is_terminal, patient_info
            )
            rewards.append(r)
        
        episodes.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "patient_info": patient_info,
        })
    
    if verbose:
        print(f"Built {len(episodes)} episodes "
              f"({n_skipped} skipped, min_steps={min_steps})")
        if episodes:
            all_actions = [a for ep in episodes for a in ep["actions"]]
            action_dist = pd.Series(all_actions).value_counts(normalize=True).sort_index()
            print(f"  Action distribution:")
            for a, pct in action_dist.items():
                print(f"    {a} ({ACTIONS[a]}): {pct:.1%}")
            
            all_rewards = [r for ep in episodes for r in ep["rewards"]]
            print(f"  Reward stats: mean={np.mean(all_rewards):.2f}, "
                  f"std={np.std(all_rewards):.2f}, "
                  f"min={np.min(all_rewards):.2f}, "
                  f"max={np.max(all_rewards):.2f}")
    
    return episodes


# ═══════════════════════════════════════════════════════════
# 5. CONVERT TO ARRAYS (for RL libraries)
# ═══════════════════════════════════════════════════════════

def episodes_to_arrays(episodes: List[Dict],
                       feature_list: List[str] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """
    Convert episodes to flat arrays for RL training.
    
    Returns:
        states: (N, state_dim)
        actions: (N,)
        rewards: (N,)
        next_states: (N, state_dim) — next observation (same as current at terminal)
        dones: (N,) — 1.0 at episode end
    """
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    
    for ep in episodes:
        T = len(ep["states"])
        for t in range(T):
            obs = flatten_state(ep["states"][t], feature_list)
            observations.append(obs)
            
            # Next observation
            if t < T - 1:
                next_obs = flatten_state(ep["states"][t + 1], feature_list)
            else:
                next_obs = obs  # Terminal: next_obs = current obs
            next_observations.append(next_obs)
            
            actions.append(ep["actions"][t])
            rewards.append(ep["rewards"][t])
            terminals.append(1.0 if t == T - 1 else 0.0)
    
    return (
        np.array(observations, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_observations, dtype=np.float32),
        np.array(terminals, dtype=np.float32),
    )


# ═══════════════════════════════════════════════════════════
# 6. TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════

def split_episodes(episodes: List[Dict],
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   seed: int = 42,
                   stratify: bool = True) -> Tuple[List, List, List]:
    """
    Patient-level stratified split.
    Ensures no patient appears in multiple splits.
    """
    np.random.seed(seed)
    
    if stratify:
        # Separate by outcome for stratification
        survived = [ep for ep in episodes
                    if ep["patient_info"]["hospital_mortality"] == 0]
        died = [ep for ep in episodes
                if ep["patient_info"]["hospital_mortality"] == 1]
        
        def _split_group(group):
            np.random.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            return (group[:n_train],
                    group[n_train:n_train + n_val],
                    group[n_train + n_val:])
        
        s_train, s_val, s_test = _split_group(survived)
        d_train, d_val, d_test = _split_group(died)
        
        train = list(s_train) + list(d_train)
        val = list(s_val) + list(d_val)
        test = list(s_test) + list(d_test)
    else:
        shuffled = episodes.copy()
        np.random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
    
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)
    
    print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    mort_train = np.mean([e["patient_info"]["hospital_mortality"] for e in train]) if train else 0
    mort_val = np.mean([e["patient_info"]["hospital_mortality"] for e in val]) if val else 0
    mort_test = np.mean([e["patient_info"]["hospital_mortality"] for e in test]) if test else 0
    print(f"  Mortality: Train={mort_train:.1%}, Val={mort_val:.1%}, Test={mort_test:.1%}")
    
    return train, val, test


# ═══════════════════════════════════════════════════════════
# 7. NORMALISATION
# ═══════════════════════════════════════════════════════════

class StateNormaliser:
    """Z-score normalisation fitted on training data."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, observations: np.ndarray):
        """Fit on training observations."""
        self.mean = np.nanmean(observations, axis=0)
        self.std = np.nanstd(observations, axis=0)
        self.std[self.std < 1e-6] = 1.0  # Avoid division by zero
        self.fitted = True
        return self
    
    def transform(self, observations: np.ndarray) -> np.ndarray:
        """Apply normalisation."""
        assert self.fitted, "Must call fit() first"
        return (observations - self.mean) / self.std
    
    def fit_transform(self, observations: np.ndarray) -> np.ndarray:
        self.fit(observations)
        return self.transform(observations)
