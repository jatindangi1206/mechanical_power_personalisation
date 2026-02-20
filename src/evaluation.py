"""
Evaluation framework for all three strategies.
Includes: classification metrics, off-policy evaluation, 
subgroup analysis, and safety audit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import ACTIONS, ACTION_DELTAS, N_ACTIONS


# ═══════════════════════════════════════════════════════════
# 1. CLASSIFICATION METRICS (Strategy 1 & 2)
# ═══════════════════════════════════════════════════════════

def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray,
                         name: str = "Model") -> Dict:
    """Comprehensive binary classification evaluation."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        precision_recall_curve, roc_curve, f1_score,
        confusion_matrix, classification_report
    )
    
    results = {}
    
    # Core metrics
    results["auroc"] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    results["auprc"] = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    results["brier_score"] = brier_score_loss(y_true, y_prob)
    
    # At optimal threshold (Youden's J)
    if len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (y_prob >= best_thresh).astype(int)
        results["optimal_threshold"] = float(best_thresh)
        results["sensitivity"] = float(tpr[best_idx])
        results["specificity"] = float(1 - fpr[best_idx])
        results["f1"] = f1_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
    
    # Calibration (ECE)
    results["ece"] = expected_calibration_error(y_true, y_prob)
    
    print(f"\n{'='*40}")
    print(f"{name} — Evaluation Results")
    print(f"{'='*40}")
    print(f"  AUROC:       {results['auroc']:.4f}")
    print(f"  AUPRC:       {results['auprc']:.4f}")
    print(f"  Brier Score: {results['brier_score']:.4f}")
    print(f"  ECE:         {results['ece']:.4f}")
    if "f1" in results:
        print(f"  F1:          {results['f1']:.4f}")
        print(f"  Sensitivity: {results['sensitivity']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
    
    return results


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    
    return ece


# ═══════════════════════════════════════════════════════════
# 2. OFF-POLICY EVALUATION (Strategy 3)
# ═══════════════════════════════════════════════════════════

def off_policy_evaluation(agent, episodes: List[Dict],
                           gamma: float = 0.99,
                           feature_list: List[str] = None) -> Dict:
    """
    Off-policy evaluation using three estimators:
    1. Direct Method (DM) — Q-function estimates
    2. Importance Sampling (IS) 
    3. Per-step analysis
    """
    from src.mdp_dataset import flatten_state, ALL_STATE_FEATURES
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    results = {}
    
    # ── Direct Method ──
    dm_values = []
    for ep in episodes:
        if len(ep["states"]) == 0:
            continue
        s0 = flatten_state(ep["states"][0], feature_list)
        pred = agent.predict(s0)
        dm_values.append(max(pred["q_values"]))  # V(s0) = max_a Q(s0, a)
    
    results["dm_estimate"] = float(np.mean(dm_values)) if dm_values else 0.0
    results["dm_std"] = float(np.std(dm_values)) if dm_values else 0.0
    
    # ── Observed Returns ──
    observed_returns = []
    for ep in episodes:
        ep_return = 0.0
        discount = 1.0
        for r in ep["rewards"]:
            ep_return += discount * r
            discount *= gamma
        observed_returns.append(ep_return)
    
    results["observed_return_mean"] = float(np.mean(observed_returns))
    results["observed_return_std"] = float(np.std(observed_returns))
    
    # ── Policy vs. Behaviour Agreement ──
    agreements = []
    policy_actions = []
    behaviour_actions = []
    
    for ep in episodes:
        for t, (state, actual_a) in enumerate(
                zip(ep["states"], ep["actions"])):
            s = flatten_state(state, feature_list)
            pred = agent.predict(s)
            policy_actions.append(pred["action"])
            behaviour_actions.append(actual_a)
            agreements.append(int(pred["action"] == actual_a))
    
    results["agreement_rate"] = float(np.mean(agreements))
    
    # Policy action distribution
    policy_dist = pd.Series(policy_actions).value_counts(normalize=True).sort_index()
    behaviour_dist = pd.Series(behaviour_actions).value_counts(normalize=True).sort_index()
    
    results["policy_action_dist"] = {
        ACTIONS.get(int(k), str(k)): float(v) for k, v in policy_dist.items()
    }
    results["behaviour_action_dist"] = {
        ACTIONS.get(int(k), str(k)): float(v) for k, v in behaviour_dist.items()
    }
    
    # ── Improvement Estimate ──
    results["estimated_improvement"] = (
        results["dm_estimate"] - results["observed_return_mean"]
    )
    
    print(f"\n{'='*50}")
    print(f"OFF-POLICY EVALUATION")
    print(f"{'='*50}")
    print(f"  DM Estimate:        {results['dm_estimate']:.2f} ± {results['dm_std']:.2f}")
    print(f"  Observed Return:    {results['observed_return_mean']:.2f} ± {results['observed_return_std']:.2f}")
    print(f"  Est. Improvement:   {results['estimated_improvement']:.2f}")
    print(f"  Agreement Rate:     {results['agreement_rate']:.1%}")
    print(f"  Policy Distribution:")
    for act, pct in results["policy_action_dist"].items():
        print(f"    {act}: {pct:.1%}")
    
    return results


# ═══════════════════════════════════════════════════════════
# 3. SAFETY AUDIT
# ═══════════════════════════════════════════════════════════

def safety_audit(agent, episodes: List[Dict],
                  feature_list: List[str] = None,
                  safety_filter=None) -> Dict:
    """
    Audit the policy for safety violations.
    Counts how often the policy would recommend unsafe actions.
    """
    from src.mdp_dataset import flatten_state, ALL_STATE_FEATURES
    from src.models import SafetyFilter
    
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    if safety_filter is None:
        safety_filter = SafetyFilter()
    
    total_steps = 0
    violations = {
        "safety_override": 0,
        "high_mp_recommended": 0,
        "high_dp_state": 0,
        "hypoxemia_state": 0,
    }
    alerts_log = []
    
    for ep in episodes:
        for t, state in enumerate(ep["states"]):
            total_steps += 1
            s = flatten_state(state, feature_list)
            pred = agent.predict(s)
            raw_action = pred["action"]
            
            filtered_action, alert = safety_filter.filter(state, raw_action)
            
            if alert is not None:
                violations["safety_override"] += 1
                alerts_log.append({
                    "icustay_id": state.get("icustay_id"),
                    "hour": t,
                    "raw_action": ACTIONS[raw_action],
                    "filtered_action": ACTIONS[filtered_action],
                    "alert": alert,
                })
            
            # Check resulting MP
            mp = state.get("mechanical_power", 0) or 0
            target_mp = mp + ACTION_DELTAS.get(filtered_action, 0)
            if target_mp > 25:
                violations["high_mp_recommended"] += 1
            
            dp = state.get("driving_pressure")
            if pd.notna(dp) and dp > 15:
                violations["high_dp_state"] += 1
            
            spo2 = state.get("spo2")
            if pd.notna(spo2) and spo2 < 88:
                violations["hypoxemia_state"] += 1
    
    rates = {k: v / max(total_steps, 1) for k, v in violations.items()}
    
    print(f"\n{'='*50}")
    print(f"SAFETY AUDIT")
    print(f"{'='*50}")
    print(f"  Total decision steps: {total_steps}")
    for k, v in violations.items():
        print(f"  {k}: {v} ({rates[k]:.2%})")
    
    return {
        "total_steps": total_steps,
        "violations": violations,
        "violation_rates": rates,
        "alerts_log": alerts_log[:20],  # First 20 alerts
    }


# ═══════════════════════════════════════════════════════════
# 4. SUBGROUP ANALYSIS
# ═══════════════════════════════════════════════════════════

def subgroup_analysis(model, episodes: List[Dict],
                       feature_list: List[str] = None,
                       mode: str = "classifier") -> pd.DataFrame:
    """
    Evaluate model performance across clinical subgroups.
    """
    from sklearn.metrics import roc_auc_score
    from src.mdp_dataset import flatten_state, ALL_STATE_FEATURES
    
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    # Define subgroups
    subgroup_filters = {
        "All": lambda ep: True,
        "Mortality=Yes": lambda ep: ep["patient_info"]["hospital_mortality"] == 1,
        "Mortality=No": lambda ep: ep["patient_info"]["hospital_mortality"] == 0,
    }
    
    # Add MP-based subgroups
    def _get_initial_mp(ep):
        if ep["states"]:
            return ep["states"][0].get("mechanical_power", np.nan)
        return np.nan
    
    subgroup_filters["High MP (>20)"] = lambda ep: (
        pd.notna(_get_initial_mp(ep)) and _get_initial_mp(ep) > 20
    )
    subgroup_filters["Low MP (≤20)"] = lambda ep: (
        pd.notna(_get_initial_mp(ep)) and _get_initial_mp(ep) <= 20
    )
    
    results = []
    for name, filter_fn in subgroup_filters.items():
        subset = [ep for ep in episodes if filter_fn(ep)]
        n = len(subset)
        if n < 5:
            continue
        
        mort_rate = np.mean([
            ep["patient_info"]["hospital_mortality"] for ep in subset
        ])
        
        row = {"Subgroup": name, "N": n, "Mortality Rate": f"{mort_rate:.1%}"}
        
        if mode == "policy":
            # Agreement rate for RL
            agreements = []
            for ep in subset:
                for state, actual_a in zip(ep["states"], ep["actions"]):
                    s = flatten_state(state, feature_list)
                    pred = model.predict(s)
                    agreements.append(int(pred["action"] == actual_a))
            row["Agreement"] = f"{np.mean(agreements):.1%}" if agreements else "N/A"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"SUBGROUP ANALYSIS")
    print(f"{'='*50}")
    print(df.to_string(index=False))
    
    return df


# ═══════════════════════════════════════════════════════════
# 5. COMPARISON TABLE
# ═══════════════════════════════════════════════════════════

def comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a formatted comparison table across strategies."""
    metrics = ["auroc", "auprc", "brier_score", "ece", "f1",
               "agreement_rate", "dm_estimate"]
    
    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for strategy, res in results.items():
            val = res.get(metric, "N/A")
            if isinstance(val, float):
                row[strategy] = f"{val:.4f}"
            else:
                row[strategy] = str(val)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    return df
