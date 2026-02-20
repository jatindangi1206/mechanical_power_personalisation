"""
Explainability and visualisation module.
Provides: feature importance, SHAP analysis, trajectory plots,
          clinical dashboard components, and policy comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import ACTIONS, ACTION_DELTAS, N_ACTIONS, ARTEFACTS_DIR

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 120,
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colour scheme
COLORS = {
    "survival": "#2ecc71",
    "death": "#e74c3c",
    "mp_safe": "#27ae60",
    "mp_warn": "#f39c12",
    "mp_danger": "#c0392b",
    "policy": "#3498db",
    "behaviour": "#95a5a6",
    "highlight": "#9b59b6",
}


# ═══════════════════════════════════════════════════════════
# 1. DATA EXPLORATION PLOTS
# ═══════════════════════════════════════════════════════════

def plot_cohort_summary(cohort: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Summary dashboard for the study cohort."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Study Cohort Summary", fontsize=16, fontweight="bold")
    
    # Age distribution
    ax = axes[0, 0]
    cohort["age"].hist(bins=20, ax=ax, color=COLORS["policy"], edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution")
    ax.axvline(cohort["age"].median(), color="red", ls="--", label=f"Median: {cohort['age'].median():.0f}")
    ax.legend()
    
    # Gender
    ax = axes[0, 1]
    cohort["gender"].value_counts().plot.bar(ax=ax, color=[COLORS["policy"], COLORS["highlight"]])
    ax.set_title("Gender Distribution")
    ax.set_ylabel("Count")
    
    # ICU LOS
    ax = axes[0, 2]
    cohort["icu_los_hours"].hist(bins=20, ax=ax, color=COLORS["mp_safe"], edgecolor="white")
    ax.set_xlabel("ICU LOS (hours)")
    ax.set_title("ICU Length of Stay")
    
    # Mortality
    ax = axes[1, 0]
    mort_data = [
        cohort["hospital_mortality"].sum(),
        len(cohort) - cohort["hospital_mortality"].sum()
    ]
    ax.pie(mort_data, labels=["Died", "Survived"],
           colors=[COLORS["death"], COLORS["survival"]],
           autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Hospital Mortality (n={len(cohort)})")
    
    # Admission type
    ax = axes[1, 1]
    if "admission_type" in cohort.columns:
        cohort["admission_type"].value_counts().plot.bar(ax=ax, color=COLORS["behaviour"])
        ax.set_title("Admission Type")
        ax.set_ylabel("Count")
    
    # ICU type
    ax = axes[1, 2]
    if "first_careunit" in cohort.columns:
        cohort["first_careunit"].value_counts().plot.bar(ax=ax, color=COLORS["highlight"])
        ax.set_title("Care Unit")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "cohort_summary.png", bbox_inches="tight")
    return fig


def plot_mp_distribution(hourly_data: pd.DataFrame,
                         cohort: pd.DataFrame,
                         save: bool = True) -> plt.Figure:
    """Mechanical power distribution analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mechanical Power Analysis", fontsize=14, fontweight="bold")
    
    mp_data = hourly_data["mechanical_power"].dropna()
    
    # Overall distribution
    ax = axes[0]
    ax.hist(mp_data, bins=50, color=COLORS["policy"], edgecolor="white", alpha=0.7)
    ax.axvline(mp_data.median(), color="red", ls="--", 
               label=f"Median: {mp_data.median():.1f}")
    ax.axvline(20, color=COLORS["mp_warn"], ls="--", 
               label="Warning (20 J/min)")
    ax.axvline(25, color=COLORS["mp_danger"], ls="--", 
               label="Critical (25 J/min)")
    ax.set_xlabel("Mechanical Power (J/min)")
    ax.set_ylabel("Frequency")
    ax.set_title("MP Distribution")
    ax.legend(fontsize=9)
    
    # MP vs mortality
    ax = axes[1]
    merged = hourly_data.merge(
        cohort[["icustay_id", "hospital_mortality"]],
        on="icustay_id", how="left"
    )
    survived = merged[merged["hospital_mortality"] == 0]["mechanical_power"].dropna()
    died = merged[merged["hospital_mortality"] == 1]["mechanical_power"].dropna()
    
    if len(survived) > 0 and len(died) > 0:
        ax.hist(survived, bins=30, alpha=0.6, label=f"Survived (n={len(survived)})",
                color=COLORS["survival"], edgecolor="white")
        ax.hist(died, bins=30, alpha=0.6, label=f"Died (n={len(died)})",
                color=COLORS["death"], edgecolor="white")
        ax.set_xlabel("Mechanical Power (J/min)")
        ax.set_title("MP by Outcome")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
    
    # MP over time (sample trajectories)
    ax = axes[2]
    sample_stays = hourly_data["icustay_id"].unique()[:5]
    for stay_id in sample_stays:
        subset = hourly_data[hourly_data["icustay_id"] == stay_id].sort_values("hour")
        if len(subset) > 0:
            hours = range(len(subset))
            mp_vals = subset["mechanical_power"].values
            ax.plot(hours, mp_vals, alpha=0.7, marker=".", markersize=3)
    ax.set_xlabel("Hours on ventilator")
    ax.set_ylabel("Mechanical Power (J/min)")
    ax.set_title("Sample MP Trajectories")
    ax.axhline(20, color=COLORS["mp_warn"], ls="--", alpha=0.5)
    ax.axhline(25, color=COLORS["mp_danger"], ls="--", alpha=0.5)
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "mp_distribution.png", bbox_inches="tight")
    return fig


def plot_ventilator_parameters(hourly_data: pd.DataFrame,
                               save: bool = True) -> plt.Figure:
    """Distribution of key ventilator parameters."""
    params = [
        ("tidal_volume", "Tidal Volume (mL)", "blue"),
        ("rr", "Respiratory Rate (/min)", "green"),
        ("peep_final", "PEEP (cmH2O)", "orange"),
        ("peak_pressure", "Peak Pressure (cmH2O)", "red"),
        ("driving_pressure", "Driving Pressure (cmH2O)", "purple"),
        ("fio2", "FiO2", "teal"),
    ]
    
    avail = [(p, l, c) for p, l, c in params if p in hourly_data.columns
             and hourly_data[p].notna().sum() > 0]
    
    n = len(avail)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ventilator parameters available", ha="center")
        return fig
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Ventilator Parameter Distributions", fontsize=14, fontweight="bold")
    
    for i, (param, label, color) in enumerate(avail[:6]):
        ax = axes[i]
        data = hourly_data[param].dropna()
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_title(f"{label}\n(n={len(data)}, med={data.median():.1f})")
    
    for i in range(len(avail), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "vent_params.png", bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════
# 2. MODEL EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════

def plot_training_curves(history: Dict, save: bool = True) -> plt.Figure:
    """Plot CQL training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("CQL Training Progress", fontsize=14, fontweight="bold")
    
    axes[0].plot(history["loss"], color=COLORS["policy"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss")
    
    axes[1].plot(history["q_mean"], color=COLORS["mp_safe"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Q-value")
    axes[1].set_title("Q-value (should stabilise)")
    
    axes[2].plot(history["cql_loss"], color=COLORS["mp_warn"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("CQL Penalty")
    axes[2].set_title("Conservative Penalty")
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "training_curves.png", bbox_inches="tight")
    return fig


def plot_roc_pr_curves(y_true: np.ndarray, 
                        predictions: Dict[str, np.ndarray],
                        save: bool = True) -> plt.Figure:
    """ROC and PR curves for all strategies."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Strategy Comparison: ROC & Precision-Recall Curves",
                 fontsize=14, fontweight="bold")
    
    colors = [COLORS["policy"], COLORS["highlight"], COLORS["mp_safe"]]
    
    for i, (name, y_prob) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC={roc_auc:.3f})")
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2,
                     label=f"{name} (AUC={pr_auc:.3f})")
    
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    
    baseline = y_true.mean()
    axes[1].axhline(baseline, color="k", ls="--", alpha=0.3,
                    label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "roc_pr_curves.png", bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names: List[str],
                            top_n: int = 15, save: bool = True) -> plt.Figure:
    """XGBoost feature importance plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    
    ax.barh(range(len(idx)), importance[idx], color=COLORS["policy"])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names)
                        else f"Feature_{i}" for i in idx])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Features — XGBoost")
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "feature_importance.png", bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════
# 3. POLICY VISUALISATION
# ═══════════════════════════════════════════════════════════

def plot_policy_comparison(episodes: List[Dict],
                            agent = None,
                            feature_list: List[str] = None,
                            n_episodes: int = 3,
                            save: bool = True) -> plt.Figure:
    """Compare behaviour policy vs. learned policy on sample trajectories."""
    from src.mdp_dataset import flatten_state, ALL_STATE_FEATURES
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    n_episodes = min(n_episodes, len(episodes))
    fig, axes = plt.subplots(n_episodes, 3, figsize=(18, 4 * n_episodes))
    if n_episodes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Behaviour vs. Learned Policy", fontsize=14, fontweight="bold")
    
    action_colors = {
        0: COLORS["mp_safe"], 1: "#85c1e9",
        2: COLORS["behaviour"], 3: "#f5b041", 4: COLORS["mp_danger"]
    }
    
    for i in range(n_episodes):
        ep = episodes[i]
        T = len(ep["states"])
        hours = range(T)
        
        # MP trajectory
        mp_vals = [s.get("mechanical_power", np.nan) for s in ep["states"]]
        axes[i, 0].plot(hours, mp_vals, color=COLORS["policy"], marker=".", label="MP")
        axes[i, 0].axhline(20, color=COLORS["mp_warn"], ls="--", alpha=0.5)
        axes[i, 0].axhline(25, color=COLORS["mp_danger"], ls="--", alpha=0.5)
        axes[i, 0].set_ylabel("MP (J/min)")
        axes[i, 0].set_title(f"Patient {i+1} — MP Trajectory "
                              f"(Mort: {ep['patient_info']['hospital_mortality']})")
        
        # Behaviour actions
        axes[i, 1].bar(hours, [ACTION_DELTAS[a] for a in ep["actions"]],
                       color=[action_colors[a] for a in ep["actions"]],
                       alpha=0.7)
        axes[i, 1].set_ylabel("ΔMP (J/min)")
        axes[i, 1].set_title("Clinician Actions (Observed)")
        
        # Learned policy actions (if agent available)
        if agent is not None:
            policy_actions = []
            for state in ep["states"]:
                s = flatten_state(state, feature_list)
                pred = agent.predict(s)
                policy_actions.append(pred["action"])
            
            axes[i, 2].bar(hours,
                           [ACTION_DELTAS[a] for a in policy_actions],
                           color=[action_colors[a] for a in policy_actions],
                           alpha=0.7)
            axes[i, 2].set_ylabel("ΔMP (J/min)")
            axes[i, 2].set_title("CQL Policy (Recommended)")
        else:
            axes[i, 2].text(0.5, 0.5, "No agent provided",
                            ha="center", va="center",
                            transform=axes[i, 2].transAxes)
    
    for ax in axes[-1]:
        ax.set_xlabel("Hour")
    
    # Legend
    patches = [mpatches.Patch(color=c, label=ACTIONS[a])
               for a, c in action_colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        fig.savefig(ARTEFACTS_DIR / "policy_comparison.png", bbox_inches="tight")
    return fig


def plot_reward_analysis(episodes: List[Dict],
                          save: bool = True) -> plt.Figure:
    """Analyse reward distribution and components."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Reward Analysis", fontsize=14, fontweight="bold")
    
    # Episode returns
    returns = []
    mortality = []
    for ep in episodes:
        ep_return = sum(ep["rewards"])
        returns.append(ep_return)
        mortality.append(ep["patient_info"]["hospital_mortality"])
    
    # Return distribution
    axes[0].hist(returns, bins=30, color=COLORS["policy"], edgecolor="white")
    axes[0].set_xlabel("Episode Return")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Return Distribution")
    
    # Returns by outcome
    survived_returns = [r for r, m in zip(returns, mortality) if m == 0]
    died_returns = [r for r, m in zip(returns, mortality) if m == 1]
    
    data = []
    labels = []
    if survived_returns:
        data.append(survived_returns)
        labels.append("Survived")
    if died_returns:
        data.append(died_returns)
        labels.append("Died")
    
    if data:
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        colors_box = [COLORS["survival"], COLORS["death"]]
        for patch, color in zip(bp["boxes"], colors_box[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    axes[1].set_ylabel("Episode Return")
    axes[1].set_title("Returns by Outcome")
    
    # Step reward distribution
    all_rewards = [r for ep in episodes for r in ep["rewards"]]
    axes[2].hist(all_rewards, bins=50, color=COLORS["highlight"],
                 edgecolor="white", log=True)
    axes[2].set_xlabel("Step Reward")
    axes[2].set_ylabel("Frequency (log)")
    axes[2].set_title("Step Reward Distribution")
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "reward_analysis.png", bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════
# 4. CLINICAL DECISION DISPLAY (for dashboard)
# ═══════════════════════════════════════════════════════════

def display_clinical_decision(state: Dict, prediction: Dict,
                               safety_alert: Optional[str] = None):
    """
    Format a clinical decision recommendation for display.
    Returns a formatted string.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("    MP RECOMMENDATION ENGINE")
    lines.append("=" * 50)
    
    # Current state
    lines.append("\n── Current Patient State ──")
    mp = state.get("mechanical_power", "N/A")
    lines.append(f"  Mechanical Power: {mp:.1f} J/min" if isinstance(mp, (int, float)) else f"  Mechanical Power: {mp}")
    
    for key, label in [
        ("spo2", "SpO2"), ("map", "MAP"), ("heart_rate", "HR"),
        ("driving_pressure", "ΔP"), ("peep_final", "PEEP"),
        ("fio2", "FiO2"), ("vt_per_pbw", "VT/PBW"),
    ]:
        val = state.get(key)
        if pd.notna(val):
            lines.append(f"  {label}: {val:.1f}")
    
    # Recommendation
    lines.append("\n── Recommendation ──")
    action = prediction["action"]
    action_label = prediction["action_label"]
    delta = ACTION_DELTAS[action]
    
    color_marker = "🟢" if action == 2 else ("🔴" if abs(delta) > 3 else "🟡")
    lines.append(f"  {color_marker} Action: {action_label} (ΔMP = {delta:+.0f} J/min)")
    
    if isinstance(mp, (int, float)):
        target = mp + delta
        lines.append(f"  Target MP: {target:.1f} J/min")
    
    lines.append(f"  Confidence: {prediction['confidence']:.0%}")
    
    if safety_alert:
        lines.append(f"\n  ⚠️ SAFETY ALERT: {safety_alert}")
    
    lines.append("\n── Q-values ──")
    for a, q in enumerate(prediction["q_values"]):
        marker = " ◀" if a == action else ""
        lines.append(f"  {ACTIONS[a]:18s}: {q:+7.2f}{marker}")
    
    lines.append("=" * 50)
    
    output = "\n".join(lines)
    print(output)
    return output


def plot_action_value_heatmap(agent, episodes: List[Dict],
                               feature_list: List[str] = None,
                               save: bool = True) -> plt.Figure:
    """Heatmap of Q-values across states (dimensionality-reduced)."""
    from src.mdp_dataset import flatten_state, ALL_STATE_FEATURES
    if feature_list is None:
        feature_list = ALL_STATE_FEATURES
    
    # Collect Q-values
    mp_vals = []
    q_matrices = []
    
    for ep in episodes:
        for state in ep["states"]:
            mp = state.get("mechanical_power")
            if pd.notna(mp):
                s = flatten_state(state, feature_list)
                pred = agent.predict(s)
                mp_vals.append(mp)
                q_matrices.append(pred["q_values"])
    
    if len(mp_vals) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center")
        return fig
    
    # Bin by MP level
    mp_arr = np.array(mp_vals)
    q_arr = np.array(q_matrices)
    
    bins = np.arange(0, max(mp_arr) + 5, 5)
    bin_labels = [f"{b:.0f}-{b+5:.0f}" for b in bins[:-1]]
    bin_idx = np.digitize(mp_arr, bins) - 1
    
    heatmap_data = []
    actual_labels = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            heatmap_data.append(q_arr[mask].mean(axis=0))
            actual_labels.append(bin_labels[b])
    
    if len(heatmap_data) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data in bins", ha="center")
        return fig
    
    heatmap = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(actual_labels) * 0.5 + 2)))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(len(actual_labels)))
    ax.set_yticklabels(actual_labels)
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels([ACTIONS[a] for a in range(N_ACTIONS)], rotation=45, ha="right")
    ax.set_xlabel("Action")
    ax.set_ylabel("Current MP (J/min)")
    ax.set_title("Q-value Heatmap: Action Preferences by MP Level")
    plt.colorbar(im, ax=ax, label="Mean Q-value")
    
    plt.tight_layout()
    if save:
        fig.savefig(ARTEFACTS_DIR / "q_value_heatmap.png", bbox_inches="tight")
    return fig
