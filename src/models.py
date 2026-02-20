"""
Model implementations:
  - Strategy 1 & 2: XGBoost classifiers
  - Strategy 3: Conservative Q-Learning (CQL) agent
  - State encoder, Q-network, safety filter
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import warnings

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import ACTIONS, ACTION_DELTAS, N_ACTIONS, RL_CONFIG

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# 1. STRATEGY 1 & 2 — XGBOOST BASELINES
# ═══════════════════════════════════════════════════════════

def prepare_xgboost_features_s1(episodes: List[Dict],
                                 feature_list: List[str] = None
                                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Strategy 1: Static features at admission.
    Uses only first-timestep data.
    
    Returns:
        (X, y, feature_names)
    """
    if feature_list is None:
        feature_list = [
            "age", "bmi", "pbw", "mechanical_power",
            "driving_pressure", "pf_ratio", "spo2",
            "heart_rate", "map", "fio2", "peep_final",
            "tidal_volume", "rr", "vt_per_pbw", "mp_per_pbw",
        ]
    
    X, y = [], []
    for ep in episodes:
        if len(ep["states"]) == 0:
            continue
        state = ep["states"][0]  # Admission snapshot
        features = []
        for f in feature_list:
            val = state.get(f, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)
        X.append(features)
        y.append(ep["patient_info"]["hospital_mortality"])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), feature_list


def prepare_xgboost_features_s2(episodes: List[Dict],
                                 feature_list: List[str] = None,
                                 time_points_h: List[int] = None,
                                 windows: List[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Strategy 2: Multi-timepoint features.
    Extracts features at specified hours + trajectory stats.
    
    Returns:
        (X, y, feature_names)
    """
    if feature_list is None:
        feature_list = [
            "mechanical_power", "driving_pressure", "pf_ratio",
            "spo2", "heart_rate", "map", "fio2",
            "peep_final", "vt_per_pbw",
        ]
    # Accept 'windows' as alias for 'time_points_h'
    if time_points_h is None:
        time_points_h = windows if windows is not None else [0, 6, 12, 18, 24]
    
    # Build feature name list
    feat_names = ["age", "bmi", "pbw"]
    for h in time_points_h:
        for f in feature_list:
            feat_names.append(f"{f}_h{h}")
    for f in ["mechanical_power", "driving_pressure", "spo2"]:
        for stat in ["mean", "std", "slope", "max", "min"]:
            feat_names.append(f"{f}_{stat}")
    
    X, y = [], []
    for ep in episodes:
        states = ep["states"]
        T = len(states)
        if T == 0:
            continue
        
        features = []
        
        # Static features
        s0 = states[0]
        for f in ["age", "bmi", "pbw"]:
            val = s0.get(f, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)
        
        # Timepoint features
        for h in time_points_h:
            idx = min(h, T - 1)
            s = states[idx]
            for f in feature_list:
                val = s.get(f, 0.0)
                features.append(float(val) if pd.notna(val) else 0.0)
        
        # Trajectory statistics (over available data, up to 24h)
        max_t = min(T, 25)
        for f in ["mechanical_power", "driving_pressure", "spo2"]:
            vals = [states[t].get(f, np.nan) for t in range(max_t)]
            vals = [v for v in vals if pd.notna(v)]
            if len(vals) > 1:
                features.append(np.mean(vals))
                features.append(np.std(vals))
                features.append(vals[-1] - vals[0])  # Slope proxy
                features.append(max(vals))
                features.append(min(vals))
            else:
                features.extend([0.0] * 5)
        
        X.append(features)
        y.append(ep["patient_info"]["hospital_mortality"])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), feat_names


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   strategy_name: str = "Strategy",
                   params: Dict = None) -> object:
    """Train XGBoost classifier for mortality prediction."""
    from xgboost import XGBClassifier
    
    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / max(n_pos, 1)
    
    # Default params
    default_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_weight,
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    }
    
    # Override with user params
    if params is not None:
        default_params.update(params)
    
    model = XGBClassifier(**default_params)
    
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    
    print(f"\n{strategy_name} — XGBoost trained")
    print(f"  Training samples: {len(y_train)} "
          f"(mortality rate: {y_train.mean():.1%})")
    
    return model


# ═══════════════════════════════════════════════════════════
# 2. STRATEGY 3 — Q-NETWORK (PyTorch)
# ═══════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Strategy 3 (CQL) will not work.")


if TORCH_AVAILABLE:
    
    class StateEncoder(nn.Module):
        """
        Encodes heterogeneous patient state into a latent vector.
        Handles static + snapshot features via MLP.
        (Temporal LSTM extension available for longer sequences.)
        """
        def __init__(self, state_dim: int, latent_dim: int = 128,
                     dropout: float = 0.3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, latent_dim),
            )
        
        def forward(self, x):
            return self.encoder(x)
    
    
    class DuelingQNetwork(nn.Module):
        """
        Dueling Q-Network: Q(s,a) = V(s) + A(s,a) - mean(A).
        With MC dropout for uncertainty estimation.
        """
        def __init__(self, state_dim: int, n_actions: int = 5,
                     hidden_dim: int = 256, latent_dim: int = 128,
                     dropout: float = 0.3):
            super().__init__()
            self.state_encoder = StateEncoder(state_dim, latent_dim, dropout)
            
            self.trunk = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            self.value_head = nn.Linear(hidden_dim, 1)
            self.advantage_head = nn.Linear(hidden_dim, n_actions)
        
        def forward(self, x):
            z = self.state_encoder(x)
            h = self.trunk(z)
            V = self.value_head(h)
            A = self.advantage_head(h)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            return Q
        
        def predict_with_uncertainty(self, x, n_samples: int = 20):
            """MC Dropout uncertainty estimation."""
            self.train()
            with torch.no_grad():
                q_samples = torch.stack([self.forward(x) for _ in range(n_samples)])
            self.eval()
            return q_samples.mean(dim=0), q_samples.std(dim=0)
    
    
    class CQLAgent:
        """
        Conservative Q-Learning agent for offline RL.
        Implements CQL penalty to prevent OOD action overestimation.
        """
        def __init__(self, state_dim: int, n_actions: int = 5,
                     config: Dict = None, **kwargs):
            """
            Args:
                state_dim: dimension of state vector
                n_actions: number of discrete actions
                config: dict with hyperparameters (optional)
                **kwargs: individual hyperparams (override config)
                    hidden_dim, lr, gamma, cql_alpha, tau, batch_size, dropout
            """
            if config is None:
                config = dict(RL_CONFIG)
            else:
                config = dict(config)
            
            # Allow kwargs to override config
            kwarg_map = {
                "hidden_dim": "hidden_dim",
                "lr": "lr",
                "gamma": "gamma",
                "cql_alpha": "cql_alpha",
                "tau": "tau",
                "batch_size": "batch_size",
                "dropout": "dropout",
            }
            for kw, cfg_key in kwarg_map.items():
                if kw in kwargs:
                    config[cfg_key] = kwargs[kw]
            
            self.state_dim = state_dim
            self.n_actions = n_actions
            self.gamma = config.get("gamma", 0.99)
            self.alpha = config.get("cql_alpha", config.get("alpha", 1.0))
            self.lr = config.get("lr", config.get("learning_rate", 3e-4))
            self.batch_size = config.get("batch_size", 256)
            self.tau = config.get("tau", 0.005)
            
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Online and target Q-networks
            self.q_network = DuelingQNetwork(
                state_dim, n_actions,
                hidden_dim=config.get("hidden_dims", [256, 256])[0],
                dropout=config.get("dropout", 0.3),
            ).to(self.device)
            
            self.target_network = DuelingQNetwork(
                state_dim, n_actions,
                hidden_dim=config.get("hidden_dims", [256, 256])[0],
                dropout=config.get("dropout", 0.3),
            ).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimiser = optim.Adam(self.q_network.parameters(), lr=self.lr)
            self.tau = 0.005  # Soft target update
            
            self.training_history = {"loss": [], "q_mean": [], "cql_loss": []}
        
        def _soft_update_target(self):
            """Polyak averaging for target network."""
            for tp, op in zip(self.target_network.parameters(),
                              self.q_network.parameters()):
                tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)
        
        def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
            """
            Single CQL training step.
            
            CQL loss = standard Bellman loss + α × (logsumexp(Q) − Q(s,a_data))
            The CQL penalty pushes down Q-values for unseen actions.
            """
            states = torch.FloatTensor(batch["observations"]).to(self.device)
            actions = torch.LongTensor(batch["actions"]).to(self.device)
            rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
            next_states = torch.FloatTensor(batch["next_observations"]).to(self.device)
            dones = torch.FloatTensor(batch["terminals"]).to(self.device)
            
            # ── Standard DQN Bellman Target ──
            with torch.no_grad():
                next_q = self.target_network(next_states)
                next_v = next_q.max(dim=1)[0]
                target = rewards + (1 - dones) * self.gamma * next_v
            
            # ── Current Q-values ──
            current_q = self.q_network(states)
            q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # ── Bellman loss ──
            bellman_loss = F.mse_loss(q_selected, target)
            
            # ── CQL conservative penalty ──
            # Push down Q-values for all actions (logsumexp)
            # Pull up Q-values for actions actually taken in data
            logsumexp_q = torch.logsumexp(current_q, dim=1).mean()
            data_q = q_selected.mean()
            cql_loss = logsumexp_q - data_q
            
            # ── Total loss ──
            loss = bellman_loss + self.alpha * cql_loss
            
            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimiser.step()
            
            self._soft_update_target()
            
            return {
                "loss": loss.item(),
                "bellman_loss": bellman_loss.item(),
                "cql_loss": cql_loss.item(),
                "q_mean": current_q.mean().item(),
            }
        
        def train(self, states: np.ndarray = None, actions: np.ndarray = None,
                  rewards: np.ndarray = None, next_states: np.ndarray = None,
                  dones: np.ndarray = None,
                  observations: np.ndarray = None, terminals: np.ndarray = None,
                  n_epochs: int = 100, batch_size: int = None,
                  verbose: bool = True) -> Dict:
            """
            Full training loop over the offline dataset.
            
            Accepts either:
              - states, actions, rewards, next_states, dones  (notebook style)
              - observations, actions, rewards, terminals     (d3rlpy style)
            
            Returns training_history dict.
            """
            # Handle both calling conventions
            if states is not None:
                observations = states
            if dones is not None:
                terminals = dones
            
            if observations is None or actions is None or rewards is None:
                raise ValueError("Must provide states/observations, actions, and rewards")
            if terminals is None:
                terminals = np.zeros(len(observations), dtype=np.float32)
            
            if batch_size is not None:
                self.batch_size = batch_size
            
            N = len(observations)
            
            # Construct next_observations if not provided
            if next_states is not None:
                next_observations = next_states
            else:
                next_observations = np.zeros_like(observations)
                next_observations[:-1] = observations[1:]
                next_observations[-1] = observations[-1]
                for i in range(N - 1):
                    if terminals[i] == 1.0:
                        next_observations[i] = observations[i]
            
            steps_per_epoch = max(1, N // self.batch_size)
            
            for epoch in range(n_epochs):
                epoch_losses = []
                indices = np.random.permutation(N)
                
                for step in range(steps_per_epoch):
                    start = step * self.batch_size
                    end = min(start + self.batch_size, N)
                    idx = indices[start:end]
                    
                    if len(idx) < 2:
                        continue
                    
                    batch = {
                        "observations": observations[idx],
                        "actions": actions[idx],
                        "rewards": rewards[idx],
                        "next_observations": next_observations[idx],
                        "terminals": terminals[idx],
                    }
                    
                    metrics = self.train_step(batch)
                    epoch_losses.append(metrics)
                
                if epoch_losses:
                    avg_loss = np.mean([m["loss"] for m in epoch_losses])
                    avg_q = np.mean([m["q_mean"] for m in epoch_losses])
                    avg_cql = np.mean([m["cql_loss"] for m in epoch_losses])
                    
                    self.training_history["loss"].append(avg_loss)
                    self.training_history["q_mean"].append(avg_q)
                    self.training_history["cql_loss"].append(avg_cql)
                    
                    if verbose and (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch+1}/{n_epochs}: "
                              f"loss={avg_loss:.4f}, "
                              f"Q_mean={avg_q:.2f}, "
                              f"CQL={avg_cql:.4f}")
            
            return self.training_history
        
        def predict(self, state: np.ndarray) -> Dict:
            """
            Predict best action for a given state.
            Returns action, Q-values, and uncertainty.
            """
            self.q_network.eval()
            with torch.no_grad():
                x = torch.FloatTensor(state).to(self.device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                q_values = self.q_network(x)
                action = q_values.argmax(dim=1).item()
            
            # Uncertainty via MC dropout
            q_mean, q_std = self.q_network.predict_with_uncertainty(x)
            
            confidence = self._confidence(q_mean[0], q_std[0])
            
            return {
                "action": action,
                "action_label": ACTIONS[action],
                "q_values": q_values[0].cpu().numpy().tolist(),
                "uncertainty": q_std.max().item(),
                "confidence": confidence,
            }
        
        def _confidence(self, q_mean, q_std):
            best_action = q_mean.argmax()
            relative_unc = q_std[best_action] / (q_mean.std() + 1e-6)
            return float(1.0 / (1.0 + relative_unc))
        
        def save(self, path: str):
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions,
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                },
            }, path)
            print(f"Model saved to {path}")
        
        def load(self, path: str):
            ckpt = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(ckpt["q_network"])
            self.target_network.load_state_dict(ckpt["target_network"])
            self.optimiser.load_state_dict(ckpt["optimiser"])
            print(f"Model loaded from {path}")

else:
    # ── Stubs when PyTorch is not installed ──────────────────
    class StateEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for StateEncoder. "
                "Install it with: pip install torch"
            )

    class DuelingQNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for DuelingQNetwork. "
                "Install it with: pip install torch"
            )

    class CQLAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for CQLAgent (Strategy 3). "
                "Install it with: pip install torch"
            )


# ═══════════════════════════════════════════════════════════
# 3. SAFETY FILTER
# ═══════════════════════════════════════════════════════════

class SafetyFilter:
    """
    Hard safety constraints applied post-model.
    Cannot be overridden by the RL agent.
    Based on ARDSNet and ERS/ESICM guidelines.
    """
    
    def filter(self, state: Dict, recommended_action: int
               ) -> Tuple[int, Optional[str]]:
        """
        Apply safety rules. Returns (filtered_action, alert_message).
        """
        alerts = []
        action = recommended_action
        
        spo2 = state.get("spo2")
        pplat = state.get("plateau_pressure")
        dp = state.get("driving_pressure")
        mp = state.get("mechanical_power", 0) or 0
        map_val = state.get("map")
        vt_pbw = state.get("vt_per_pbw")
        
        # Rule 1: Critical hypoxaemia — cannot decrease support
        if pd.notna(spo2) and spo2 < 85:
            if action in [0, 1]:
                action = 2
                alerts.append(
                    f"SpO2={spo2:.0f}% < 85% — blocked MP decrease"
                )
        
        # Rule 2: High Pplat — cannot increase
        if pd.notna(pplat) and pplat > 30:
            if action in [3, 4]:
                action = 1
                alerts.append(
                    f"Pplat={pplat:.0f} cmH2O > 30 — blocked MP increase"
                )
        
        # Rule 3: High driving pressure
        if pd.notna(dp) and dp > 15:
            if action in [3, 4]:
                action = 1
                alerts.append(
                    f"ΔP={dp:.0f} cmH2O > 15 — blocked MP increase"
                )
        
        # Rule 4: MP ceiling
        target_mp = mp + ACTION_DELTAS.get(action, 0)
        if target_mp > 30:
            if action in [3, 4]:
                action = 2
                alerts.append(
                    f"Target MP={target_mp:.1f} > 30 J/min — capped"
                )
        
        # Rule 5: MP floor
        if target_mp < 5:
            if action in [0, 1]:
                action = 2
                alerts.append(
                    f"Target MP={target_mp:.1f} < 5 J/min — capped"
                )
        
        # Rule 6: Haemodynamic instability
        if pd.notna(map_val) and map_val < 60:
            if action in [0, 4]:
                action = 1 if recommended_action == 0 else 3
                alerts.append(
                    f"MAP={map_val:.0f} mmHg < 60 — no large changes"
                )
        
        # Rule 7: VT/PBW already high
        if pd.notna(vt_pbw) and vt_pbw > 7.5:
            if action in [3, 4]:
                action = 2
                alerts.append(
                    f"VT/PBW={vt_pbw:.1f} mL/kg > 7.5 — cannot increase"
                )
        
        alert_msg = "; ".join(alerts) if alerts else None
        return action, alert_msg
    
    # Alias for convenience
    check = filter
