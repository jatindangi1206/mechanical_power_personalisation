# Mechanical Power Personalisation for ICU Patients

A Clinical Decision Support System using **offline reinforcement learning** (Conservative Q-Learning) to optimise mechanical ventilation settings for ICU patients, with the goal of minimising ventilator-induced lung injury while maintaining adequate oxygenation.

## Architecture

Three strategies are implemented in increasing sophistication:

| Strategy | Method | Input | Output |
|----------|--------|-------|--------|
| **S1** | Static XGBoost | Admission snapshot (15 features) | Mortality risk score |
| **S2** | Time-Window XGBoost | Multi-timepoint trajectory (126 features) | Mortality risk score |
| **S3** | CQL Offline RL | 27-dim state vector per timestep | 5-action MP adjustment policy |

### Action Space (5 discrete actions)
- `large_decrease` (-5 J/min), `small_decrease` (-2), `maintain` (0), `small_increase` (+2), `large_increase` (+5)

### Safety Filter
Seven hard-coded clinical rules (ARDSNet/ERS guidelines) that cannot be overridden by the RL agent, including SpO2 floor, Pplat ceiling, driving pressure limits, MP bounds, haemodynamic stability, and VT/PBW constraints.

## Project Structure

```
mechanical_power_personalisation/
├── config/
│   └── settings.py          # All constants, ItemIDs, hyperparameters
├── src/
│   ├── data_extraction.py   # MIMIC-III loading & cohort construction
│   ├── preprocessing.py     # Outlier removal, MP calculation, resampling
│   ├── mdp_dataset.py       # MDP episodes, action discretisation, rewards
│   ├── models.py            # XGBoost baselines, CQL agent, safety filter
│   ├── evaluation.py        # Metrics, OPE, safety audit, subgroup analysis
│   └── explainability.py    # Visualisation & clinical dashboard functions
├── notebooks/
│   └── mp_personalisation_pipeline.ipynb  # Full 10-phase orchestration
├── artefacts/               # Saved models, figures, CSVs (auto-created)
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib seaborn tqdm
```

### 2. Set data path
Edit `config/settings.py` and set `DATA_DIR` to your MIMIC-III database directory.

### 3. Run the notebook
Open `notebooks/mp_personalisation_pipeline.ipynb` and run all cells sequentially. The notebook covers 10 phases:

1. **Data Extraction** — Cohort building, ventilator/vital/lab extraction
2. **Preprocessing** — Outlier removal, MP calculation (Gattinoni 2016), hourly resampling
3. **MDP Construction** — Episode building, action discretisation, reward function
4. **S1 Baseline** — Static XGBoost mortality classifier
5. **S2 Baseline** — Time-window XGBoost with trajectory features
6. **S3 CQL** — Conservative Q-Learning with Dueling DQN + MC Dropout
7. **Evaluation** — Classification metrics, OPE, safety audit
8. **Comparison** — Strategy comparison table
9. **Visualisation** — Policy comparison, Q-value heatmaps, clinical decisions
10. **Summary** — Pipeline results and next steps

## Key Technical Details

- **Mechanical Power** (Gattinoni 2016): `0.098 × RR × VT × (Ppeak - 0.5×ΔP)`
- **State vector**: 27 dimensions (3 static + 15 snapshot + 7 lab + 2 derived)
- **Reward function**: Terminal (±100 mortality), intermediate (oxygenation, haemodynamics, lactate), VILI penalty, weaning bonus
- **CQL loss**: Standard Bellman + α × (logsumexp(Q) - Q(s, a_data))
- **Uncertainty**: MC Dropout (30 forward passes) for epistemic uncertainty

## Data

Uses the MIMIC-III Clinical Database Demo (100 patients). For production use, scale to full MIMIC-III/IV (>15,000 ventilated stays).

## Validated Results (Demo Data)

- **Cohort**: 83 ICU stays, 31.3% hospital mortality
- **Episodes**: 81 episodes, 8,138 transitions
- **S1 AUROC**: 0.68 | **S2 AUROC**: 0.50 (limited by small sample)
- **CQL**: Successfully trains and learns conservative policy
- **Safety filter**: Correctly overrides unsafe actions