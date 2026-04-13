# Energy Consumption Prediction in Data Centres using Deep Recurrent Neural Networks Optimised by Ninja Optimisation Algorithm (NiOA)

# Author

Anwesha Singh  
B.Tech (Computer Science Engineering)  
Manipal University Jaipur  
 

---

## Overview

This project aims to present a reproducible deep learning framework for short-term energy increment prediction in data centre environments. The proposed project uses Bidirectional Deep Recurrent Neural Network (DRNN) architecture, optimised using the Ninja Optimisation Algorithm (NiOA), to work on complex temporal dependencies in multi-sensor energy data.

The primary objective of this work is to design a robust and leakage-free forecasting pipeline capable of multi-horizon prediction while maintaining strict time-series integrity.

## Research Motivation

Data centres are major contributors to global energy consumption and carbon emissions. Accurate short-term forecasting of energy increments enables:

- Proactive carbon footprint optimisation  
- Intelligent workload scheduling  
- Improved infrastructure efficiency  
- Sustainable computing practices  

However, energy data is highly temporal, noisy, and multi-dimensional. Classical machine learning models often fail to capture long-term dependencies inherent in such data. Also, manual hyperparameter tuning is inefficient and tedious.

This project has tried to work on addresses challenges using:

- Deep Recurrent Neural Networks (Bidirectional LSTM)
- Meta-heuristic hyperparameter optimisation using NiOA
- Multi-horizon cumulative forecasting formulation
- Strict reproducibility and anti-leakage design

## Key Works

The major works in this project are:

1. **Multi-Horizon Forecasting Reformulation**

   Instead of predicting micro-level instantaneous fluctuations, the target is reformulated as:

   \
   ΔE_k(t) = E(t+k) - E(t)
   \

   This converts noisy short-term prediction into cumulative forecasting across multiple horizons (1 second to 15 minutes), improving signal strength and stability.

2. **Strict Time-Series Correctness**

   - Chronological timestamp-based splitting  
   - No random shuffling  
   - Train-only feature scaling  
   - No target scaling  
   - Deterministic reproducibility  

3. **Bidirectional Stacked DRNN Architecture**

   The model architecture consists of:

   - Bidirectional LSTM layer  
   - Additional stacked LSTM layers  
   - Dropout regularisation  
   - Batch Normalisation  
   - Global pooling  
   - Fully connected layers  

   This layerd architecture leads to robust contextual learning.

4. **Ninja Optimisation Algorithm (NiOA)**

   Hyperparameters are tuned using a meta-heuristic optimisation strategy inspired by ninjas' adaptive exploration–exploitation balance mechanisms. 


## Dataset 

The dataset consists of real-world data centre sensor measurements collected over multiple months. It includes:

- Electrical readings (voltage, current, power, energy)
- System workload readings (CPU utilisation)
- Thermal radings (sensor temperature)
- Timestamped high-frequency sampling (at every second)

Preprocessing steps include:

- Timestamp sorting
- Forward-fill missing value handling
- Z-score based outlier removal
- Feature engineering (temporal features, lag variables)
- Train-only standardisation
---

## Directory Structure

```
CarbonEmissionPredictionRNN-NinOA/
│
├── core/                          # Shared library — used by all notebooks
│   ├── config.py                  # Master configuration (paths, seeds, NiOA params)
│   ├── preprocessing.py           # Data loading, cleaning, target engineering, splitting
│   ├── models.py                  # NinjaOptimizationAlgorithm + create_lstm_model
│   ├── train.py                   # TimeLimitCallback + objective_function_lstm
│   ├── evaluate.py                # compute_metrics + evaluate_keras_model
│   └── utils.py                   # Seeds, scaler, sequences, tf.data, GPU setup
│
├── notebooks/                     # Jupyter notebooks — run in order
│   ├── 01_NiOA_DRNN_Training.ipynb       # Main: optimise + train + save splits
│   ├── 02_Benchmark_Classical_ML.ipynb   # LR, SVR, XGBoost, MLP
│   ├── 03_Benchmark_Deep_Learning.ipynb  # Vanilla LSTM, CNN-LSTM
│   └── 04_Multi_Horizon_Analysis.ipynb   # Aggregate results, publication plots
│
├── benchmarking/
│   └── utils/
│       ├── data_loader.py         # Loads frozen canonical splits
│       └── metrics.py             # Shared metrics + result saving + table builder
│
├── data/
│   ├── raw/                       # Place 2agosto -dic 2021.csv here
│   └── processed/                 # (Optional) intermediate processed files
│
├── results/                       # All outputs — never modify manually
│   ├── splits/
│   │   └── horizon_{k}/           # Canonical frozen splits for each horizon
│   │       ├── X_train.npy
│   │       ├── y_train.npy
│   │       ├── X_val.npy
│   │       ├── y_val.npy
│   │       ├── X_test.npy
│   │       ├── y_test.npy
│   │       ├── scaler.pkl
│   │       ├── feature_cols.json
│   │       └── split_metadata.json
│   │
│   ├── nioa_drnn/
│   │   └── k{HORIZON}_seq{SEQ}_{TIMESTAMP}/
│   │       ├── model/NiOA_DRNN_k{k}.h5
│   │       ├── predictions/{y_test_true, y_test_pred}.npy
│   │       ├── plots/{pred_vs_actual, residuals, training_curve,
│   │       │         nioa_convergence, time_series_overlay}.png
│   │       ├── best_params.json
│   │       ├── convergence.npy
│   │       ├── metrics.json
│   │       ├── scaler.pkl
│   │       └── training_config.json
│   │
│   ├── benchmark/
│   │   └── horizon_{k}/
│   │       ├── {model_name}/
│   │       │   ├── metrics.json
│   │       │   ├── y_test_pred.npy
│   │       │   ├── y_test_true.npy
│   │       │   └── result_summary.json
│   │       └── summary_all.csv
│   │
│   └── analysis/
│       ├── full_results.csv
│       ├── mae_pivot.csv
│       ├── mae_vs_horizon.png
│       ├── r2_vs_horizon.png
│       └── mae_bar_all_horizons.png
│
├── requirements.txt
└── README.md
```

## Methodology

The experimental workflow follows these steps:

1. Load and clean raw data
2. Chronological train–validation–test split
3. Train-only feature scaling
4. Sequence generation via sliding window
5. Hyperparameter optimisation using NiOA
6. Final model training with early stopping
7. Evaluation using MAE, RMSE, R², and MAPE
8. Artifact saving for reproducibility

## Multi-Horizon Framework (In Progress)


- Very-short term (1-second increments)
- Short-term cumulative (1 minute)
- Medium-term cumulative (5 minutes)
- Extended short-term cumulative (15 minutes)

## Benchmarking Framework (In Progress)

- Linear Regression  
- Support Vector Regression  
- XGBoost  
- Multi-Layer Perceptron  
- ARIMA (univariate baseline)  
- LSTM (baseline)  
- CNN-LSTM hybrid  
- DRNN + Bayesian optimisation (Optuna) 
- DRNN + NiOA  


---

## Execution Order

### Step 1 — Prepare environment

```bash
pip install -r requirements.txt
```

Place the raw data file inside `data/raw/`:

```
data/raw/2agosto -dic 2021.csv
```

### Step 2 — Train the proposed NiOA-DRNN model

Open `notebooks/01_NiOA_DRNN_Training.ipynb`.

Set `HORIZON` in Cell 2 to the desired horizon, then run all cells.  
Repeat for each horizon: **1, 60, 300, 900, 1800**.

This notebook:
- Preprocesses the data
- Splits chronologically (70 / 15 / 15 %)
- **Saves canonical splits** (used identically by all benchmark models)
- Runs NiOA optimisation on a 30 % subset
- Trains the final DRNN with best hyperparameters
- Saves model, predictions, metrics, and plots

### Step 3 — Run benchmark models

For each horizon, open and run:
- `notebooks/02_Benchmark_Classical_ML.ipynb`  (Linear Regression, SVR, XGBoost, MLP)
- `notebooks/03_Benchmark_Deep_Learning.ipynb` (Vanilla LSTM, CNN-LSTM)

Set `HORIZON` to match the splits you wish to evaluate.

### Step 4 — Aggregate and analyse

Run `notebooks/04_Multi_Horizon_Analysis.ipynb` to:
- Build the full cross-model, cross-horizon comparison table
- Generate MAE vs horizon, R² vs horizon, and bar chart figures

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Chronological split | Prevents data leakage; simulates real deployment |
| Train-only scaler | Future data must not influence feature normalisation |
| Target never scaled | Eliminates inverse-transform mismatch errors |
| Canonical frozen splits | Guarantees identical evaluation conditions for all models |
| tf.data generator | Avoids GPU OOM errors on large sequence arrays |
| sMAPE with epsilon | Prevents division-by-zero for near-zero increments |

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| MAE | mean( |y − ŷ| ) |
| RMSE | √ mean( (y − ŷ)² ) |
| R² | 1 − SS_res / SS_tot |
| sMAPE | 100 × mean( 2|y−ŷ| / (|y|+|ŷ|+ε) ) |

---

## Notes on Short-Horizon Performance

At k = 1 second, the prediction target (ΔE) is dominated by sensor noise
and exhibits very low magnitude.  The model tends to predict near the
conditional mean, resulting in R² values close to zero or mildly negative.
This is a known characteristic of fine-resolution energy forecasting and
is not indicative of a model deficiency.  Performance improves
systematically as the horizon increases, consistent with the theoretical
expectation that longer aggregation windows reduce the relative noise
variance.
