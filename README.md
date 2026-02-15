# Energy Consumption Prediction in Data Centres using Deep Recurrent Neural Networks Optimised by Ninja Optimisation Algorithm (NiOA)

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

---

## Key Works

The major works in this project are:

1. **Multi-Horizon Forecasting Reformulation**

   Instead of predicting micro-level instantaneous fluctuations, the target is reformulated as:

   \[
   ΔE_k(t) = E(t+k) - E(t)
   \]

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

## Project Structure


├── config.py
├── data_preprocessing.py
├── models.py
├── train.py
├── evaluate.py
├── utils.py
├── results/
│ ├── models/
│ ├── predictions/
│ ├── plots/
│ └── metrics.json
└── notebooks/


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

## Future Work

- Completion of full benchmarking experiments
- Statistical significance testing

## Author

Anwesha Singh  
B.Tech (Computer Science Engineering)  
Manipal University Jaipur  


