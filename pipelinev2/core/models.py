# core/models.py
"""
NiOA-Optimised Bidirectional Deep Recurrent Neural Network.

This module defines:
  1. NinjaOptimizationAlgorithm — A population-based meta-heuristic that
     adaptively balances exploration and exploitation to discover optimal
     DRNN hyperparameter configurations.

  2. create_lstm_model — Constructs and compiles the Bidirectional Stacked
     LSTM regression architecture used throughout this study.

Algorithm Reference
-------------------
NiOA is inspired by the adaptive hunting strategies of ninjas, wherein
agents alternate between broad environmental reconnaissance (exploration)
and directed movement towards the highest-value target (exploitation).
The balance between phases shifts dynamically as optimisation progresses,
favouring exploitation in later iterations.

Author : Anwesha Singh
Dept.  : Computer Science Engineering, Manipal University Jaipur
"""

import time
import numpy as np
from typing import Callable, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    BatchNormalization,
    GlobalMaxPooling1D,
    Dense,
)


# ===========================================================================
# Ninja Optimisation Algorithm
# ===========================================================================

class NinjaOptimizationAlgorithm:
    """
    Ninja Optimisation Algorithm (NiOA).

    Maintains a population of candidate hyperparameter configurations (agents)
    and iteratively refines them through alternating exploration and exploitation
    phases.

    Parameters
    ----------
    objective_function   : Callable  Maps a param dict → float (validation loss).
    bounds               : Dict      Hyperparameter search space definition.
    n_agents             : int       Population size.
    max_iterations       : int       Number of refinement iterations after init.
    exploration_factor   : float     Perturbation scale during exploration.
    exploitation_factor  : float     Attraction strength towards best agent.
    verbose              : bool      Print per-iteration progress if True.
    """

    def __init__(
        self,
        objective_function  : Callable,
        bounds              : Dict[str, Tuple],
        n_agents            : int   = 6,
        max_iterations      : int   = 6,
        exploration_factor  : float = 2.0,
        exploitation_factor : float = 0.5,
        verbose             : bool  = True,
    ):
        self.objective_function  = objective_function
        self.bounds              = bounds
        self.n_agents            = n_agents
        self.max_iterations      = max_iterations
        self.exploration_factor  = exploration_factor
        self.exploitation_factor = exploitation_factor
        self.verbose             = verbose

        self.param_names       = list(bounds.keys())
        self.agents            = self._initialise_population()
        self.fitness           = np.full(n_agents, np.inf)
        self.best_agent        = None
        self.best_fitness      = np.inf
        self.convergence_curve = []

    # ------------------------------------------------------------------
    # Population Initialisation
    # ------------------------------------------------------------------
    def _initialise_population(self) -> List[Dict]:
        """Sample each agent uniformly from the respective search bounds."""
        agents = []
        for _ in range(self.n_agents):
            agent = {}
            for param, (values, kind) in self.bounds.items():
                if kind == "int":
                    agent[param] = int(np.random.randint(values[0], values[1] + 1))
                elif kind == "float":
                    agent[param] = float(np.random.uniform(values[0], values[1]))
                elif kind == "float_log":
                    log_min = np.log10(values[0])
                    log_max = np.log10(values[1])
                    agent[param] = float(10 ** np.random.uniform(log_min, log_max))
                elif kind == "categorical":
                    agent[param] = self._cast(np.random.choice(values))
            agents.append(agent)
        return agents

    @staticmethod
    def _cast(value):
        """Convert numpy scalars to native Python types for JSON safety."""
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.str_):
            return str(value)
        return value

    # ------------------------------------------------------------------
    # Exploration Phase
    # ------------------------------------------------------------------
    def _exploration_phase(self, agent: Dict, iteration: int) -> Dict:
        """
        Perturb agent parameters with a magnitude that decays as
        the optimisation progresses (broad early, narrow late).
        """
        new_agent = agent.copy()
        progress  = iteration / self.max_iterations
        rate      = self.exploration_factor * (1.0 - progress)

        for param, (values, kind) in self.bounds.items():
            if kind == "int":
                step = np.random.randint(
                    -max(1, int(rate)), max(1, int(rate)) + 1
                )
                new_agent[param] = int(
                    np.clip(agent[param] + step, values[0], values[1])
                )

            elif kind == "float":
                span = values[1] - values[0]
                step = np.random.uniform(-rate, rate) * span * 0.1
                new_agent[param] = float(
                    np.clip(agent[param] + step, values[0], values[1])
                )

            elif kind == "float_log":
                # Perturbations are applied in log₁₀ space to preserve
                # the scale-invariant nature of log-uniform parameters.
                log_cur = np.log10(max(agent[param], 1e-12))
                log_min = np.log10(values[0])
                log_max = np.log10(values[1])
                span    = log_max - log_min
                step    = np.random.uniform(-rate, rate) * span * 0.1
                new_agent[param] = float(
                    10 ** np.clip(log_cur + step, log_min, log_max)
                )

            elif kind == "categorical":
                if np.random.rand() < rate * 0.3:
                    new_agent[param] = self._cast(np.random.choice(values))

        return new_agent

    # ------------------------------------------------------------------
    # Exploitation Phase
    # ------------------------------------------------------------------
    def _exploitation_phase(
        self, agent: Dict, best_agent: Dict, iteration: int
    ) -> Dict:
        """
        Move each agent towards the current best configuration,
        with attraction strength that increases as iterations advance.
        """
        new_agent = {}
        progress  = iteration / self.max_iterations
        rate      = self.exploitation_factor * progress

        for param, (values, kind) in self.bounds.items():
            if kind == "int":
                step = int((best_agent[param] - agent[param]) * rate)
                new_agent[param] = int(
                    np.clip(agent[param] + step, values[0], values[1])
                )

            elif kind in ("float", "float_log"):
                step = (best_agent[param] - agent[param]) * rate
                new_agent[param] = float(
                    np.clip(agent[param] + step, values[0], values[1])
                )

            elif kind == "categorical":
                p_exploit = min(0.7 + 0.3 * progress, 0.99)
                new_agent[param] = (
                    best_agent[param] if np.random.rand() < p_exploit
                    else agent[param]
                )

        return new_agent

    # ------------------------------------------------------------------
    # Main Optimisation Loop
    # ------------------------------------------------------------------
    def optimize(self):
        """
        Execute the full NiOA optimisation procedure.

        Returns
        -------
        best_params       : Dict    Optimal hyperparameter configuration.
        best_fitness      : float   Minimum validation MSE loss achieved.
        convergence_curve : List    Best fitness value after each iteration.
        """
        print("Starting Ninja Optimisation Algorithm...\n")
        start_time = time.time()

        # -------------------------------------------------------
        # Step 1: Evaluate the initial randomly sampled population
        # -------------------------------------------------------
        for i, agent in enumerate(self.agents):
            self.fitness[i] = self.objective_function(agent)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_agent   = agent.copy()

        self.convergence_curve.append(self.best_fitness)

        # -------------------------------------------------------
        # Step 2: Iterative refinement
        # -------------------------------------------------------
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\nIteration {iteration + 1}/{self.max_iterations}")

            for i, agent in enumerate(self.agents):
                # Adaptive phase selection: exploitation probability
                # increases monotonically with iteration progress.
                progress  = (iteration + 1) / self.max_iterations
                p_exploit = 0.3 + 0.6 * progress
                use_exploit = (
                    self.best_agent is not None and
                    np.random.rand() < p_exploit
                )

                candidate = (
                    self._exploitation_phase(agent, self.best_agent, iteration)
                    if use_exploit
                    else self._exploration_phase(agent, iteration)
                )

                fitness = self.objective_function(candidate)

                # Greedy selection: retain candidate only if improved
                if fitness < self.fitness[i]:
                    self.agents[i] = candidate
                    self.fitness[i] = fitness

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent   = candidate.copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                print(f"  Best validation loss so far : {self.best_fitness:.8f}")
                print("  Best hyperparameters so far :")
                for k_name, v in self.best_agent.items():
                    print(f"    {k_name}: {v}")

        elapsed_h = (time.time() - start_time) / 3600
        print(f"\nOptimisation completed in {elapsed_h:.2f} hours")

        if self.best_agent is None:
            raise RuntimeError(
                "NiOA failed: no valid hyperparameter configuration was "
                "successfully evaluated. Please inspect the objective function "
                "for runtime errors."
            )

        return self.best_agent, self.best_fitness, self.convergence_curve


# ===========================================================================
# DRNN Model Factory
# ===========================================================================

def create_lstm_model(
    params   : Dict,
    seq_len  : int,
    num_feats: int,
) -> Sequential:
    """
    Construct and compile the Bidirectional Stacked LSTM regression model.

    Architecture
    ------------
    Input(seq_len, num_feats)
      → Bidirectional LSTM(units, return_sequences=True)
      → [LSTM(units, return_sequences=True) + Dropout] × (lstm_layers − 1)
      → BatchNormalization
      → GlobalMaxPooling1D
      → Dense(64, ReLU)
      → Dropout
      → Dense(25, ReLU)
      → Dense(1)        ← scalar regression output

    Parameters
    ----------
    params    : Dict  Hyperparameter configuration from NiOA or manual setting.
    seq_len   : int   Sliding-window sequence length (number of time steps).
    num_feats : int   Number of input feature dimensions.

    Returns
    -------
    tf.keras.Sequential  Compiled model ready for training.
    """
    model = Sequential(name="NiOA_DRNN")
    model.add(Input(shape=(seq_len, num_feats)))

    # First layer — Bidirectional LSTM captures both forward and backward
    # temporal dependencies within the input window.
    model.add(Bidirectional(LSTM(params["units"], return_sequences=True)))

    # Additional stacked LSTM layers with dropout regularisation
    for _ in range(params["lstm_layers"] - 1):
        model.add(LSTM(params["units"], return_sequences=True))
        model.add(Dropout(params["dropout"]))

    # Aggregation and regression head
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(params["dropout"]))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(1))

    # Optimiser construction
    if params["optimizer"] == "adamw":
        optimiser = tf.keras.optimizers.experimental.AdamW(
            learning_rate=params["learning_rate"]
        )
    else:
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        )

    model.compile(loss="mse", optimizer=optimiser, metrics=["mae"])
    return model
