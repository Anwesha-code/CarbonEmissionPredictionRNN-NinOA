import numpy as np
import time
from typing import Dict, Tuple, List, Callable
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dropout,
    BatchNormalization, GlobalMaxPooling1D, Dense
)


class NinjaOptimizationAlgorithm:

    def __init__(
        self,
        objective_function: Callable,
        bounds: Dict[str, Tuple],
        n_agents: int = 6,
        max_iterations: int = 6,
        exploration_factor: float = 2.0,
        exploitation_factor: float = 0.5,
        verbose: bool = True
    ):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_agents = n_agents
        self.max_iterations = max_iterations
        self.exploration_factor = exploration_factor
        self.exploitation_factor = exploitation_factor
        self.verbose = verbose

        self.param_names = list(bounds.keys())
        self.agents = self._initialize_population()
        self.fitness = np.full(n_agents, np.inf)

        self.best_agent = None
        self.best_fitness = np.inf
        self.convergence_curve = []

    def _initialize_population(self) -> List[Dict]:
        agents = []
        for _ in range(self.n_agents):
            agent = {}
            for param, bound in self.bounds.items():
                values, kind = bound

                if kind == 'int':
                    agent[param] = int(np.random.randint(values[0], values[1] + 1))
                elif kind == 'float':
                    agent[param] = float(np.random.uniform(values[0], values[1]))
                elif kind == 'float_log':
                    log_min, log_max = np.log10(values[0]), np.log10(values[1])
                    agent[param] = float(10 ** np.random.uniform(log_min, log_max))
                elif kind == 'categorical':
                    agent[param] = np.random.choice(values)

            agents.append(agent)
        return agents

    def _exploration_phase(self, agent: Dict, iteration: int) -> Dict:
        new_agent = agent.copy()
        progress = iteration / self.max_iterations

        for param, bound in self.bounds.items():
            values, kind = bound
            exploration_rate = self.exploration_factor * (1 - progress)

            if kind == 'int':
                step = np.random.randint(-int(exploration_rate), int(exploration_rate) + 1)
                new_agent[param] = int(np.clip(agent[param] + step, values[0], values[1]))

            elif kind in ['float', 'float_log']:
                step = np.random.uniform(-exploration_rate, exploration_rate)
                new_agent[param] = float(np.clip(agent[param] + step, values[0], values[1]))

            elif kind == 'categorical' and np.random.rand() < exploration_rate * 0.3:
                new_agent[param] = np.random.choice(values)

        return new_agent

    def _exploitation_phase(self, agent: Dict, best_agent: Dict, iteration: int) -> Dict:
        new_agent = {}
        progress = iteration / self.max_iterations

        for param, bound in self.bounds.items():
            values, kind = bound
            exploitation_rate = self.exploitation_factor * progress

            if kind == 'int':
                step = int((best_agent[param] - agent[param]) * exploitation_rate)
                new_agent[param] = int(np.clip(agent[param] + step, values[0], values[1]))

            elif kind in ['float', 'float_log']:
                step = (best_agent[param] - agent[param]) * exploitation_rate
                new_agent[param] = float(np.clip(agent[param] + step, values[0], values[1]))

            elif kind == 'categorical':
                new_agent[param] = best_agent[param] if np.random.rand() < 0.7 else agent[param]

        return new_agent

    def optimize(self):
        print("Starting Ninja Optimization Algorithm...\n")
        start_time = time.time()

        for i, agent in enumerate(self.agents):
            self.fitness[i] = self.objective_function(agent)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_agent = agent.copy()

        self.convergence_curve.append(self.best_fitness)

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\nIteration {iteration + 1}/{self.max_iterations}")

            for i, agent in enumerate(self.agents):
                phase = (
                    'exploitation'
                    if self.best_agent and np.random.rand() < (0.3 + 0.6 * iteration / self.max_iterations)
                    else 'exploration'
                )

                candidate = (
                    self._exploitation_phase(agent, self.best_agent, iteration)
                    if phase == 'exploitation'
                    else self._exploration_phase(agent, iteration)
                )

                fitness = self.objective_function(candidate)

                if fitness < self.fitness[i]:
                    self.agents[i] = candidate
                    self.fitness[i] = fitness

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = candidate.copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                print("  Best validation loss so far:", self.best_fitness)
                print("  Best hyperparameters so far:")
                for k, v in self.best_agent.items():
                    print(f"    {k}: {v}")

        elapsed = (time.time() - start_time) / 3600
        print(f"\nOptimization completed in {elapsed:.2f} hours")

        if self.best_agent is None:
            raise RuntimeError(
                "NinOA failed: no valid hyperparameter configuration was found. "
                "Check objective_function_lstm for errors."
            )

        return self.best_agent, self.best_fitness, self.convergence_curve


def create_lstm_model(params: Dict, seq_len: int, num_feats: int) -> Sequential:
    model = Sequential(name='LSTM_NiOA')
    model.add(Input(shape=(seq_len, num_feats)))

    model.add(Bidirectional(LSTM(params['units'], return_sequences=True)))

    for _ in range(params['lstm_layers'] - 1):
        model.add(LSTM(params['units'], return_sequences=True))
        model.add(Dropout(params['dropout']))

    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    if params['optimizer'] == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=params['learning_rate']
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate']
        )

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
