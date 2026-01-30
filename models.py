# models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dropout,
    BatchNormalization, GlobalMaxPooling1D, Dense
)

class NinjaOptimizationAlgorithm:
    def __init__(
        self,
        objective_function,
        bounds,
        n_agents,
        max_iterations,
        exploration_factor,
        exploitation_factor,
        verbose=True
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

    def _initialize_population(self):
        agents = []
        for _ in range(self.n_agents):
            agent = {}
            for param, bound in self.bounds.items():
                if bound[1] == 'int':
                    agent[param] = int(
                        np.random.randint(bound[0][0], bound[0][1] + 1)
                    )
                elif bound[1] == 'float':
                    agent[param] = float(
                        np.random.uniform(bound[0][0], bound[0][1])
                    )
                elif bound[1] == 'float_log':
                    agent[param] = float(
                        10 ** np.random.uniform(
                            np.log10(bound[0][0]),
                            np.log10(bound[0][1])
                        )
                    )
                elif bound[1] == 'categorical':
                    agent[param] = np.random.choice(bound[0])
            agents.append(agent)
        return agents

    def _exploration_phase(self, agent, iteration):
        new_agent = agent.copy()
        progress = iteration / self.max_iterations

        for param, bound in self.bounds.items():
            rate = self.exploration_factor * (1 - progress)
            if bound[1] in ['float', 'float_log']:
                step = np.random.uniform(-rate, rate) * (bound[0][1] - bound[0][0]) * 0.1
                new_agent[param] = np.clip(agent[param] + step, bound[0][0], bound[0][1])
            elif bound[1] == 'int':
                step = np.random.randint(-int(rate), int(rate) + 1)
                new_agent[param] = int(np.clip(agent[param] + step, bound[0][0], bound[0][1]))
            elif bound[1] == 'categorical':
                if np.random.rand() < rate * 0.3:
                    new_agent[param] = np.random.choice(bound[0])
        return new_agent

    def _exploitation_phase(self, agent, best_agent, iteration):
        new_agent = {}
        progress = iteration / self.max_iterations

        for param, bound in self.bounds.items():
            rate = self.exploitation_factor * progress
            if bound[1] in ['float', 'float_log']:
                step = (best_agent[param] - agent[param]) * rate
                new_agent[param] = np.clip(agent[param] + step, bound[0][0], bound[0][1])
            elif bound[1] == 'int':
                step = int((best_agent[param] - agent[param]) * rate)
                new_agent[param] = int(np.clip(agent[param] + step, bound[0][0], bound[0][1]))
            elif bound[1] == 'categorical':
                new_agent[param] = best_agent[param] if np.random.rand() < 0.7 else agent[param]
        return new_agent

    def optimize(self):
        for iteration in range(self.max_iterations):
            for i, agent in enumerate(self.agents):
                phase = (
                    'exploitation'
                    if self.best_agent is not None and np.random.rand() < 0.3 + 0.6 * iteration / self.max_iterations
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
                print(f"Iteration {iteration+1}: Best Loss = {self.best_fitness:.6f}")

        return self.best_agent, self.best_fitness, self.convergence_curve


def create_lstm_model(params, seq_len, num_feats):
    model = Sequential(name="LSTM_NiOA")
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

    optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=params['learning_rate']
)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
