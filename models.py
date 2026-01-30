# models.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout,
    BatchNormalization, GlobalMaxPooling1D, Input
)

class NinjaOptimizationAlgorithm:
    def __init__(self, objective_function, bounds,
                 n_agents, max_iterations,
                 exploration_factor, exploitation_factor):

        self.objective_function = objective_function
        self.bounds = bounds
        self.n_agents = n_agents
        self.max_iterations = max_iterations
        self.exploration_factor = exploration_factor
        self.exploitation_factor = exploitation_factor

        self.param_names = list(bounds.keys())
        self.agents = self._initialize_agents()
        self.fitness = np.full(n_agents, np.inf)
        self.best_agent = None
        self.best_fitness = np.inf
        self.convergence_curve = []

    def _initialize_agents(self):
        agents = []
        for _ in range(self.n_agents):
            agent = {}
            for param, (values, kind) in self.bounds.items():
                if kind == "int":
                    agent[param] = np.random.randint(values[0], values[1] + 1)
                elif kind == "float":
                    agent[param] = np.random.uniform(values[0], values[1])
                elif kind == "categorical":
                    agent[param] = np.random.choice(values)
            agents.append(agent)
        return agents

    def optimize(self):
        for iteration in range(self.max_iterations):
            for i, agent in enumerate(self.agents):
                fitness = self.objective_function(agent)

                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = agent

            self.convergence_curve.append(self.best_fitness)

        return self.best_agent, self.best_fitness, self.convergence_curve


def create_lstm_model(params, seq_len, n_features):
    model = Sequential()
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(params["units"], return_sequences=True))
    model.add(Dropout(params["dropout"]))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=params["learning_rate"]
    )

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model
