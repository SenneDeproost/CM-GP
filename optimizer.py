# +++ CM-GP/optimizer +++
#
# Genetic Evolution based optimizers
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

import sys
from typing import List
import pygad
import numpy as np
from population import CartesianPopulation
from program.operators import Operator

class PyGADOptimizer:
    def __init__(self, config, operators: List[Operator], state_space) -> None:
        self.config = config
        self.operators = operators
        self.state_dim = state_space.shape[0]

        # Create the initial population
        self.population = self._get_initial_population()

        self.best_solution = self.population[0]
        self.best_fitness = -np.inf

        # Optimizer instance
        self._optim = None

    # Create initial population
    def _get_initial_population(self) -> CartesianPopulation:
        population = CartesianPopulation(
            self.config.n_individuals,
            self.config.n_nodes,
            self.config.n_inputs,
            self.config.max_arity,
            self.operators)
        return population

    # Fit the produced actions to more optimal ones
    def fit(self, critic_states, critic_actions):
        pass


if __name__ == "__main__":

# Test for PyGAD optimizer
config =
optim = PyGADOptimizer()







