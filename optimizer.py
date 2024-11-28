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
from program.operators import Operator, SIMPLE_OPERATORS
from config import *
import gymnasium as gym

class PyGADOptimizer:
    def __init__(self, config: OptimizerConfig, operators: List[Operator], state_space: gym.Space) -> None:
        self.config = config
        self.operators = operators
        self.state_space = state_space

        # Create the initial population
        self.population = self._init_population()
        # Optimizer instance
        self._optim = self._init_optimizer()

        self.best_solution = self.population[0]
        self.best_fitness = -np.inf

    # Dunder describing
    def __str__(self) -> str:
        return f'PyGAD with pop size {self.config.n_individuals}, best fit {self.best_fitness}'

    # Create initial population
    def _init_population(self) -> CartesianPopulation:
        c = self.config
        population = CartesianPopulation(c, self.operators, self.state_space)
        return population

    # Initialize PyGAD optimizer
    def _init_optimizer(self) -> pygad.GA:
        c = self.config
        instance = pygad.GA(
            # General
            fitness_func=self.fitness_function,
            initial_population=self.population.raw_genes(),
            num_generations=c.n_generations,
            keep_parents= c.elitism,
            save_solutions=True,
            save_best_solutions=True,
            parallel_processing=["processes", None],  # Utilize all available resources
            # Crossover
            mutation_probability=c.gene_mutation_prob,
            mutation_type=c.mutation,
            random_mutation_min_val=c.mutation_val[0],
            random_mutation_max_val=c.mutation_val[1],
            # Mutation
            num_parents_mating=c.n_parents_mating,
            crossover_type=c.crossover,
            parent_selection_type=c.parent_selection
        )

        return instance


    def fitness_function(self, _, soloution, soloution_index) -> float:
        res = -np.inf

        return res

    # Fit the produced actions to more optimal ones
    def fit(self, critic_states, critic_actions) -> None:
        pass



if __name__ == "__main__":

    # Test for PyGAD optimizer
    config = OptimizerConfig()
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS, space)
    print(optim)







