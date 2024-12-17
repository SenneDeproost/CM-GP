# +++ CM-GP/optimizer +++
#
# Genetic Evolution based optimizers
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

import sys
from typing import List
import pygad
import numpy as np
from rich.box import SIMPLE

import test
from population import CartesianPopulation
from program import Operator, SIMPLE_OPERATORS
from config import *
import gymnasium as gym

from program.realization import Program, CartesianProgram


def print_fitness(ga, fitnesses):
    print('F', fitnesses.mean(), file=sys.stderr)

# Todo: Generic class, not fixed to CGP
class PyGADOptimizer:
    def __init__(self,
                 config: OptimizerConfig,
                 operators_dict: dict[int, List[Operator]],
                 state_space: gym.Space) -> None:
        self.config = config
        self.state_space = state_space
        self.operators_dict = operators_dict
        self.operators = [x for y in self.operators_dict.values() for x in y]

        # Create the initial population
        self.population, self.raw_population = self._init_population()
        self.range = self.population.range_description()

        self.best_solution_index = 0
        self.best_fitness = -np.inf
        self.best_program = self.population.realize(self.best_solution_index)

        self._critic_states, self._critic_actions = None, None

        # Optimizer instance
        self._optim = self._init_optimizer()

    # Dunder for preventing certain methods to be pickled (issue with pickling lambda operators)
    #def __getstate__(self):
    #    state = self.__dict__.copy()
    #    # Fields contain elements with non-pickelable lambdas in them
    #    del state['population']
    #    del state['best_program']
    #   del state['operators']
    #   return state

    # Dunder describing
    def __str__(self) -> str:
        return f'PyGAD optimizer with pop size {self.config.n_individuals}, best fit {self.best_fitness}'

    # Create initial population
    def _init_population(self) -> tuple[CartesianPopulation, np.ndarray]:
        c = self.config
        population = CartesianPopulation(c, self.operators_dict, self.state_space)
        raw_population = population.raw_genes()
        return population, raw_population

    # Initialize PyGAD optimizer
    def _init_optimizer(self) -> pygad.GA:
        c = self.config
        instance = pygad.GA(
            # General
            fitness_func=self.fitness_function,
            initial_population=self.raw_population,
            num_generations=c.n_generations,
            keep_parents=c.elitism,
            gene_space=self.range,
            save_solutions=True,
            save_best_solutions=True,
            on_fitness=print_fitness,
            parallel_processing=["process", None],  # Utilize all available resources
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

    def fitness_function(self, _, solution, solution_index) -> float:

        fitness = 0.0

        prog = self.population.realize(solution_index)

        batch_size = self._critic_states.shape[0]

        for i in range(batch_size):

            # Calculate MSE between proposed and improved action
            action = prog(self._critic_states[i])
            desired_action = self._critic_actions[i]

            fitness += (action - desired_action) ** 2

        # Avg
        fitness = -(fitness / batch_size)

        return fitness

    # Fit the produced actions to more optimal ones
    def fit(self, critic_states, critic_actions) -> None:
        # Update internal field for state and actions from the critic
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # Reset initial population inside optimizer
        self._optim.initial_population = self.raw_population  #self.population.raw_genes()

        # Iterate with optimizer
        self._optim.run()

        # Update population
        self.raw_population = self._optim.population
        self.population.update(self._optim.population)

        # Reset best program
        self.best_program = self.population.realize(self.best_solution_index)


if __name__ == "__main__":
    # Test for PyGAD optimizer
    config = OptimizerConfig()
    space = test.SMALL_OBS_SPACE
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS, space)
    print(optim)
