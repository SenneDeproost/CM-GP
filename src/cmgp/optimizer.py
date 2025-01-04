# +++ CM-GP/optimizer +++
#
# Genetic Evolution based optimizers
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

import sys
from copy import copy
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
    print(f'F_best: {fitnesses.max()}, F_worst: {fitnesses.min()}  F_mean: {fitnesses.mean()}', file=sys.stderr)


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
            suppress_warnings=True,
            fitness_func=self.fitness_function,
            initial_population=copy(self.raw_population),
            num_generations=c.n_generations,
            keep_elitism=c.elitism,
            gene_space=self.range,
            #gene_space={'low': 0, 'high': 10},
            save_solutions=False,
            save_best_solutions=False,
            on_fitness=print_fitness,
            parallel_processing=16,  # Utilize all available resources
            # Mutation
            mutation_probability=c.gene_mutation_prob,
            #mutation_percent_genes=c.gene_mutation_percent,
            mutation_type=c.mutation,
            random_mutation_min_val=c.mutation_val[0],
            random_mutation_max_val=c.mutation_val[1],
            # Crossover
            num_parents_mating=1,
            crossover_type=c.crossover,
            parent_selection_type=c.parent_selection
        )

        return instance

    def fitness_function(self, _, solution, solution_index) -> float:
        fitness = 0.0

        prog = self.population.realize(solution_index)

        batch_size = self._critic_states.shape[0]

        for i in range(batch_size):
            state = self._critic_states[i]

            # Calculate MSE between proposed and improved action
            action = prog(state)
            desired_action = self._critic_actions[i]

            distance = (desired_action - action) ** 2

            # Checking bug for 0 producing programs
            #if action == 0:
            #    fitness += 99
            #else:
            fitness += distance

        # Avg
        fitness = - fitness
        #fitness = (fitness / batch_size)

        # Set best solution index if needed
        #if fitness > self.best_fitness:
        #    print(f'--- New best program {prog} with fitness: {fitness} ---')
        #    self.best_fitness = fitness
        #    self.best_solution_index = solution_index
        #    self.best_program = prog

        return fitness

    # Reset solutions in the optimizer
    # Source:
    # https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/9eeaa8d5b50002f3370f9b6e150eb63e141219bb/pygad/pygad.py#L2345
    def reset_solutions(self) -> None:
        optim = self._optim

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it
        # after saving the object. A list holding the fitness value of the best solution for each generation.
        optim.best_solutions_fitness = []

        # The generation number at which the best fitness value is reached. It is only assigned the generation number
        # after the `run()` method completes. Otherwise, its value is -1.
        optim.best_solution_generation = -1

        optim.best_solutions = []  # Holds the best solution in each generation.

        optim.solutions = []  # Holds the solutions in each generation.
        # Holds the fitness of the solutions in each generation.
        optim.solutions_fitness = []

        # A list holding the fitness values of all solutions in the last generation.
        optim.last_generation_fitness = None
        # A list holding the parents of the last generation.
        optim.last_generation_parents = None
        # A list holding the offspring after applying crossover in the last generation.
        optim.last_generation_offspring_crossover = None
        # A list holding the offspring after applying mutation in the last generation.
        optim.last_generation_offspring_mutation = None
        # Holds the fitness values of one generation before the fitness values saved in the last_generation_fitness
        # attribute. Added in PyGAD 2.16.2.
        optim.previous_generation_fitness = None
        # Added in PyGAD 2.18.0. A NumPy array holding the elitism of the current generation according to the value
        # passed in the 'keep_elitism' parameter. It works only if the 'keep_elitism' parameter has a non-zero value.
        optim.last_generation_elitism = None
        # Added in PyGAD 2.19.0. A NumPy array holding the indices of the elitism of the current generation. It works
        # only if the 'keep_elitism' parameter has a non-zero value.
        optim.last_generation_elitism_indices = None
        # Supported in PyGAD 3.2.0. It holds the pareto fronts when solving a multi-objective problem.
        optim.pareto_fronts = None

    # Fit the produced actions to more optimal ones
    def fit(self, critic_states, critic_actions) -> None:
        # Update internal field for state and actions from the critic
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # ! Limit the amount of critic state actions
        #self._critic_states = critic_states[:5]
        #self._critic_actions = critic_actions[:5]

        # Reset initial population inside optimizer
        self._optim.initial_population = self.raw_population  #self.population.raw_genes()

        # Iterate with optimizer
        #self._optim = self._init_optimizer()
        self.reset_solutions()
        self._optim.run()

        # Update population
        r = []
        for i in range(100):
            print()
            print(i)
            prog = self.population.random_program()
            r.append(prog)
            print(prog)
            nodes = prog._realization['all']
            for node in nodes:
                print(node)
            print()


        print()
        res = self.population.realize_all()
        for r in res:
            print(r)
        print()

        self.raw_population = self._optim.population
        self.population.update(self._optim.population)

        # Set best results
        best_sol, best_fit, best_idx = self._optim.best_solution()
        self.best_solution_index = best_idx
        self.best_fitness = best_fit
        self.best_program = self.population.realize(self.best_solution_index)


if __name__ == "__main__":
    # Test for PyGAD optimizer
    config = OptimizerConfig()
    space = test.SMALL_OBS_SPACE
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS, space)
    print(optim)
