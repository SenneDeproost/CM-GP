# +++ CM-GP/optimizer +++
#
# Genetic Evolution based optimizers
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

import sys
import torch
from copy import copy
from typing import List
import pygad
import numpy as np
from rich.box import SIMPLE
from stable_baselines3.common.buffers import ReplayBuffer

import test
from population import CartesianPopulation
from critic import Critic
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
                 state_space: gym.Space,
                 buffer: ReplayBuffer = None,
                 critic: Critic = None,
                 buffer_batch_size: int = None,
                 action_space: gym.spaces.Box = None) -> None:
        self.config = config
        self.state_space = state_space
        self.operators_dict = operators_dict
        self.operators = [x for y in self.operators_dict.values() for x in y]

        # Use buffer to sample if present
        self.buffer = buffer
        self.critic = critic
        if self.buffer is not None:
            assert critic is not None, "Buffer given but no critic to calculate improved actions"
        self.buffer_batch_size = buffer_batch_size

        # Create the initial population
        self.population = self._init_population()
        self.range = self.population.range_description()

        self.best_solution_index = 0
        self.best_fitness = -np.inf
        self.best_program = self.population.realize(self.best_solution_index)

        self.action_space = action_space

        self._critic_states, self._critic_actions = None, None

        # Optimizer instance
        self._optim = self._init_optimizer()

        self.env = None

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
    def _init_population(self) -> CartesianPopulation:
        c = self.config
        population = CartesianPopulation(c, self.operators_dict, self.state_space)
        return population

    # on_generation function to sample new experiences from buffer
    def new_sample(self, ga) -> None:
        # Check if buffer is given in the optimizer
        if self.buffer is not None:
            self._critic_states = self.buffer.sample(self.buffer_batch_size).observations.detach().numpy().astype(np.float32)


    # Initialize PyGAD optimizer
    def _init_optimizer(self) -> pygad.GA:
        c = self.config
        instance = pygad.GA(
            # General
            suppress_warnings=True,
            fitness_func=self.fitness_function,
            initial_population=self.population.individuals,
            num_generations=c.n_generations,
            keep_elitism=c.elitism,
            gene_space=self.range,
            save_solutions=False,
            save_best_solutions=True,
            on_fitness=print_fitness,
            on_generation=self.new_sample,
            parallel_processing=1,  # Utilize all available resources
            # Mutation
            mutation_probability=c.gene_mutation_prob,
            #mutation_percent_genes=c.gene_mutation_percent,
            mutation_type=c.mutation,
            random_mutation_min_val=c.mutation_val[0],
            random_mutation_max_val=c.mutation_val[1],
            # Crossover
            num_parents_mating=c.n_parents_mating,
            crossover_type=c.crossover,
            parent_selection_type=c.parent_selection
        )

        return instance

    def fitness_function_old(self, _, solution, solution_index) -> float:
        # Update population and realize at index
        prog = self.population.realize(solution_index)

        # Compute program actions
        prog_actions = np.array([prog(state) for state in self._critic_states]).reshape((-1,1))#.astype(np.float32)
        if self.action_space is not None:
            prog_actions = prog_actions.clip(self.action_space.low, self.action_space.high)

        # Change large numbers to arbitrary large number
        #prog_actions[prog_actions==torch.inf] = 9e6

        #print(prog_actions)

        if self.buffer is not None:
            desired_actions, deltas = self.critic.improve_actions(prog_actions.astype(np.float32), self._critic_states)
            #desired_actions = desired_actions.astype(np.float32)
        else:
            desired_actions = self._critic_actions

        # MSE
        batch_size = self._critic_states.shape[0]
        distance = abs(desired_actions - prog_actions)
        #fitness = -(distance.sum())
        fitness = -(distance.sum() / batch_size)

        return fitness

    def fitness_function(self, _, solution, solution_index) -> float:

        env = copy(self.env)

        fitness = 0.0

        prog = self.population.realize(solution_index)

        obs, _ = env.reset()

        terminated, truncated = False, False

        while not terminated or not truncated:

            action = prog(obs)
            next_obs, reward, terminated, truncated, info = env.step([action])

            fitness += reward
            obs = next_obs
            #self.interactions += 1

            if terminated or truncated:
                break

        return fitness

    def fitness_function1(self, _, solution, solution_index) -> float:
        # Update population and realize at index
        prog = self.population.realize(solution_index)

        # Compute program actions
        prog_actions = np.array([prog(state) for state in self._critic_states]).reshape((-1,1))#.astype(np.float32)

        if self.action_space is not None:
            prog_actions = prog_actions.clip(self.action_space.low, self.action_space.high)

        # Change large numbers to arbitrary large number
        #prog_actions[prog_actions==torch.inf] = 9e6

        #print(prog_actions)

        #if self.buffer is not None:
        #    desired_actions, deltas = self.critic.improve_actions(prog_actions.astype(np.float32), self._critic_states)
        #    #desired_actions = desired_actions.astype(np.float32)
        #else:
        #    desired_actions = self._critic_actions

        actions = torch.from_numpy(prog_actions).float()
        states = torch.from_numpy(self._critic_states).float()
        q_values = self.critic.model(actions, states)
        self.critic.model.zero_grad()
        #q_values = np.array([prog(state) for state in self._critic_states]).reshape((-1,1))

        # MSE
        batch_size = self._critic_states.shape[0]
        #distance = sum(q_values)/batch_size
        #fitness = -(distance.sum())
        #fitness = -(distance.sum() / batch_size)
        fitness = float(q_values.mean().detach().numpy())
        #print(fitness)
        #print(np.isnan(fitness))
        #if np.isnan(fitness):
        #    fitness = -99999

        #if fitness > 1000:
        #    fitness = -99999


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
    def fit(self, critic_states=None, critic_actions=None) -> (float, float, float):
        # Reset initial population inside optimizer
        self._optim.initial_population = self.population.individuals
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # Iterate with optimizer
        self.reset_solutions()   #!!!!!
        #self._optim = self._init_optimizer()
        self.new_sample(self._optim)
        self._optim.run()

        self.population.individuals = self._optim.population

        # Set best results
        best_sol, best_fit, best_idx = self._optim.best_solution()
        self.best_solution_index = best_idx
        self.best_fitness = best_fit

        self.best_program = self.population.realize(self.best_solution_index)

        print(f'Optimizer says: best program is {self.best_program}')
        print(f'Optimizer says: best fitness is {self.best_fitness}')

        return (self._optim.last_generation_fitness.max(),
                self._optim.last_generation_fitness.min(),
                self._optim.last_generation_fitness.mean())


if __name__ == "__main__":
    # Test for PyGAD optimizer
    config = OptimizerConfig()
    space = test.SMALL_OBS_SPACE
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS, space)
    print(optim)
