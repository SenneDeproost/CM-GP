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
from pygad.utils.nsga2 import NSGA2
import numpy as np
from rich.box import SIMPLE
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributed.rpc import new_method

import test
from population import CartesianPopulation
from critic import Critic
from program import Operator, SIMPLE_OPERATORS
from config import *
import gymnasium as gym

from program.realization import Program, CartesianProgram
from program.realization import realize_from_array, realize_subs_from_array
from src.cmgp import critic


def print_fitness(ga, fitnesses):
    #print(f'F_best: {fitnesses.max()}, F_worst: {fitnesses.min()}  F_mean: {fitnesses.mean()}', file=sys.stderr)
    pass



# Todo: Generic class, not fixed to CGP
class PyGADOptimizer:
    def __init__(self,
                 config: OptimizerConfig,
                 operators_dict: dict[int, List[Operator]],
                 observation_space: gym.Space,
                 buffer: ReplayBuffer = None,
                 critic: Critic = None,
                 buffer_batch_size: int = None,
                 action_space: gym.spaces.Box = None,
                 env=None) -> None:
        self.interactions = 0
        self.config = config
        self.observation_space = observation_space
        self.env = env
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

        self.best_index = 0
        self.best_fitness = -np.inf
        self.best_solution = self.population[self.best_index]
        self.best_program = self.population.realize(self.best_index)
        self.best_programs = []

        self.action_space = action_space

        self._critic_states, self._critic_actions = None, None

        # Optimizer instance
        self._init_optimizer()

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
        population = CartesianPopulation(c, self.operators_dict, self.observation_space)
        return population

    # on_generation function to sample new experiences from buffer
    def new_sample(self) -> None:
        # Check if buffer is given in the optimizer
        if self.buffer is not None:
            self._critic_states = self.buffer.sample(self.buffer_batch_size).observations.detach().numpy().astype(
                np.float32)
            #t = self.buffer.sample(10)
            #self._critic_states = np.array([x for xs in t for x in xs]).astype(np.float32)


    def on_generation(self, ga) -> None:
        fit = ga.last_generation_fitness
        print(f'F_best: {fit.max()}, F_worst: {fit.min()}  F_mean: {fit.mean()}', file=sys.stderr)

    def on_mutation(self, ga, offspring):

        for o in offspring:
            prog = realize_from_array(
                genes=o,
                genome_space=self.population.genome_space,
                config=self.config.program,
                state_space=self.population.state_space,
                operators=self.operators_dict
            )
            #print(f'Mutated: {prog}')

    # Initialize PyGAD optimizer Re-init is not expensive.
    def _init_optimizer(self) -> None:
        c = self.config
        from utils.genetic import lexicase_selection
        instance = pygad.GA(
            # General
            suppress_warnings=True,
            fitness_func=self.fitness_function_q,
            initial_population=self.population.individuals,
            num_generations=c.n_generations,
            keep_elitism=c.elitism,
            gene_space=self.range,
            #save_solutions=False,
            #save_best_solutions=True,
            on_generation=self.on_generation,
            on_mutation=self.on_mutation,
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
            parent_selection_type=c.parent_selection,
            keep_parents=c.keep_parents,
        )

        self._optim = instance
        self._optim.__optim = self  # Todo: fix dirty code

    def fitness_individual_sample(self, solution, sample):

        prog = realize_from_array(
            genes=solution,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        res = 0

        prog_action = np.array(prog(sample)).reshape(-1, 1)
        prog_action = prog_action.clip(self.action_space.low, self.action_space.high)
        desired_action, deltas = self.critic.improve_actions(prog_action.astype(np.float32), [sample])
        res += abs(desired_action - prog_action)

        return -res[0][0]

    def fitness_function_gradients(self, _, solution, solution_index) -> float:

        # Compute improved actions
        #prog = self.population.realize(solution_index)

        prog = realize_from_array(
            genes=solution,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        #print(f'Program: {prog}')

        prog_actions = np.array([prog(state) for state in self._critic_states]).reshape((-1, 1))  #.astype(np.float32)
        prog_actions = prog_actions.clip(self.action_space.low, self.action_space.high)
        desired_actions, deltas = self.critic.improve_actions(prog_actions.astype(np.float32), self._critic_states)

        # Printing
        #print(f'Prog action: {prog_actions[0]}')
        #print(f'Desired action: {desired_actions[0]}')

        # MSE
        batch_size = self._critic_states.shape[0]
        distance = abs(desired_actions - prog_actions)
        #fitness = -(distance.sum() / batch_size)
        fitness = -sum(distance)

        return fitness

    def fitness_function_gradients_subs(self, _, solution, solution_index) -> float:

        # Compute improved actions
        #prog = self.population.realize(solution_index)

        progs = realize_subs_from_array(
            genes=solution,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        subs = []
        n_nodes = np.array([len(prog._realization['expressed']) for prog in progs])

        # Calculate distances of all sub programs
        for prog in progs:
            prog_actions = np.array([prog(state) for state in self._critic_states]).reshape(
                (-1, 1))  #.astype(np.float32)
            prog_actions = prog_actions.clip(self.action_space.low, self.action_space.high)
            desired_actions, deltas = self.critic.improve_actions(prog_actions.astype(np.float32), self._critic_states)

            # MSE
            batch_size = self._critic_states.shape[0]
            distance = abs(desired_actions - prog_actions)
            fitness = (distance.sum() / batch_size)
            subs.append(fitness)

        subs = np.array(subs)
        #avg_subs = subs.mean()
        avg_subs = np.average(subs, weights=n_nodes)

        prog_fit = self.fitness_function_gradients(_, solution, solution_index)

        return prog_fit / avg_subs

    def fitness_function_direct(self, _, solution, solution_index) -> float:

        env = copy(self.env)

        fitness = 0.0

        #prog_wrong = self.population.realize(solution_index) # WRONG
        prog = realize_from_array(
            genes=solution,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        #print()
        #print(f'Wrong: {prog_wrong}')
        #print(f'Correct: {prog}')

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

    def fitness_function_q(self, _, solution, solution_index) -> float:

        #prog = self.population.realize(solution_index)
        prog = realize_from_array(
            genes=solution,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        # Compute program actions
        prog_actions = np.array([prog(state) for state in self._critic_states]).reshape((-1, 1))  #.astype(np.float32)


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

    # Fit the produced actions to more optimal ones
    def fit_old(self, critic_states=None, critic_actions=None) -> (float, float, float):

        # Iterate with optimizer
        self._optim = self._init_optimizer()
        self._optim.initial_population = self.population.individuals
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # Sample from buffer if given
        if self.buffer is not None:
            self.new_sample()

        # Calculate initial fitness
        self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

        # Run for each generation the whole evolutionary loop.

        for gen in range(self.config.n_generations - 1):
            # If replay buffer is given, sample new experience in each generation
            if self.buffer is not None:
                self.new_sample()

            # Reinit test
            self._init_optimizer()
            self._optim.initial_population = self.population.individuals
            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # The genetic operations on the population
            self._optim.run_select_parents()
            self._optim.run_crossover()
            self._optim.run_mutation()
            self._optim.run_update_population()

            # Calc fitness function
            self._optim.previous_generation_fitness = self._optim.last_generation_fitness.copy()
            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # Print
            self.on_generation(self._optim)

            # Update population with population of optimizer
            self.population.individuals = self._optim.population

        # Best solution
        self.best_solution, self.best_fitness, self.best_index = self._optim.best_solution(
            pop_fitness=self._optim.last_generation_fitness)
        self.best_program = self.population.realize(self.best_index)

        # Print
        print(f'Best program is {self.best_program} with fitness {self.best_fitness}')

        return (self._optim.last_generation_fitness.max(),
                self._optim.last_generation_fitness.min(),
                self._optim.last_generation_fitness.mean())

        # Fit the produced actions to more optimal ones

    # Run n_generations with the optimizer
    def run_optimizer(self):
        # Run for each generation the whole evolutionary loop.
        for gen in range(self.config.n_generations):
            # If replay buffer is given, sample new experience in each generation
            if self.buffer is not None:
                pass
                #self.new_sample()

            # Reinit test
            self._init_optimizer()
            self._optim.initial_population = self.population.individuals
            #self._optim.run()

            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # The genetic operations on the population
            self._optim.run_select_parents()
            self._optim.run_crossover()
            self._optim.run_mutation()
            self._optim.run_update_population()

            # Calc fitness function
            self._optim.previous_generation_fitness = self._optim.last_generation_fitness.copy()
            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # Print
            self.on_generation(self._optim)

            # Update population with population of optimizer
            self.population.individuals = copy(self._optim.population)

    def run_direct_validation(self, genes):

        env = copy(self.env)
        fitness = 0.0

        #prog_wrong = self.population.realize(solution_index) # WRONG
        prog = realize_from_array(
            genes=genes,
            genome_space=self.population.genome_space,
            config=self.config.program,
            state_space=self.population.state_space,
            operators=self.operators_dict
        )

        obs, _ = env.reset()

        terminated, truncated = False, False

        while not terminated or not truncated:

            action = prog(obs)
            next_obs, reward, terminated, truncated, info = env.step([action])

            fitness += reward
            obs = next_obs
            self.interactions += 1

            if terminated or truncated:
                break

        return fitness

    def fit(self, critic_states=None, critic_actions=None) -> (float, float, float):

        # Iterate with optimizer
        self._init_optimizer()
        self._optim.initial_population = self.population.individuals
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # Sample from buffer if given
        if self.buffer is not None:
            self.new_sample()

        # Calculate initial fitness
        self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

        # Run the optimizer
        self.run_optimizer()

        # Candidate is best from optimizer
        self.best_solution, self.best_fitness, self.best_index = self._optim.best_solution(
            pop_fitness=self._optim.last_generation_fitness)

        self.best_program = self.population.realize(self.best_index)

        # Test by replacing population with best program !!!!!
        #self.population.individuals = np.tile(self.best_solution, np.array([self.config.n_individuals, 1]))

        #candidate_score = self.run_direct_validation(candidate_solution)
        #best_program_score = self.run_direct_validation(self.best_solution)

        # Test if candidate performs better than current best in direct interaction

        # Print
        print(f'Candidate is {self.best_program} with fitness {self.best_fitness}')
        #print(f'Best program is {self.best_program} with score {best_program_score}')
        #print(f'Best candidate is {candidate_program} with score {candidate_score}')

        #if candidate_score > best_program_score:
        #    print(f"New best program: {self.best_program}")


        return (self._optim.last_generation_fitness.max(),
                self._optim.last_generation_fitness.min(),
                self._optim.last_generation_fitness.mean())


if __name__ == "__main__":
    # Test for PyGAD optimizer
    config = OptimizerConfig()
    space = test.SMALL_OBS_SPACE
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS, space)
    print(optim)
