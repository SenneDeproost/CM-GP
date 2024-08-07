import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS

class ProgramOptimizer:
    def __init__(self, config, state_dim):

        # Create the initial population
        # We create it so these random programs try all the operators and read all the state variables
        self.initial_population = np.random.random((config.num_individuals, config.num_genes * 1))  # Random numbers between 0 and 1
        self.initial_population[0:-1:1] *= -(NUM_OPERATORS + state_dim)         # Tokens between -NUM_OPERATORS - state_dim and 0
        #self.initial_population[1:-1:1] *= 3.0                                  # Log_std between 0 and 3
        #self.initial_population = np.random.random_integers(
        #    size=(config.num_individuals, config.num_genes),
        #    low=-(NUM_OPERATORS + state_dim),
        #    high=-(NUM_OPERATORS + 1)
        #)

        self.best_solution = self.initial_population[0]
        self.best_fitness = -np.inf
        self.pop_fitness = -np.inf

        self.config = config
        self.state_dim = state_dim

    def increase_individual_size(self):
        s = self.initial_population.shape
        s = (s[0], s[1] + 1)
        self.initial_population = np.resize(self.initial_population, s)
        #self.initial_population = np.pad(self.initial_population, (0, 1), mode='constant', constant_values=(1))
        #self.initial_population = np.pad(self.initial_population, (0, 1), mode='mean')

    def get_action(self, state):
        program = Program(genome=self.best_solution)
        return program(state)

    def get_best_solution_str(self):
        program = Program(genome=self.best_solution)
        return program.to_string([0.0] * self.state_dim)

    def _fitness_func(self, ga_instance, solution, solution_idx):
        batch_size = self.states.shape[0]
        sum_error = 0.0
        sum_lookedat = 0.0

        # Evaluate the program several times, because evaluations are stochastic
        #for eval_run in range(self.config.num_eval_runs):
        for index in range(batch_size):
                # Create the Program here to sample the tokens for every eval run and every index
            program = Program(genome=solution)

            if program.to_string([0]) == 'False':
                sum_error += np.inf

                # MSE for the loss
            action = program(self.states[index])
            desired_action = self.actions[index]

            sum_error += np.mean((action - desired_action) ** 2)

            # Num input variables looked at
            sum_lookedat += program.num_inputs_looked_at(self.states[index])

        avg_error = (sum_error / (batch_size * self.config.num_eval_runs))
        avg_lookedat = (sum_lookedat / (batch_size * self.config.num_eval_runs))
        #print(sum_lookedat)

        fitness = -avg_error #/ avg_lookedat**2 # / (sum_lookedat + 0.01) # FIXME: random equation

        return fitness

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N,), we assume continuous actions

            NOTE: One ProgramOptimizer has to be used for each action dimension
        """
        self.states = states        # picklable self._fitness_func needs these instance variables
        self.actions = actions

        self.ga_instance = pygad.GA(
            fitness_func=self._fitness_func,
            initial_population=self.initial_population,
            num_generations=self.config.num_generations,
            num_parents_mating=self.config.num_parents_mating,
            keep_parents=self.config.keep_parents,

            # Work with non-deterministic objective functions
            keep_elitism=self.config.keep_elitism,
            save_solutions=False,
            save_best_solutions=False,

            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            mutation_probability=self.config.mutation_probability,
            #mutation_percent_genes=self.config.mutation_percent_genes,
            #mutation_probability=[0.8, 0.1],
            #mutation_num_genes=[2, 2],
            #mutation_percent_genes=[self.config.mutation_percent_genes]*2,
            random_mutation_max_val=2,
            random_mutation_min_val=-2,
            gene_space={
                'low': -(NUM_OPERATORS + self.state_dim),
                'high': 10
            }
            #parallel_processing=["process", None]
        )
        self.ga_instance.run()

        # Allow the population to survive
        self.initial_population = self.ga_instance.population
        #self.pop_fitness = self.ga_instance.cal_pop_fitness()
        #self.pop_fitness = np.mean(self.pop_fitness[np.isfinite(self.pop_fitness)])

        # Best solution for now
        best_sol, best_fit, idx = self.ga_instance.best_solution()
        fit = self.ga_instance.last_generation_fitness
        mean_fit = np.mean(fit[np.isfinite(fit)])
        print(f'Population fitness: {fit}')
        print(f'Mean population fitness: {mean_fit}')
        self.pop_fitness = mean_fit
        #if fit > self.best_fitness:
        self.best_solution = best_sol
        self.best_fitness = best_fit

