import sys

from torch import candidate

sys.path.append('./src/cmgp/')

import test
from config import CartesianConfig, OptimizerConfig
from population import Genome, generate_cartesian_genome_space, genes_per_node, CartesianPopulation
from program import SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS
from program.realization import CartesianProgram
import gymnasium as gym
import numpy as np

import pygad
import test


def print_fitness(ga, fitnesses):
    print(f'F_best: {fitnesses.max()}, F_worst: {fitnesses.min()}  F_mean: {fitnesses.mean()}', file=sys.stderr)


def on_generation(ga):
    print(f'Generation {ga.generations_completed}')
    sol = ga.best_solutions[-1]
    fit = ga.best_solutions_fitness[-1]
    genome = Genome(genes=sol, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, config.program)
    print(f'Best: {prog}, with fit: {fit}')


config = OptimizerConfig()

config.program.n_nodes = 6
config.program.n_inputs = 2
config.n_individuals = 100

space = test.TINY_OBS_SPACE
input = np.random.random((1000, 2))


gs = generate_cartesian_genome_space(config.program, 2, SIMPLE_OPERATORS_DICT)
population = CartesianPopulation(config, SIMPLE_OPERATORS_DICT, space)

genome = Genome(genes=test.BIG_GENE_1_OUTPUT, genome_space=gs)
prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, config.program)

#input = test.TINY_INPUT
batch_size = input.shape[0]


def fitness_function(ga, solution, solution_index) -> float:
    fitness = 0.0

    genome = Genome(genes=solution, genome_space=gs) # Important not to realize via population abstraction
    candidate = CartesianProgram(genome, space, SIMPLE_OPERATORS, config.program)
    # Update the population

    for i in range(batch_size):
        # Calculate MSE between proposed and improved action

        ground = prog(input[i])
        pred = candidate(input[i])

        distance = (ground - pred) ** 2

        fitness += distance

    # Avg
    #fitness = -(fitness / batch_size)
    fitness = -fitness / batch_size

    return fitness


ga = pygad.GA(
    # General
    suppress_warnings=True,
    fitness_func=fitness_function,
    initial_population=population.individuals,
    num_generations=1000,
    keep_elitism=5,
    gene_space=population.range_description(),
    #gene_space={'low': 0, 'high': 10},
    save_solutions=False,
    save_best_solutions=True,
    on_fitness=print_fitness,
    on_generation=on_generation,
    parallel_processing=None,  # Utilize all available resources
    # Mutation
    mutation_probability=0.1,
    #mutation_percent_genes=c.gene_mutation_percent,
    mutation_type='random',
    random_mutation_min_val=-20,
    random_mutation_max_val=20,
    # Crossover
    num_parents_mating=90,
    crossover_type='single_point',
    parent_selection_type='sss'
)

ga.run()

# Update population
population.individuals = ga.population
print('done')
