import sys

from tensorflow.python.saved_model.signature_serialization import find_function_to_export

sys.path.append('../src/cmgp/')
sys.path.append('../')

from program import SIMPLE_FUNCTIONS_DICT
from critic import Critic

from config import CartesianConfig, OptimizerConfig, CriticConfig
from population import Genome, generate_cartesian_genome_space, genes_per_node, CartesianPopulation
from program import SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS, SIMPLE_FUNCTIONS, SIMPLE_FUNCTIONS_DICT
from program.realization import CartesianProgram
from math import isnan
import gymnasium as gym
import numpy as np
import torch
import pygad
import envs
from copy import copy

ENV = gym.make('InvertedPendulum-v4')
N_OBS = 10000

amount = np.array([1.1, 0.2, 1.5, 3.14])
obs_min, obs_max = -amount, amount
obs_size = ENV.observation_space.shape[0]
OBS = np.random.uniform(low=obs_min, high=obs_max, size=(N_OBS, obs_size))



def print_fitness(ga, fitnesses):
    print(f'F_best: {fitnesses.max()}, F_worst: {fitnesses.min()}  F_mean: {fitnesses.mean()}', file=sys.stderr)


def on_generation(ga):
    print(f'Generation {ga.generations_completed}')
    population.individuals = ga.population
    sol = ga.best_solutions[-1]
    fit = ga.best_solutions_fitness[-1]
    genome = Genome(genes=sol, genome_space=gs)
    prog = CartesianProgram(genome, ENV.observation_space, SIMPLE_FUNCTIONS, config.program)

    #all = population.realize_all()
    #[print(str(a)) for a in all]

    print(f'Best: {prog}, with fit: {fit}')

def fitness_function(ga, solution, solution_index) -> float:

    env = ENV

    genome = Genome(genes=solution, genome_space=gs) # Important not to realize via population abstraction
    candidate = CartesianProgram(genome, ENV.observation_space, SIMPLE_FUNCTIONS, config.program)

    global OBS
    pred_actions = []
    for obs in OBS:
        pred_action = candidate(obs)
        # Check if action is outside of space range
        #if not (env.action_space.low <= pred_action <= env.action_space.high):
        #    fitness = -9999
        #    return fitness

        pred_action = np.clip(pred_action, env.action_space.low, env.action_space.high)
        pred_actions.append(pred_action)

    obs, pred_actions = torch.tensor(OBS, dtype=torch.float).float(), torch.tensor(pred_actions, dtype=torch.float).reshape(-1, 1)
    with torch.no_grad():
        q = critic.model(obs, pred_actions).view(-1)
    fitness = sum(q.detach()).item()
    #print(fitness)

    if isnan(fitness):
        print('nan in critic')
        fitness = -9999999


    #print()
    #print(fitness)
    #print(type(fitness))
    #print()

    return fitness

config = OptimizerConfig()

config.program.n_nodes = 50 #50
config.program.n_inputs = ENV.observation_space.shape[0]
config.program.max_constant = 0
config.program.n_outputs = 1
config.program.max_node_arity = 2


config.n_individuals = 100 # 1000
config.n_parents_mating = 90
config.elitism = 0
config.n_generations = 1000

delta_gene_values = config.program.max_constant+len(SIMPLE_FUNCTIONS)+config.program.n_inputs
delta_gene_values = delta_gene_values
config.mutation_val = (-delta_gene_values, delta_gene_values)
print(f'Mutation val: {config.mutation_val}')
config.gene_mutation_probability = 0.2 # 0.1

population = CartesianPopulation(config=config,
                                 operators_dict=SIMPLE_FUNCTIONS_DICT,
                                 state_space=ENV.observation_space)

gs = generate_cartesian_genome_space(config=config.program,
                                     n_inputs=config.program.n_inputs,
                                     operators_dict=SIMPLE_FUNCTIONS_DICT)

critic_config = CriticConfig()
critic = Critic(ENV, critic_config)
critic.model.load_state_dict(torch.load('./trained_critic_InvertedPendulum-v4.pth'))



ga = pygad.GA(
    # General
    suppress_warnings=True,
    fitness_func=fitness_function,
    initial_population=population.individuals,
    num_generations=config.n_generations,
    keep_elitism=config.elitism,
    gene_space=population.range_description(),
    save_solutions=False,
    save_best_solutions=True,
    on_fitness=print_fitness,
    on_generation=on_generation,
    parallel_processing=1,  # Utilize all available resources
    # Mutation
    mutation_probability=config.gene_mutation_probability,
    #mutation_percent_genes=c.gene_mutation_percent,
    mutation_type=config.mutation,
    random_mutation_min_val=config.mutation_val[0],
    random_mutation_max_val=config.mutation_val[1],
    # Crossover
    num_parents_mating=config.n_parents_mating,
    crossover_type=config.crossover,
    parent_selection_type=config.parent_selection,
)

ga.run()

# Update population
population.individuals = ga.population
print('done')



