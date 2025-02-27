# +++ CM-GP/utils/genetic +++
#
# Custom genetic operations for genetic loop
#
# 20/02/2025 - Senne Deproost

from pygad import GA
from pygad.utils.parent_selection import ParentSelection
import numpy as np


###
# SELECTION
###

def lexicase_selection(population_fitness: np.ndarray[float], num_parents: int, ga: GA):
    fitness_sorted = sorted(range(len(population_fitness)), key=lambda k: population_fitness[k])
    # Reverse the sorted solutions so that the best solution comes first.
    fitness_sorted.reverse()

    # Access the optimizer in the ga instance
    optim = ga._PyGADOptimizer__optim

    if ga.gene_type_single:
        parents = np.empty((num_parents, ga.population.shape[1]), dtype=ga.gene_type[0])
    else:
        parents = np.empty((num_parents, ga.population.shape[1]), dtype=object)

    # Go over each given example and calculate the iteratively set of candidates.

    for state in optim._critic_states:
        res = []
        for individual in ga.population:
            fit = optim.fitness_individual_sample(individual, state)
            res.append(fit)
        res = np.array(res)
        print('Don')


    return parents, fitness_sorted[:num_parents]
