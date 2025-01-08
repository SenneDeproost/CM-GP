import sys

#sys.path.insert(0, "../")
from typing import List

import gym
import numpy as np
from optimizer import PyGADOptimizer
from config import OptimizerConfig
from program.operators import Operator
from population import generate_cartesian_genome_space, Genome
from program.realization import CartesianProgram
from program import SIMPLE_OPERATORS, SIMPLE_OPERATORS_DICT
import test
import pytest


class PyGADOptimizer(PyGADOptimizer):

    def __init__(self, config: OptimizerConfig,
                 operators_dict: dict[int, List[Operator]],
                 state_space: gym.Space) -> None:

        super().__init__(config, operators_dict, state_space)

        gs = generate_cartesian_genome_space(self.config.program, 2, SIMPLE_OPERATORS_DICT)
        genome = Genome(genes=test.BIG_GENE_2_OUTPUT, genome_space=gs)
        self.ground = CartesianProgram(genome, state_space, SIMPLE_OPERATORS, self.config.program)
        print('done')


    def fitness_function(self, _, solution, solution_index) -> float:
        fitness = 0.0

        prog = self.population.realize(solution_index)

        input_size = self.state_space.shape[0]

        i = test.SMALL_INPUT

        batch_size = self._critic_states.shape[0]

        #for i in range(batch_size):
       #     # Calculate MSE between proposed and improved action
       #     action = prog(self._critic_states[i])
       #     desired_action = self._critic_actions[i]

       #    fitness += (action - desired_action) ** 2

        action = prog(i)
        desired_action = self.ground(i)
        fitness += (action - desired_action) ** 2

        # Avg
        fitness = -(fitness / batch_size)

        return fitness

    def fit(self):

        self._critic_states = np.random.random((1000, 2))

        self._optim.initial_population = self.raw_population
        self._optim.run()

        # Update population
        self.raw_population = self._optim.population
        self.population.update(self._optim.population)

        # Reset best program
        self.best_program = self.population.realize(self.best_solution_index)



def test_optimization():
    from program import SIMPLE_OPERATORS_DICT
    import test

    config = OptimizerConfig()
    config.n_individuals = 10
    config.n_parents_mating = 9
    config.program.n_outputs = 2
    config.elitism = 0
    config.program.n_nodes = 6
    config.program.n_inputs = 2
    config.n_generations = 20


    space = gym.spaces.Box(low=-20, high=20, shape=(2,))
    input = test.SMALL_INPUT
    optim = PyGADOptimizer(config, SIMPLE_OPERATORS_DICT, space)

    print(f'Input: {input}')
    print(f'Ground program {optim.ground}:')

    for i in range(5):
        optim.fit()
        #print(optim.raw_population)
        print(optim.best_program)
        print()

    print(f'Input: {input}')
    print(f'Ground program {optim.ground}:')
    print(optim.ground(input))
    print(f'Best found program {optim.best_program}:')
    print(optim.best_program(input))













