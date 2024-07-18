import cma
from cma import CMAEvolutionStrategy
import numpy as np
from postfix_program import Program

from dataclasses import dataclass
import pyrallis


class ProgramOptimizer:

    def __init__(self, config, n_input_variables):
        # Create the initial population
        self.initial_program = [-1.0] * config.num_genes

        self.best_solution = self.initial_program
        self.best_fitness = None

        self.n_input_variables = n_input_variables

        self.config = config
        self.ga_instance = CMAEvolutionStrategy(self.initial_program, config.sigma0)

    def get_best_program(self):
        return Program(genome=self.best_solution)

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N, action_shape), we assume continuous actions
        """

        def fitness_function(soloution):
            batch_size = states.shape[0]
            action_size = actions.shape[1]
            sum_error = 0.0

            program = Program(soloution)

            for index in range(batch_size):
                action = program(states[index], len_output=action_size)
                desired_action = actions[index]

                sum_error += np.mean((action - desired_action) ** 2)

            fitness = -(sum_error / batch_size)

            return fitness

        self.ga_instance.optimize(fitness_function)

        res_soloution = self.ga_instance.result[0]
        res_fitness = self.ga_instance.result.fbest
        if self.best_solution is None or self.best_fitness:
            self.best_fitness = res_fitness
            self.best_solution = res_soloution


@dataclass
class Config:
    sigma0: float = 1
    "Initial standard deviation of the CMA evolution strategy."
    num_genes: int = 3
    "Number of genes in the program."


@pyrallis.wrap()
def main(config: Config):
    optim = ProgramOptimizer(config, n_input_variables=1)

    states = np.random.random_sample((10, 2))
    actions = np.sum(states, axis=1)
    actions = np.reshape(actions, (10, 1))

    optim.fit(states, actions)
    res = optim.get_best_program()


if __name__ == '__main__':
    main()
