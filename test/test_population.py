from exceptiongroup import catch

import test
from config import OptimizerConfig
from population import generate_cartesian_genome_space, CartesianPopulation
from program import SIMPLE_OPERATORS
from program.realization import CartesianProgram


def test_random_program_from_population():
    space = test.SMALL_OBS_SPACE

    c = OptimizerConfig()
    c.program.n_nodes = 10#c.program.max_node_arity  # Minimal amount of nodes for operator with highest n_operands
    c.n_individuals = 10000

    i = test.SMALl_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS, space)

    # Realize programs
    for idx in range(c.n_individuals):
        print(f' \n Individual {idx}')
        genome = pop.individuals[idx]
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
        print(prog.evaluate(i))
