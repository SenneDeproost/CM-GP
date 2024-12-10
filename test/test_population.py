from exceptiongroup import catch

import test
from config import OptimizerConfig
from population import generate_cartesian_genome_space, CartesianPopulation
from program.operators import SIMPLE_OPERATORS
from program.realization import CartesianProgram


def test_random_program_from_population():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = OptimizerConfig()
    c.program.n_nodes = 6#c.program.max_node_arity  # Minimal amount of nodes for operator with highest n_operands
    c.n_individuals = 1000
    gs = generate_cartesian_genome_space(c.program, input_size)

    i = test.SMALl_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS, space)

    # Realize programs
    for idx in range(c.n_individuals):
        genome = pop.individuals[idx]
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
