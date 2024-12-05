import test
from config import OptimizerConfig
from population import generate_cartesian_genome_space, CartesianPopulation
from program.operators import SIMPLE_OPERATORS
from program.realization import CartesianProgram


def test_random_program_from_population():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = OptimizerConfig()
    c.program.n_nodes = 2
    c.n_individuals = 100
    gs = generate_cartesian_genome_space(c.program, input_size)

    i = test.SMALl_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS, space)

    # Realize programs
    for idx in range(c.n_individuals):
        genome = pop.individuals[idx]
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
