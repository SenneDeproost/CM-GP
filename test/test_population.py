import sys
sys.path.append('./src/cmgp/')

import test
from config import OptimizerConfig
from population import CartesianPopulation
from program import SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS, SIMPLE_FUNCTIONS, SIMPLE_FUNCTIONS_DICT
from program.realization import CartesianProgram

def test_random_program_from_population():

    # Tiny
    space = test.TINY_OBS_SPACE

    c = OptimizerConfig()
    c.program.n_nodes = 3 #c.program.max_node_arity  # Minimal amount of nodes for operator with highest n_operands
    c.program.n_outputs = 1
    c.n_individuals = 100

    i = test.TINY_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS_DICT, space)

    # Realize programs
    for idx in range(c.n_individuals):
        print(f' \n Individual {idx}')
        genome = pop.get_genome(idx)
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
        print(prog.evaluate(i))
        [print(sub) for sub in pop.realize_subs(idx)]


    # Small
    space = test.SMALL_OBS_SPACE

    c = OptimizerConfig()
    c.program.n_nodes = 10#c.program.max_node_arity  # Minimal amount of nodes for operator with highest n_operands
    c.program.n_outputs = 1
    c.n_individuals = 100

    i = test.SMALL_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS_DICT, space)

    # Realize programs
    for idx in range(c.n_individuals):
        print(f' \n Individual {idx}')
        genome = pop.get_genome(idx)
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
        print(prog.evaluate(i))
        [print(sub) for sub in pop.realize_subs(idx)]

    # Small with 2 outputs
    space = test.SMALL_OBS_SPACE

    c = OptimizerConfig()
    c.program.n_nodes = 10#c.program.max_node_arity  # Minimal amount of nodes for operator with highest n_operands
    c.program.n_outputs = 2
    c.n_individuals = 100

    i = test.SMALL_INPUT
    pop = CartesianPopulation(c, SIMPLE_OPERATORS_DICT, space)

    # Realize programs
    for idx in range(c.n_individuals):
        print(f' \n Individual {idx}')
        genome = pop.get_genome(idx)
        prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
        print(prog)
        print(prog.evaluate(i))
        #[print(sub) for sub in pop.realize_subs(idx)]
