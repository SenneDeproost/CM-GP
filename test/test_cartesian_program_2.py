import sys

sys.path.insert(0, "../")

import numpy as np

import test
from config import CartesianConfig
from population import Genome, generate_cartesian_genome_space, genes_per_node
from program import SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS
from program.realization import CartesianProgram


def test_exp_cartesian_program():
    genome = np.array([-14., - 1., - 1., - 1., - 1.,
                       -4.65032239, 0., 0., 0., 0.,
                       -2.93042557, 0.14872832, 0.88220684, 0.37313618, 0.4734725,
                       1.92556271])

    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = 3
    c.n_outputs = 1
    gs = generate_cartesian_genome_space(c, 1, SIMPLE_OPERATORS_DICT)

    # Program: exp(input_0)
    # Output: 0.00012340980408667956
    i = test.SMALL_INPUT
    genome = Genome(genes=genome, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)
    assert prog.to_string() == 'exp(input_0)'
    assert prog.to_string(i) == f'exp({i[0]})'
    assert prog.evaluate(i) == 0.00012340980408667956


def test_add_cartesian_program():
    genome = np.array([-14., -1., -1., -1., -1.,
                       7.04613667, 0., 0., 0., 0.,
                       -6.9158864, 0.68848519, 0.13873804, 0.69769573, 0.22276585,
                       1.93023767])

    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = 3
    c.n_outputs = 1
    gs = generate_cartesian_genome_space(c, 1, SIMPLE_OPERATORS_DICT)

    # Program:
    # Output:
    i = test.SMALL_INPUT
    genome = Genome(genes=genome, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)
    print('done')
