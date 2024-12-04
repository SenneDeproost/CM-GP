import gymnasium as gym
import test
from config import CartesianConfig
from population import OperatorGeneSpace, CartesianGeneSpace, Genome, generate_cartesian_genome_space, genes_per_node
from program.operators import SIMPLE_OPERATORS
from program.realization import CartesianProgram
from test import SMALl_INPUT


def test_small_program_1_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.SMALL_GENE.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: id(input_0) + 5.0
    # Output: -4.0
    i = test.SMALl_INPUT
    genome = Genome(genes=test.SMALL_GENE, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(prog._realization, test.SMALl_INPUT) == -4.0
    assert prog.to_string() == 'id(input_0) + 5.0'
    assert prog.to_string(test.SMALl_INPUT) == f'id({SMALl_INPUT[0]}) + 5.0'


def test_big_program_1_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.BIG_GENE.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: (id(input_0) + 5.0) * input_1
    # Output: -396.0
    i = test.SMALl_INPUT
    genome = Genome(genes=test.BIG_GENE, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(prog._realization, i) == -4.0
    assert prog.to_string() == '(id(input_0) + 5.0) * input_1'
    assert prog.to_string(i) == f'(id({[0]}) + 5.0) * i{[1]}'
