import test
from config import CartesianConfig
from population import Genome, generate_cartesian_genome_space, genes_per_node
from program.operators import SIMPLE_OPERATORS
from program.realization import CartesianProgram


def test_small_program_1_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.SMALL_GENE_1_OUTPUT.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: id(input_0) + 5.0
    # Output: -4.0
    i = test.SMALl_INPUT
    genome = Genome(genes=test.SMALL_GENE_1_OUTPUT, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(i) == -4.0
    assert prog.to_string() == 'id(input_0) + 5.0'
    assert prog.to_string(i) == f'id({i[0]}) + 5.0'

def test_small_program_2_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.SMALL_GENE_2_OUTPUT.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: ∑[ (id(input_0) + 5.0) , id(input_0) ]
    # Output: -13.0
    i = test.SMALl_INPUT
    genome = Genome(genes=test.SMALL_GENE_2_OUTPUT, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(i) == -13.0
    assert prog.to_string() == '∑[ id(input_0), id(input_0) + 5.0 ]'
    assert prog.to_string(i) == f'∑[ id({i[0]}), id({i[0]}) + 5.0 ]'

def test_big_program_1_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.BIG_GENE_1_OUTPUT.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: (id(input_0) + 5.0) * input_1
    # Output: -396.0
    i = test.SMALl_INPUT
    genome = Genome(genes=test.BIG_GENE_1_OUTPUT, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(i) == -396.0
    assert prog.to_string() == '(id(input_0) + 5.0) * input_1'
    assert prog.to_string(i) == f'(id({i[0]}) + 5.0) * {i[1]}'

def test_big_program_2_output():
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    c = CartesianConfig()
    c.n_nodes = int(test.BIG_GENE_2_OUTPUT.shape[0] / genes_per_node(c))
    gs = generate_cartesian_genome_space(c, input_size)

    # Program: ∑[ id(input_0) + 5.0, (id(input_0) + 5.0) * input_1 ]
    # Output: -400
    i = test.SMALl_INPUT
    genome = Genome(genes=test.BIG_GENE_2_OUTPUT, genome_space=gs)
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)

    assert prog.evaluate(i) == -400.0
    assert prog.to_string() == '∑[ id(input_0) + 5.0, (id(input_0) + 5.0) * input_1 ]'
    assert prog.to_string(i) == f'∑[ id({i[0]}) + 5.0, (id({i[0]}) + 5.0) * {i[1]} ]'