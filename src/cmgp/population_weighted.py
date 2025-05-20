# +++ CM-GP/population +++
#
# Population components for evolution
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

from collections import OrderedDict
from copy import copy
from typing import List, Callable, Union, Counter
import numpy as np
from fontTools.misc.plistlib import end_integer
from sympy.polys.polyoptions import Order
from tensorflow.python.ops.gen_nn_ops import selu_grad

from config import CartesianConfig, OptimizerConfig
import gymnasium as gym

from program import Operator, InputVar, SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS

#from src.cmgp.program.realization import CartesianProgram

EMPTY = -1


# Calculate the number of genes per node
def gene_length_node(config: CartesianConfig, operators: List[Operator]) -> int:
    n_operators = len(operators)
    n = n_operators + config.max_node_arity  # Distribution over operators + information for connections
    return n

# Calculate length of genome
def genome_length(config: CartesianConfig, n_inputs:int,  operators: List[Operator]) -> int:
    function_values = (n_inputs + len(operators)) * config.n_nodes
    connection_values = config.max_node_arity * config.n_nodes**2 # ?? !! Not sure
    root_values = config.n_outputs * config.n_nodes
    return function_values + connection_values + root_values


# Class for gene ranges
class GeneRange:
    def __init__(self,
                 min: np.ndarray,
                 max: np.ndarray,
                 values: List = None) -> None:


        # Check if the gene range is valid
        delta = max - min
        assert np.all(delta >= 0), "Invalid gene range values"
        assert len(min) == len(max), "Invalid gene range length"

        self.min, self.max = min, max
        self.values = values
        self.empty = False

        # Range can be empty
        if range is None and values is None:
            self.empty = True

    #def __getitem__(self, i: int) -> Union[float, int, None]:
    #    if self.empty:
    #        return None
    #    elif self.values is not None:
    #        return self.values[i]
    #    else:
    #        return self.range[i]

    # Dunder for printing
    def __str__(self) -> str:
        return f'Gene range {self.min}->{self.max}'

    # Membership test
    #def contains(self, values: np.ndarray[Union[float, int]]) -> bool:

        # Values
    #    if not (self.values is None):
    #        return value in self.values
    #    return False

    # Translate for optimizer
    def description(self) -> Union[dict, list]:
            return {
                'low': self.min,
                'high': self.max
            }


# Abstract for different types of GeneSpace
class GeneSpace:
    def __init__(self, gene_range: GeneRange) -> None:
        self.gene_range = gene_range

    def __repr__(self) -> str:
        return f'Gene space with range ( {self.gene_range.min} -> {self.gene_range.max} )'

    # Sampling method within the range of the space
    def sample(self) -> Union[int, float]:

        # Check for empty range
        if self.gene_range.empty:
            return EMPTY
        else:
            return np.random.uniform(low=self.gene_range.min, high=self.gene_range.max)

    # Different types of rounding possible
    @staticmethod
    def _round(value: float, mode: str = 'regular') -> int:

        match mode:
            case 'regular':
                return round(value)
            case 'stochastic':
                return int(value + (np.random.random() - 0.5))
            case _:
                ValueError(f'Unknown rounding type: {mode}')
                return None



# Distributional gene space with weights
class DistributionalGeneSpace(GeneSpace):
    def __init__(self, n_values: int) -> None:
        self.n_values = n_values
        self.gene_range = GeneRange(min=np.zeros(self.n_values), max=np.ones(self.n_values))
        super().__init__(self.gene_range)

    def sample(self) -> np.ndarray[float]:
        value = super().sample()
        return value




# Genome of individual program
class Genome:
    def __init__(self,
                 genome_space: List[GeneSpace],
                 #n_genes: Union[int, None] = None,
                 genome_length: Union[int, None] = None,
                 pop_index: int = -1,
                 values: np.ndarray = None) -> None:
        self.genome_space = genome_space
        self.pop_index = pop_index
        self.values = values
        self.genome_length = genome_length

        #assert len(genome_space) == n_genes, "Invalid genome space"

        # Check if genes are given or need to be initialized by the genome
        if self.values is not None:
            self.values = values
        else:
            assert self.genome_length, "No genome length given"
            self.values = np.zeros(self.genome_length)
            self._init_genome()

    # Dunder for genome length
    def __len__(self) -> int:
        return self.genome_length

    # Dunder for genome
    def __str__(self) -> str:
        return str(f'{self.values}')

    # Makes circular list possible
    def _get_gene_space(self, i):
        return self.genome_space[i % len(self.genome_space)]

    # Sample gene values from the respective gene spaces. When gene space has exclusions, take them into account
    def _init_genome(self) -> None:

        # Go over each space in the genome space and sample values
        res = []
        for gs in self.genome_space:
            res.extend(gs.sample())
        self.values = np.array(res)


    # Helper function to return for every set of genes of a certain length the value
    def every_ith_gene(self, n: int, seq_len: int) -> list[float]:
        res = []
        for i in range(int(len(self) / seq_len)):
            res.append(self.genes[i * seq_len + n])
        return res


# Abstract for population of individual genomes
class Population:
    def __init__(self, n_individuals: int, genome_length: int, genome_space: List[GeneSpace]) -> None:
        self.n_individuals = n_individuals
        self.genome_length = genome_length
        self.genome_space = genome_space

        self.individuals = np.zeros((n_individuals, self.genome_length))
        self._init_population()

    def __getitem__(self, i) -> np.ndarray[float]:
        return self.individuals[i]

    # Return Genome abstraction from genes at index in population
    def get_genome(self, index: int) -> Genome:
        values = self.individuals[index]
        genome = Genome(values=values, genome_space=self.genome_space, pop_index=index)
        return genome

    # Initialize population by populating it with genomes
    def _init_population(self) -> None:
        for i in range(self.n_individuals):
            values = []
            for gs in self.genome_space:
                v = gs.sample()
                values.extend(v)
                print(len(v))

            self.individuals[i] = np.array(values)


# Helper function to return all output node genes
#def has_output(genome: Genome, config: CartesianConfig) -> bool:
#    outputs = genome.every_ith_gene(n=1, seq_len=genes_per_node(config))
#    has = 1 in outputs
#    return has


# Resolve genomes with no output node
#def resolve_output(genome: Genome, config: CartesianConfig) -> None:
#    # Make last node an output node
#    i = -(config.max_node_arity + 1)
#    genome.genes[i] = 1

# Generate Cartesian gene space
def generate_cartesian_genome_space(config: CartesianConfig,
                                    n_inputs: int,
                                    operators_dict: dict[int, List[Operator]]) -> List[GeneSpace]:
    gs = []
    highest_n_operands = max(operators_dict.keys())
    min_allowed_operator_index = 1
    operators = [x for y in operators_dict.values() for x in y]
    n_operators = len(operators)
    n_functions = n_operators + n_inputs

    for i_node in range(config.n_nodes):

        # Build the range of allowed operators
        if i_node <= highest_n_operands:
            # Operator range is based on the number of preceding nodes
            n_allowed_operators = len(operators_dict[i_node])
            # An operator who's allowed to have values has a range from 0 to 0
            fun_min = np.zeros(n_functions)
            allowed = np.ones(n_allowed_operators)
            fun_max = np.concatenate((np.ones(n_inputs), # Input variables are always allowed
                                 np.zeros(n_operators - n_allowed_operators),
                                 allowed))
            function_range = GeneRange(min=fun_min, max=fun_max)
            print(f'Function {i_node}: {function_range}')
        else:
            # Normal restrictions on the whole set of operators
            function_range = GeneRange(min=np.zeros(n_functions), max=np.ones(n_functions))


            #operator_range = GeneRange(range=(-len(operators) - n_inputs + 1, config.max_constant))

        gs.append(GeneSpace(function_range))

        # Todo: check this
        # Ensure DAG by only connecting to previous node indices in loop
        con_min = np.zeros(config.n_nodes)
        allowed = np.ones(i_node)
        con_max = np.concatenate((allowed,
                                  np.zeros(config.n_nodes - i_node)))
        for i in range(config.max_node_arity):
            connection_range = GeneRange(min=con_min, max=con_max)
            gs.append(GeneSpace(connection_range))
            print(f'Connection {i_node}: {connection_range}')

    # At the end, add root genes
    root_min = np.zeros(config.n_nodes)
    root_max = np.ones(config.n_nodes)
    root_range = GeneRange(min=root_min, max=root_max)

    for i in range(config.n_outputs):
        gs.append(GeneSpace(root_range))
        print(f'Root {i}: {root_range}')
    return gs


# Population for Cartesian representation
class CartesianPopulation(Population):
    def __init__(self,
                 config: OptimizerConfig = OptimizerConfig(),
                 operators_dict: dict[int, List[Operator]] = SIMPLE_OPERATORS_DICT,
                 state_space: gym.Space = None) -> None:
        global min_allowed_index
        self.config = config.program
        self.optim_config = config
        self.state_space = state_space
        self.operators_dict = operators_dict
        self.operators = [x for y in self.operators_dict.values() for x in y]
        self.n_inputs = np.prod(state_space.shape)

        # List of individuals that have been realized into programs recently
        self.realizations = []

        # Node represented by
        # gene 0: distribution over all operators
        # gene 1 to max_arity-1: connections with other nodes

        # last output_nodes genes: output indicators

        #    <--------------------------- Genome ---------------------------->
        #  */=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/*
        #    | | | | |   |  | | | | | | | | | | | | | |  |  | | | | | | | | | |
        #  */=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/*
        #   <- Input ->  |     <- Operators/Inputs->     |   <- Root nodes ->


        # Construct genome space

        # Todo: [!] proper genome spaces can be give to PyGad optimizer
        self.genome_space = generate_cartesian_genome_space(self.config, self.n_inputs, self.operators_dict)

        # Create population from different gene spaces
        super().__init__(self.optim_config.n_individuals,
                         genome_length(self.config, self.n_inputs, self.operators), # WRONG
                         self.genome_space)

    def __str__(self) -> str:
        return f'Cartesian pop with {self.n_individuals} individuals of genome length {self.n_genes}'

    # Dunder for individual genome access
    def __getitem__(self, i) -> np.ndarray[float]:
        return self.individuals[i]

    # Get population as an array of genomes
    #def raw_genes(self) -> np.ndarray:
    #    return np.array([i.genes for i in self.individuals])

    # Get the realization of genome with index
    def realize(self, index: int):
        from program.realization import CartesianProgram
        values = self.individuals[index]
        genome = Genome(values=values, genome_space=self.genome_space, pop_index=index)
        realization = CartesianProgram(
            genome=genome,
            input_space=self.state_space,
            operators=self.operators,
            config=self.config
        )
        return realization

    # Realize the whole population
    def realize_all(self):
        res = []
        for i, individual in enumerate(self.individuals):
            res.append(self.realize(i))
        return res

    # Realize all subprograms of an individual as if they are single output programs
    def realize_subs(self, index: int):
        from program.realization import CartesianProgram
        res = []
        genome = self.individuals[index]
        for i in range(self.config.n_nodes):
            g = copy(genome)
            # Set output gene to index in the loop
            g[-self.config.n_outputs] = i # Todo better soloution for programs with multiple inputs
            g = Genome(values=g, genome_space=self.genome_space, pop_index=index)
            realization = CartesianProgram(
                genome=g,
                input_space=self.state_space,
                operators=self.operators,
                config=self.config
            )
            res.append(realization)
        return res

    # Get range description for PyGad optimizer
    def range_description(self) -> List[dict]:
        res = []
        for gene in self.genome_space:
            res.append(gene.gene_range.description())
        return res

    # Generate random program from population
    def random_program(self):
        from program.realization import CartesianProgram

        genome = Genome(genome_space=self.genome_space)

        from program.realization import CartesianProgram

        program = CartesianProgram(
            genome=genome,
            input_space=self.state_space,
            operators=self.operators,
            config=self.config
        )

        return program


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    operators = SIMPLE_OPERATORS
    c = OptimizerConfig()

    # Gene range test
    n_inputs = space.shape[0]
    n_functions = len(SIMPLE_OPERATORS) + n_inputs
    function_dist = DistributionalGeneSpace(n_values=n_functions)
    connection_dist = DistributionalGeneSpace(n_values=c.program.max_node_arity)
    root_dist = DistributionalGeneSpace(n_values=c.program.n_nodes)

    # Genome test
    #genome_len = genome_length(c.program, n_inputs, operators)
    #gs = [function_dist, connection_dist, root_dist]
    #genome = Genome(genome_length=genome_len, genome_space=gs)

    #gs = generate_cartesian_genome_space(c.program, n_inputs, SIMPLE_OPERATORS_DICT)
    #genome = Genome(genome_length=genome_len, genome_space=gs)


    # Genome test with Cartesian gene space
    pop = CartesianPopulation(config=c, operators_dict=SIMPLE_OPERATORS_DICT, state_space=space)


    print('done')

    # Gene space test
    #gs = OperatorGeneSpace(GeneRange(range=(-1, 0)), SIMPLE_OPERATORS)
    #print(gs[-0.4])
    #print(gs[-1])

    # Genome test
    #genome = Genome(n_genes=1, genome_space=[gs])
    #print(genome.express_gene(0))

    # Population test
    #config = OptimizerConfig()
    #env = gym.make('CartPole-v1')
    #env.reset()
    #space = env.observation_space
    #pop = CartesianPopulation(config, SIMPLE_OPERATORS_DICT, space)
    #print(pop)
