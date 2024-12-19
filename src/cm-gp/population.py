# +++ CM-GP/population +++
#
# Population components for evolution
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher
from collections import OrderedDict
from typing import List, Callable, Union, Counter
import numpy as np
from sympy.polys.polyoptions import Order
from tensorflow.python.ops.gen_nn_ops import selu_grad

from config import CartesianConfig, OptimizerConfig
import gymnasium as gym

from program import Operator, InputVar, SIMPLE_OPERATORS_DICT, SIMPLE_OPERATORS

EMPTY = -1


# Calculate the amount of genes per node
def genes_per_node(config: CartesianConfig) -> int:
    n = 1 + config.max_node_arity  # One gene for function and rest for connection
    return n


# Class for gene ranges
class GeneRange:
    def __init__(self, range: Union[tuple[int, int], tuple[float, float]] = None,
                 values: Union[None, list[float], list[int]] = None) -> None:

        # Check if the gene range is valid
        if range is not None:
            assert range[0] <= range[1], "Invalid range"

        self.range = range
        self.values = values
        self.empty = False

        # Range can be empty
        if range is None and values is None:
            self.empty = True

    def __getitem__(self, i: int) -> Union[float, int, None]:
        if self.empty:
            return None
        elif self.values is not None:
            return self.values[i]
        else:
            return self.range[i]

    # Dunder for printing
    def __str__(self) -> str:
        if self.range is None:
            return f'Gene range with values {self.values}'
        else:
            return f'Gene range {self.range[0]}->{self.range[1]}'

    # Membership test
    def contains(self, value: Union[int, float]) -> bool:
        # Empty
        if self.empty:
            return False
        # Range
        elif not (self.range is None):
            return self.range[0] <= value <= self.range[1]
        # Values
        elif not (self.values is None):
            return value in self.values

    # Translate for optimizer
    def description(self) -> Union[dict, list]:
        if self.range is None:
            if self.values is None:
                return [0]  # Todo: Proper resolvement
            else:
                return self.values
        else:
            return {
                'low': self.range[0],
                'high': self.range[1]
            }


# Abstract for different types of GeneSpace
class GeneSpace:
    def __init__(self, gene_range: GeneRange) -> None:
        self.gene_range = gene_range

    # Sampling method within the range of the space
    def sample(self, strategy: str = 'uniform') -> Union[int, float]:

        # Check for empty range
        if self.gene_range.empty:
            return EMPTY

        # Strategies for sampling
        match strategy:

            # Normal uniform sampling float
            case 'uniform':
                if not (self.gene_range.values is None):
                    return np.random.choice(self.gene_range.values)
                else:
                    assert self.gene_range.range is not None, "Gene values not from range"
                    return np.random.uniform(self.gene_range[0], self.gene_range[1])

            # Invalid sampling strategy
            case _:
                raise ValueError(f'Unknown sampling strategy {strategy}')

    # Ensure the value is within range or in between values
    def contain(self, value: Union[int, float]) -> Union[int, float]:

        c = self.gene_range.range if self.gene_range.range is not None else self.gene_range.values

        if value <= c[0]:
            return c[0]
        elif value >= c[-1]:
            return c[-1]
        else:
            return value

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


# Gene Space for Operators
class OperatorGeneSpace(GeneSpace):
    def __init__(self, gene_range: GeneRange,
                 operators: List[Operator]) -> None:
        super().__init__(gene_range)
        self.operators = operators

    # Return corresponding realization from the gene space
    def __getitem__(self, value: float) -> Union[float, Operator, InputVar]:

        # Non-constant encoded as negative value
        if value <= 0:
            index = self._round(-value)

            # Value is index in input space
            if index > len(self.operators) - 1:  # We negated the index, so we need >
                index = index - len(self.operators)
                return InputVar(index)
            # Value is an operator
            else:
                return self.operators[index]

        # Constant
        else:
            return value

    # Sample random
    def sample(self, strategy: str = 'uniform') -> Union[int, float]:

        value = super().sample()

        return value

    def __str__(self) -> str:
        return f'Operator gene space with {str(self.gene_range)}'


# Todo: Check if exclusion works here
# Gene space for output nodes
class BinaryGeneSpace(GeneSpace):
    def __init__(self, gene_range: GeneRange = GeneRange(values=[0, 1])) -> None:
        # Boolean gene range
        super().__init__(gene_range)

    # Just return the value
    def __getitem__(self, value: int, *args, **kwargs) -> int:
        v = self._round(value)
        return self.contain(v)

    # Todo: Solve loop when rounding stochastic
    # Sample with exclusions taken into account
    def sample(self, strategy: str = 'uniform') -> int:
        v = super().sample(strategy)
        return v

    # String representation dunder
    def __str__(self) -> str:
        return f'Binary gene space {str(self.gene_range)}'


# Gene space for connection between nodes
class IntegerGeneSpace(GeneSpace):
    def __init__(self, gene_range: GeneRange) -> None:
        super().__init__(gene_range)

    # Just return the value
    def __getitem__(self, value: float, *args, **kwargs) -> float:
        v = self._round(value)
        # Check if there is no indication of first node resolvement in the given gene space
        if not self.gene_range.empty:
            assert self.gene_range.contains(v), 'Value out of gene range'  # Check after rounding
        return v

    # Todo: Solve loop when rounding stochastic
    # Todo: Will there be an issue because of the seperate gene spaces?
    # Sample with exclusions taken into account
    def sample(self, strategy: str = 'uniform') -> int:
        v = super().sample(strategy)
        return v

    # String representation dunder
    def __str__(self) -> str:
        return f'Integer gene space {str(self.gene_range)}'


# Todo: Keep indices of the type of nodes
# Genome of individual program
class Genome:
    def __init__(self, genome_space: List[GeneSpace], n_genes: Union[int, None] = None,
                 pop_index: int = -1,
                 genes: np.ndarray = None) -> None:
        self.n_genes = n_genes
        self.genome_space = genome_space  #[y for x in genome_space for y in x]  # Flatten incoming genome space
        self.pop_index = pop_index

        # Check if genes are given or need to be initialized by the genome
        if genes is not None:
            self.genes = genes
            self.n_genes = len(genes)
        else:
            self.genes = np.zeros(n_genes)
            self._init_genome()

    # Dunder for genome length
    def __len__(self) -> int:
        return self.n_genes

    # Dunder for genome
    def __str__(self) -> str:
        return str(f'{self.genes}')

    # Accessor to gene in genome, returning value from corresponding gene space
    def express_gene(self, index: int) -> Union[float, Operator]:
        gene_space = self._get_gene_space(index)  # Circular
        gene = self.genes[index]
        return gene_space[gene]

    # Makes circular list possible
    def _get_gene_space(self, i):
        return self.genome_space[i % len(self.genome_space)]

    # Sample gene values from the respective gene spaces. When gene space has exclusions, take them into account
    def _init_genome(self) -> None:

        # Loop over the genes
        for i, gene in enumerate(self.genes):
            gene_space = self.genome_space[i]
            self.genes[i] = gene_space.sample()

    # Helper function to return for every set of genes of a certain length the value
    def every_ith_gene(self, n: int, seq_len: int) -> list[float]:
        res = []
        for i in range(int(len(self) / seq_len)):
            res.append(self.genes[i * seq_len + n])
        return res


# Abstract for population of individual genomes
class Population:
    def __init__(self, n_individuals: int, n_genes: int, genome_space: List[GeneSpace]) -> None:
        self.n_individuals = n_individuals
        self.n_genes = n_genes
        self.genome_space = genome_space

        self.individuals = []
        self._init_population()

    def __getitem__(self, i) -> Genome:
        return self.individuals[i]

    # Initialize population by populating it with genomes
    def _init_population(self) -> None:
        for i in range(self.n_individuals):
            genome = Genome(n_genes=self.n_genes, genome_space=self.genome_space, pop_index=i)
            self.individuals.append(genome)


# Helper function to return all output node genes
def has_output(genome: Genome, config: CartesianConfig) -> bool:
    outputs = genome.every_ith_gene(n=1, seq_len=genes_per_node(config))
    has = 1 in outputs
    return has


# Resolve genomes with no output node
def resolve_output(genome: Genome, config: CartesianConfig) -> None:
    # Make last node an output node
    i = -(config.max_node_arity + 1)
    genome.genes[i] = 1


# Generate Cartesian gene space
def generate_cartesian_genome_space(config: CartesianConfig,
                                    n_inputs: int,
                                    operators_dict: dict[int, List[Operator]]) -> List[GeneSpace]:
    gs = []
    highest_n_operands = max(operators_dict.keys())
    min_allowed_operator_index = 1
    operators = [x for y in operators_dict.values() for x in y]

    for i_node in range(config.n_nodes):

        # Build the range of allowed operators
        if i_node <= highest_n_operands:
            # Operator range is based on the amount of preceding nodes
            n_operators = len(operators_dict[i_node])
            min_allowed_operator_index -= n_operators
            operator_range = GeneRange(range=(min_allowed_operator_index, config.max_constant))
        else:
            # Normal restrictions on the whole set of operators
            operator_range = GeneRange(range=(-len(operators) - n_inputs + 1, config.max_constant))

        # Todo: check this
        # Ensure DAG by only connecting to previous node indices in loop
        connection_range = GeneRange() if i_node < n_inputs else GeneRange(range=(0, i_node - 1))

        #
        # INPUT
        #

        # The first nodes should be input nodes
        if i_node <= n_inputs - 1:

            # Index of the corresponding input variable
            # input_0 at 0
            # input_1 at 1
            # etc..

            input_range = GeneRange(values=[-len(operators) - i_node])

            # Operator, which is input variable of index i_node
            gs.append(OperatorGeneSpace(gene_range=input_range, operators=operators))

            # Connections
            for _ in range(config.max_node_arity):
                gs.append(IntegerGeneSpace(connection_range))


        #
        # OPERATORS AND CONSTANTS
        #

        # Followed by operators
        else:

            # Ensure that operators are not selected that require more operands than currently available

            # Operator
            gs.append(OperatorGeneSpace(operator_range, operators))

            # Connections
            for _ in range(config.max_node_arity):
                gs.append(IntegerGeneSpace(connection_range))

    #
    # OUTPUT INDICATORS
    #

    # Followed by output indicators

    output_range = GeneRange(range=(0, config.n_nodes - 1))

    for _ in range(config.n_outputs):
        gs.append(IntegerGeneSpace(output_range))

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

        self.genes_per_node = genes_per_node(self.config)
        self.n_genes = self.config.n_nodes * self.genes_per_node + self.config.n_outputs
        self.n_inputs = np.prod(state_space.shape)

        # List of individuals that have been realized into programs recently
        self.realizations = []

        # Node represented by
        # gene 0: function
        # gene 1 to max_arity-1: operators

        # last output_nodes genes: output indicators

        #     <------------------- Genome -------------------->
        #  */=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/=/* | */=/=/=/=/=/*
        #    | | | | |   |  | | | | | | | | | |    |  | | | | | |
        #  */=/=/=/=/=/* | */=/=/=/=/=/=/=/=/=/=/* | */=/=/=/=/=/*
        #   <- Input ->  |     <- Operators ->     |  <- Output ->

        #inputvar_range = (-len(operators) - self.n_inputs, -len(operators))

        # Construct genome space

        # Todo: [!] proper genome spaces can be give to PyGad optimizer
        self.genome_space = generate_cartesian_genome_space(self.config, self.n_inputs, self.operators_dict)

        # Create population from different gene spaces
        super().__init__(config.n_individuals,
                         self.n_genes,
                         self.genome_space)

        # Resolve output
        # Implement better resolvement
        #for genome in self.individuals:
        #    if not has_output(genome, config=self.config):
        #        resolve_output(genome, config=self.config)

    def __str__(self) -> str:
        return f'Cartesian pop with {self.n_individuals} individuals of genome length {self.n_genes}'

    # Dunder for individual genome access
    def __getitem__(self, i):
        return self.individuals[i].genes

    # Get population as an array of genomes
    def raw_genes(self) -> np.ndarray:
        return np.array([i.genes for i in self.individuals])

    # Get the realization of genome with index
    def realize(self, index):
        from program.realization import CartesianProgram
        individual = self.individuals[index]
        realization = CartesianProgram(
            genome=individual,
            input_space=self.state_space,
            operators=self.operators,
            config=self.config,
        )
        return realization

    # Realize the whole population
    def realize_all(self):
        res = []
        for i, individual in enumerate(self.individuals):
            self.realize(i)
        return res

    # Get range description for PyGad optimizer
    def range_description(self) -> List[dict]:
        res = []
        for gene in self.genome_space:
            res.append(gene.gene_range.description())
        return res

    # Update all genes of the population
    def update(self, new_population: np.ndarray[float]) -> None:
        for i, genome in enumerate(self.individuals):
            self.individuals[i].genes = new_population[i]

if __name__ == '__main__':
    # Gene space test
    gs = OperatorGeneSpace(GeneRange(range=(-1, 0)), SIMPLE_OPERATORS)
    print(gs[-0.4])
    print(gs[-1])

    # Genome test
    genome = Genome(n_genes=1, genome_space=[gs])
    print(genome.express_gene(0))

    # Population test
    config = OptimizerConfig()
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    pop = CartesianPopulation(config, SIMPLE_OPERATORS_DICT, space)
    print(pop)