# +++ CM-GP/population +++
#
# Population components for evolution
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher
from collections import OrderedDict
from typing import List, Callable, Union
import numpy as np
from sympy.polys.polyoptions import Order

from config import CartesianConfig, OptimizerConfig
import gymnasium as gym

from program import operators
from program.operators import Operator, SIMPLE_OPERATORS, InputVar


# Calculate the amount of genes per node
def genes_per_node(config: CartesianConfig) -> int:
    n = 2 + config.max_node_arity
    return n


# Abstract for different types of GeneSpace
class GeneSpace:
    def __init__(self, gene_range: tuple[float, float]) -> None:

        assert gene_range[0] <= gene_range[1], 'Invalid gene space range'

        self.gene_range = gene_range

    # Sampling method within the range of the space
    def sample(self, strategy: str = 'uniform') -> np.ndarray:

        match strategy:
            case 'uniform':
                return np.random.uniform(*self.gene_range)
            case _:
                raise ValueError(f'Unknown sampling strategy {strategy}')

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
    def __init__(self, gene_range: tuple[float, float], operators: List[Operator]) -> None:
        super().__init__(gene_range)
        self.operators = operators
        self.n_inputs = self.gene_range[0] + len(operators)

    # Return corresponding realization from the gene space
    def __getitem__(self, value: float) -> Union[float, Operator, InputVar]:

        assert self.gene_range[0] <= value <= self.gene_range[1], 'Value out of gene range'

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

    def __str__(self) -> str:
        return f'Operator space {self.gene_range[0]}->{self.gene_range[1]} with {len(self.operators)} operators'


# ToDo: Exclude own index to prevent infinite loop
# Gene space for Cartesian coordinates
class CartesianGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[float, float], excludes: Union[OrderedDict, None] = None) -> None:
        super().__init__(gene_range)
        self.excludes = excludes  # If given, excludes will be applied in sampling

    # Just return the value.
    def __getitem__(self, value: float, *args, **kwargs) -> float:
        assert self.gene_range[0] <= value <= self.gene_range[1], 'Value out of gene range'
        return self._round(value)

    def __str__(self) -> str:
        return f'Cartesian space {self.gene_range[0]}->{self.gene_range[1]}'


# Genome of individual program
class Genome:
    def __init__(self, genome_space: List[GeneSpace], n_genes: Union[int, None] = None,
                 genes: np.ndarray = None) -> None:
        self.n_genes = n_genes
        self.genome_space = genome_space

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
        for i, gene in enumerate(self.genes):
            gene_space = self._get_gene_space(i)
            self.genes[i] = gene_space.sample()


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
            genome = Genome(n_genes=self.n_genes, genome_space=self.genome_space)
            self.individuals.append(genome)


# Population for Cartesian representation
class CartesianPopulation(Population):
    def __init__(self,
                 config: OptimizerConfig = OptimizerConfig(),
                 operators: List[Operator] = SIMPLE_OPERATORS,
                 state_space: gym.Space = None) -> None:
        self.config = config.program
        self.optim_config = config
        self.n_genes = self.config.n_nodes * genes_per_node(self.config)
        self.state_space = state_space
        self.n_inputs = np.prod(state_space.shape)

        # List of individuals that have been realized into programs recently
        self.realizations = []

        # GCP encoding for N nodes with n_outputs and max_arity
        # gene 0 -> N-1: function
        # gene N -> 2N*max_arity - 1: connections made between max_arity nodes (-1 indication no connection)
        # gene 2N*max_arity -> 2N*max_arity+n_outputs - 1: output nodes

        # ---> Translatable as:
        # Node represented by
        # gene 0: function
        # gene 1: output? --> Differs from CGPAX
        # gene 2 to 2+max_arity-1 -> determined the arity of operator in set with the highest number of operands

        operator_range = (-len(operators) - self.n_inputs, self.config.max_constant)
        output_range = (0, 1)
        connection_range = (0, self.config.n_nodes - 1)

        # Create population from different gene spaces
        super().__init__(config.n_individuals,
                         self.n_genes,
                         list([
                             OperatorGeneSpace(operator_range, operators),  # Function
                             CartesianGeneSpace(output_range),  # Binary indicator if node is output
                             *[CartesianGeneSpace(connection_range) for _ in
                               range(self.config.max_node_arity)]]))  # Connections between nodes

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
        individual = self.individuals[index]
        realization = individual.realize()
        return realization

    # Realize the whole population
    def realize_all(self):
        res = []
        for individual in self.individuals:
            individual.realize()
        return res


# Todo: check role of input_size
# Generate Cartesian gene space
def generate_cartesian_genome_space(config: CartesianConfig, input_size: int) -> List[GeneSpace]:
    gs = [
        OperatorGeneSpace(  # Operator
            (-len(SIMPLE_OPERATORS) - input_size, config.max_constant),
            SIMPLE_OPERATORS),
        CartesianGeneSpace((0, 1)),  # Binary indicator for output node
        *[CartesianGeneSpace((0, config.n_nodes)) for _ in range(config.max_node_arity)],  # Receiving connections
    ]
    return gs


if __name__ == '__main__':
    # Gene space test
    gs = OperatorGeneSpace((-1, 0), SIMPLE_OPERATORS)
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
    pop = CartesianPopulation(config, SIMPLE_OPERATORS, space)
    print(pop)
