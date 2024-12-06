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
    def sample(self, strategy: str = 'uniform', excludes: set = {}) -> Union[int, float]:

        match strategy:
            # Normal uniform sampling float
            case 'uniform':
                return np.random.uniform(*self.gene_range)
            # Sampling from range of integers, with excluded values
            case 'choice':
                r = list(np.arange(self.gene_range[0], self.gene_range[1] + 1))
                [r.remove(x) for x in list(excludes)]

                # Todo: better resolvement of last connections
                # Resolve connections at the end of the traversal
                if len(r) == 0:
                    return 0
                else:
                    return np.random.choice(r)

            # Invalid sampling strategy
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
        return f'Operator gene space {self.gene_range[0]}->{self.gene_range[1]} with {len(self.operators)} operators'

    # Todo: if needed, implement
    def reset_excludes(self) -> None:
        pass


# Todo: Check if exclusion works here
# Gene space for output nodes
class BooleanGeneSpace(GeneSpace):
    def __init__(self, counter: Union[Counter, bool, None] = None) -> None:
        # Boolean gene range
        super().__init__((0, 1))
        self.counter = counter
        # If excludes are not given but enabled, initialize new set of excludes
        if counter is True:
            self.counter = Counter(false=0, true=0)

    # Just return the value
    def __getitem__(self, value: int, *args, **kwargs) -> int:
        v = self._round(value)
        assert 0 <= v <= 1, 'Value out of gene range'
        return v

    # Todo: Solve loop when rounding stochastic
    # Sample with exclusions taken into account
    def sample(self, strategy: str = 'choice') -> int:
        v = super().sample(strategy)
        k = 'true' if v == 1 else 'false'
        self.counter[k] += 1
        return v

    # String representation dunder
    def __str__(self) -> str:
        return f'Boolean gene space {self.gene_range[0]}->{self.gene_range[1]}'

    # Reset excludes
    def reset_excludes(self) -> None:
        self.counter = Counter(false=0, true=0)


# Gene space for connection between nodes
class IntegerGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[int, int], excludes: Union[set, bool, None] = None) -> None:
        super().__init__(gene_range)

        self.excludes = excludes
        # If excludes are not given but enabled, initialize new set of excludes
        if excludes is True:
            self.excludes = set()

    # Just return the value
    def __getitem__(self, value: float, *args, **kwargs) -> float:
        v = self._round(value)
        assert self.gene_range[0] <= v <= self.gene_range[1], 'Value out of gene range'
        return v

    # Todo: Solve loop when rounding stochastic
    # Todo: Will there be an issue because of the seperate gene spaces?
    # Sample with exclusions taken into account
    def sample(self, strategy: str = 'choice') -> int:
        v = super().sample(strategy, excludes=self.excludes)
        self.excludes.add(v)
        return v

    # String representation dunder
    def __str__(self) -> str:
        return f'Integer gene space {self.gene_range[0]}->{self.gene_range[1]}'

    # Reset excludes
    def reset_excludes(self) -> None:
        self.excludes = set()


# Genome of individual program
class Genome:
    def __init__(self, genome_space: List[GeneSpace], n_genes: Union[int, None] = None,
                 pop_index: int = -1,
                 genes: np.ndarray = None) -> None:
        self.n_genes = n_genes
        self.genome_space = genome_space
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

    # Reset the gene spaces excludes
    def _reset_excludes(self) -> None:
        for gs in self.genome_space:
            gs.reset_excludes()

    # Sample gene values from the respective gene spaces. When gene space has exclusions, take them into account
    def _init_genome(self) -> None:

        # Loop over the genes
        for i, gene in enumerate(self.genes):
            gene_space = self._get_gene_space(i)
            self.genes[i] = gene_space.sample()

        # Reset the genome space excludes
        self._reset_excludes()

    # Helper function to return for every set of genes of a certain length the value
    def every_ith_gene(self, n: int, seq_len: int) -> list[float]:
        res = []
        for i in range(int(len(self) / seq_len)):
            res.append(self.genes[i*seq_len + n])
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
def resolve_output(genome: Genome) -> None:
    # Make first node an output node
    genome.genes[1] = 1

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
        connection_range = (0, self.config.n_nodes - 1)

        # Create population from different gene spaces
        super().__init__(config.n_individuals,
                         self.n_genes,
                         list([
                             OperatorGeneSpace(operator_range, operators),  # Function
                             BooleanGeneSpace(counter=True),  # Binary indicator if node is output
                             *[IntegerGeneSpace(connection_range, excludes={0}) for _ in  # Exclude first index
                               range(self.config.max_node_arity)]]))  # Connections between nodes
        # Resolve output
        for genome in self.individuals:
            if not has_output(genome, config=self.config):
                resolve_output(genome)

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
        BooleanGeneSpace(counter=True),  # Binary indicator for output node
        *[IntegerGeneSpace((0, config.n_nodes)) for _ in range(config.max_node_arity)],  # Receiving connections
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
