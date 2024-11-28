# +++ CM-GP/population +++
#
# Population components for evolution
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

from typing import List, Callable, Union
import numpy as np
from config import CartesianConfig, OptimizerConfig
import gymnasium as gym

from program import operators
from program.operators import Operator, SIMPLE_OPERATORS


# Abstract for different types of GeneSpace
class GeneSpace:
    def __init__(self, gene_range: tuple[float, float]) -> None:

        assert gene_range[0] < gene_range[1], 'Invalid gene space range'

        self.gene_range = gene_range

    # Sampling method within the range of the space
    def sample(self, strategy: str = 'uniform'):

        if strategy == 'uniform':
            return np.random.uniform(*self.gene_range)
        else:
            raise ValueError(f'Unknown sampling strategy {strategy}')

    # Different types of rounding possible
    def _round(self, value: float, mode: str = 'regular') -> int:

        if mode == 'regular':
            res = round(value)
        elif mode == 'stochastic':
            res = int(value + (np.random.random() - 0.5))
        else:
            raise ValueError(f'Unknown rounding type: {mode}')

        assert self.gene_range[0] <= res <= self.gene_range[1], 'Rounding value out of gene range'

        return res


# Gene Space for Operators
class OperatorGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[float, float], operators: List[Operator]) -> None:
        super().__init__(gene_range)
        self.operators = operators
        self.n_inputs = self.gene_range[0] + len(operators)

    # Return corresponding realization from the gene space
    def __getitem__(self, value: float) -> Union[float, Operator]:

        # Non-constant encoded as negative value
        if value < 0:
            index = self._round(-value)

            # Value is index in input space
            if index > len(self.operators) - 1:  # We negated the index, so we need >
                return index
            # Value is an operator
            else:
                return self.operators[index]

        # Constant
        else:
            return value

    def __str__(self) -> str:
        return f'Operator space {self.gene_range[0]}->{self.gene_range[1]} with {len(self.operators)} operator and '


# Gene space for Cartesian coordinates
class CartesianGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[float, float]) -> None:
        super().__init__(gene_range)

    # Just return the value
    def __getitem__(self, value: float, *args, **kwargs) -> float:
        return self._round(value)

    def __str__(self) -> str:
        return f'Cartesian space {self.gene_range[0]}->{self.gene_range[1]}'


# Genome of individual program
class Genome:
    def __init__(self, n_genes: int, gene_spaces: List[GeneSpace]) -> None:
        self.n_genes = n_genes
        self.gene_spaces = gene_spaces
        self.genome = np.zeros(n_genes)
        self._init_genome()

    # Dunder for genome
    def __str__(self) -> str:
        return str(f'{self.genome}')

    # Makes circular list possible
    def _get_gene_space(self, i):
        return self.gene_spaces[i % len(self.gene_spaces)]

    # Sample gene values from the respective gene spaces
    def _init_genome(self) -> None:
        for i, gene in enumerate(self.genome):
            gene_space = self._get_gene_space(i)
            self.genome[i] = gene_space.sample()


# Abstract for population of individual genomes
class Population:
    def __init__(self, n_individuals: int, n_genes: int, gene_spaces: List[GeneSpace]) -> None:
        self.n_individuals = n_individuals
        self.n_genes = n_genes
        self.gene_spaces = gene_spaces

        self.individuals = []
        self._init_population()

    def __getitem__(self, i) -> Genome:
        return self.individuals[i]

    # Initialize population by populating it with genomes
    def _init_population(self) -> None:
        for i in range(self.n_individuals):
            genome = Genome(self.n_genes, self.gene_spaces)
            self.individuals.append(genome)


# Population for Cartesian representation
class CartesianPopulation(Population):
    def __init__(self,
                 config: OptimizerConfig = OptimizerConfig(),
                 operators: List[Operator] = SIMPLE_OPERATORS,
                 state_space: gym.Space = None) -> None:
        self.config = config.program
        self.optim_config = config
        self.n_genes = 2 * self.config.n_nodes + self.config.max_node_arity
        self.state_space = state_space
        self.n_inputs = np.prod(state_space.shape)

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


if __name__ == '__main__':
    # Gene space test
    gs = OperatorGeneSpace((-5, 5), SIMPLE_OPERATORS)
    print(gs[-0.4])
    print(gs[-5])

    # Population test
    config = OptimizerConfig()
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    pop = CartesianPopulation(config, SIMPLE_OPERATORS, space)
    print(pop)
