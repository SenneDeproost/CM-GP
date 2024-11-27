# +++ CM-GP/population +++
#
# Population components for evolution
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

from typing import List, Callable, Union
import numpy as np
from program.operators import Operator, SIMPLE_OPERATORS


# Abstract for different types of GeneSpace
class GeneSpace:
    def __init__(self, gene_range: tuple[float, float]) -> None:
        self.gene_range = gene_range

    # Sampling method within the range of the space
    def sample(self, strategy: str = 'uniform'):

        if strategy == 'uniform':
            return np.random.uniform(*self.gene_range)
        else:
            raise ValueError(f'Unknown sampling strategy {strategy}')

    # Different types of rounding possible
    def _round(self, value: float, type: str = 'regular') -> int:

        if type == 'regular':
            return round(value)
        elif type == 'stochastic':
            return int(value + (np.random.random() - 0.5))
        else:
            raise ValueError(f'Unknown rounding type: {type}')


# Gene Space for Operators
class OperatorGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[float, float], operators: List[Operator]) -> None:
        super().__init__(gene_range)
        self.operators = operators

    # Return corresponding realization from the gene space
    def __call__(self, value: float, *args, **kwargs) -> Union[float, Operator]:

        # Non-constant encoded as negative value
        if value < 0:
            index = self._round(-value)
            return self.operators[index]
        else:
            return value

    def __str__(self) -> str:
        return f'Operator space {self.gene_range[0]}->{self.gene_range[1]} with {len(self.operators)} operator'


# Gene space for Cartesian coordinates
class CartesianGeneSpace(GeneSpace):
    def __init__(self, gene_range: tuple[float, float]) -> None:
        super().__init__(gene_range)

    # Just return the value
    def __call__(self, value: float, *args, **kwargs) -> float:
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
    def __init__(self, n_individuals: int, n_genes: int, gene_ranges: List[tuple[float, float]],
                 operators: List[Operator]) -> None:
        # Check if the amount of genes is a multiple of 3 (X, Y, operator)
        assert n_genes % 4 == 0, "Amount of genes must be divisible by 4"
        assert len(gene_ranges) % 4 == 0, "Number of gene ranges must be divisible by 4"

        # GCP encoding:
        # Node based:
        # gene 0: node X
        # gene 1: node Y
        # gene 2: function
        # gene 3: output node

        self.gene_ranges = gene_ranges
        super().__init__(n_individuals, n_genes, list([
            CartesianGeneSpace(self.gene_ranges[0]),
            CartesianGeneSpace(self.gene_ranges[1]),
            OperatorGeneSpace(self.gene_ranges[2], operators),
            CartesianGeneSpace(self.gene_ranges[3])]))


if __name__ == '__main__':
    pop = CartesianPopulation(10,
                              16,
                              [(.0, 10.0), (.0, 10.0), (-1.0, 1.0), (0., 10.0)],
                              operators=SIMPLE_OPERATORS)
    print(pop)
