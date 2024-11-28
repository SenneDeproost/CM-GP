# +++ CM-GP/program/realization +++
#
# Executable programs as the realization of the genome
#
# 28/11/2024 - Senne Deproost

import numpy as np
from typing import Callable, List
import gymnasium as gym
from networkx.classes import nodes

from population import Genome, OperatorGeneSpace, CartesianGeneSpace
from operators import Operator, SIMPLE_OPERATORS
from dataclasses import dataclass, asdict


# Encoding
# gene 0: function of the node
# gene 1: output node or not
# gene 2 to 2+max_arity-1 -> determined the arity of operator in set with the highest number of operands

# Node as part of the Cartesian graph
@dataclass
class Node:
    function: Callable
    output: bool
    connections: List[int]


# Program base class
class Program:
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator]):
        self.genome = genome
        self.input_space = input_space
        self.operators = operators


# Cartesian graph based program:
class CartesianProgram(Program):
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator]):
        super().__init__(genome, input_space, operators)
        self._realization = self._realize()  # Realization is giant lambda

    # Call dunder for easy execution
    def __call__(self, input: List[float]) -> float:
        res = self._realization(*input)
        return res

    # Realization of genome into callable
    def _realize(self, *input) -> List[Callable]:
        res = []

        # Map the expressions of the node
        #length npde
        genes = self.genome.genes.reshape((4,-1))
        #for gene in genes:
        #    Node(
        #        function =
        #    )
        print('donme')

        # Start backtracking form each output node

        # Accumulate functions


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space

    # Test for 5 nodes of 4 genes
    max_node_arity = 2
    gs = [
        OperatorGeneSpace((-5, 5), SIMPLE_OPERATORS),
        CartesianGeneSpace((0, 1)),
        *[CartesianGeneSpace((0, 5)) for _ in range(2)],
    ]

    genome = Genome(n_genes=20, gene_spaces=gs)
    print(genome)

    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS)
    print(prog)
