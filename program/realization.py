# +++ CM-GP/program/realization +++
#
# Executable programs as the realization of the genome
#
# 28/11/2024 - Senne Deproost

import numpy as np
from typing import Callable, List, Union
import gymnasium as gym
from networkx.classes import nodes

from config import CartesianConfig
from envs.simple_envs import SimpleGoalEnv
from population import Genome, OperatorGeneSpace, CartesianGeneSpace
from operators import Operator, SIMPLE_OPERATORS, InputVar
from dataclasses import dataclass, asdict

# Encoding
# gene 0: function of the node
# gene 1: output node or not
# gene 2 to 2+max_arity-1 -> determined the arity of operator in set with the highest number of operands

# Node as part of the Cartesian graph
class Node:
    def __init__(self, function: Union[Operator, InputVar, float], output: bool, connections: List[int]):
        self.function = function
        self.output = output
        self.connections = connections

    def __str__(self) -> str:
        return f' [ {self.function} | {self.output} | {self.connections} ]'

# Program base class
class Program:
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator]):
        self.genome = genome
        self.input_space = input_space
        self.operators = operators


# Cartesian graph based program:
# Todo: List of operators can be removed since it is encapsulated in the genespace instance
class CartesianProgram(Program):
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator], config: CartesianConfig):
        super().__init__(genome, input_space, operators)
        self.genome = genome
        self.config = config
        self._realization = self._realize()  # Realization is giant lambda

    # Call dunder for easy execution
    def __call__(self, input: List[float]) -> float:
        res = self._realization(*input)
        return res

    # Dunder describing program
    # Todo: Correct printing of a program -> Maybe for the Program parent class?
    def __str__(self) -> str:
        pass

    # Realization of genome into callable
    def _realize(self, *input) -> List[Callable]:
        c = self.config

        # Accumulators
        res = []
        nodes = {
            'all': [],
            'output': []
        }

        # Amount of genes to encode a node
        genes_per_node = 2 + c.max_node_arity

        # Go over each set of genes and capture node + indices of the output nodes
        for i in range(c.n_nodes):
            # Process
            n_index = i*genes_per_node
            f_index, o_index, c_indices = n_index, n_index+1, (n_index+2, n_index+1+c.max_node_arity)
            operator = self.genome.express_gene(f_index)        # Gene space has list of operators that can be realized
            output = self.genome.express_gene(o_index) == 1     # Translate binary value to boolean
            connections = [int(self.genome.express_gene(j)) for j in c_indices]

            # Transform into node abstraction and put into accumulator
            node = Node(operator, output, connections)
            nodes['all'].append(node)
            if output:
                nodes['output'].append(i)

        # Recursive function, traversing backwards in the graph
        def traverse(index: int, acc=[]):
            node = nodes['all'][index]
            for connection in node.connections:
                traverse(connection, acc)
            fun = node.function
            if isinstance(fun, Operator):
                print('op')
            elif isinstance(fun, InputVar):
                print('in')
            elif isinstance(fun, float):
                print('fun')
            else:
                raise ValueError("Node from not allowed type")


        # Start backtracking form each output node

        node_res = []

        for output_idx in nodes['output']:
            node_res.append(traverse(output_idx))





        # Accumulate functions


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    input_size = env.observation_space.shape[0]

    # Test for 3 nodes of 4 genes
    c = CartesianConfig()
    c.n_nodes = 3
    gs = [
        OperatorGeneSpace((-len(SIMPLE_OPERATORS) - input_size, c.max_constant), SIMPLE_OPERATORS),
        CartesianGeneSpace((0, 1)),
        *[CartesianGeneSpace((0, c.n_nodes)) for _ in range(c.max_node_arity)],
    ]

    genome = Genome(n_genes=len(gs)*c.n_nodes, gene_spaces=gs)
    print(genome)

    # Test valid program
    genes = np.array([-14, 0.0, 1, 0.69991735,
                      -5, 0.0, 0, 1.37312973,
                      0, 1.0, 1, 2.34511764])

    genome = Genome(genes=genes, gene_spaces=gs)

    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)
    print(prog)
