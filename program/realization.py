# +++ CM-GP/program/realization +++
#
# Executable programs as the realization of the genome
#
# 28/11/2024 - Senne Deproost

import numpy as np
from typing import Callable, List, Union, Any, Tuple
import gymnasium as gym
import inspect

from numpy import ndarray

from config import CartesianConfig
from envs.simple_envs import SimpleGoalEnv
from population import Genome, OperatorGeneSpace, CartesianGeneSpace
from program.operators import Operator, SIMPLE_OPERATORS, InputVar
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
        res = self.evaluate(*input)
        return res

    # Dunder describing program
    # Todo: Correct printing of a program -> Maybe for the Program parent class?
    def __str__(self) -> str:
        return "test"

    # Process the nodes
    def _process_nodes(self):
        # Amount of genes to encode a node
        genes_per_node = 2 + self.config.max_node_arity

        # Accumulator
        nodes = {
            'all': [],
            'output': []
        }

        # Go over each set of genes and capture node + indices of the output nodes
        for i in range(self.config.n_nodes):
            # Process
            n_index = i * genes_per_node
            f_index, o_index, c_indices = n_index, n_index + 1, (n_index + 2, n_index + 1 + self.config.max_node_arity)
            operator = self.genome.express_gene(f_index)  # Gene space has list of operators that can be realized
            output = self.genome.express_gene(o_index) == 1  # Translate binary value to boolean
            connections = [int(self.genome.express_gene(j)) for j in c_indices]

            # Transform into node abstraction and put into accumulator
            node = Node(operator, output, connections)
            nodes['all'].append(node)
            if output:
                nodes['output'].append(i)

        return nodes

    # Recursive function, traversing backwards in the graph
    @staticmethod
    def _traverse(node: Node,
                  on_operator: Callable, on_float: Callable, on_inputvar: Callable) -> list[Any]:

        fun = node.function

        # Operator
        if isinstance(fun, Union[Operator, tuple]):
            return on_operator(node)

        # Input variable
        elif isinstance(fun, Union[InputVar, Callable]):
            return on_inputvar(node)

        # Float
        elif isinstance(fun, float):
            return on_float(node)

        else:
            raise ValueError("Node with invalid type")

    # Realization of genome into callable
    def _realize(self) -> list[list[Any]]:

        # Accumulator
        res = []

        # Process nodes into traversable structure
        nodes = self._process_nodes()

        # Functions for different types of nodes
        # Operator
        def on_operator(node: Node) -> Operator:
            n_connections = node.function.n_operands
            operator = node.function

            # Accumulate branches
            for connection in node.connections[:operator.n_operands]:  # Limit to n_operand connections
                connected_node = nodes['all'][connection]
                operator.operands.append(self._traverse(connected_node, on_operator, on_float, on_inputvar))

            return operator

        # Input variable
        def on_inputvar(node: Node) -> InputVar:
            return node.function

        # Float
        def on_float(node: Node) -> float:
            return node.function

        # Start backtracking form each output node
        for output_idx in nodes['output']:
            node = nodes['all'][output_idx]
            r = self._traverse(node, on_operator, on_float, on_inputvar)
            res.append(r)

        # Result is a list of all backtracks from all output nodes, summed with the sum operator
        return res

    # Evaluate the realized function
    def evaluate(self, f: Union[Callable, float, tuple], input: ndarray) -> float:

        res = 0.0

        # ToDo: Single float and input var cases

        # When multiple functions are given, accumulate their results

        if len(f) > 1:
            for _f in f:
                res += self._evaluate(_f, input)
        else:
            res = self._evaluate(f[0], input)

        return res

    def _evaluate(self, f: Union[Callable, float, tuple], input: ndarray) -> float:

        # Function
        if isinstance(f, Operator):
            operands = []

            # Accumulate branches
            for operand in f.operands:
                operands.append(self._evaluate(operand, input))

            # Todo: better apply of operator to operands
            return f(operands)

        # Input variable
        elif isinstance(f, InputVar):
            return f(input)

        # Float
        elif isinstance(f, float):
            return f

        else:
            raise ValueError("Unsupported function")


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.reset()
    space = env.observation_space
    input_size = env.observation_space.shape[0]

    # Test for 3 nodes of 4 genes
    c = CartesianConfig()
    c.n_nodes = 4
    gs = [
        OperatorGeneSpace((-len(SIMPLE_OPERATORS) - input_size, c.max_constant), SIMPLE_OPERATORS),
        CartesianGeneSpace((0, 1)),
        *[CartesianGeneSpace((0, c.n_nodes)) for _ in range(c.max_node_arity)],
    ]

    genome = Genome(n_genes=len(gs) * c.n_nodes, gene_spaces=gs)
    print(genome)

    # Test valid program
    genes = np.array([-14, 0.0, 1, 0.69991735,  # [ input_0 | False | [1, 1] ]
                      -11, 0.0, 0, 1.37312973,  # [ id      | False | [0, 1] ]
                      5, 0.0, 0, 0,  # [ 5.0     | False | [0, 0] ]
                      0, 1.0, 1, 2.34511764])  # [ +       | True  | [1, 2] ]

    genome = Genome(genes=genes, gene_spaces=gs)

    i = np.array([-9, 99, 999, 9999])
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)
    res = prog.evaluate(prog._realization, i)
    print(prog)