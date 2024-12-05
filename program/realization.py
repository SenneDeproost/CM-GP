# +++ CM-GP/program/realization +++
#
# Executable programs as the realization of the genome
#
# 28/11/2024 - Senne Deproost

import numpy as np
from typing import Callable, List, Union, Any, Tuple
import gymnasium as gym
import test

from numpy import ndarray

from config import CartesianConfig
from envs.simple_envs import SimpleGoalEnv
from population import Genome, OperatorGeneSpace, generate_cartesian_genome_space, genes_per_node
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
        self.connected_nodes = []

    def __str__(self) -> str:
        return f' [ {self.function} | {self.output} | {self.connections} ]'

    # Helper function for traversal
    def traverse(self, input: Union[ndarray[float], None],
                 on_operator: Callable, on_inputvar: Callable, on_float: Callable):

        function = self.function

        # Operator
        if isinstance(function, Operator):
            return on_operator(self, input)

        # Input variable
        elif isinstance(function, InputVar):
            return on_inputvar(self, input)

        # Float
        elif isinstance(function, float):
            return on_float(self, input)

        else:
            raise ValueError("Node with invalid function type")


# Program base class
class Program:
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator]):
        self.genome = genome
        self.input_space = input_space
        self.operators = operators


def SumString(x: list[str]) -> str:
    res = f'∑[ {x[0]}'
    for x in x[1:]:
        res += f', {x}'
    return f'{res} ]'


# Cartesian graph based program:
# Todo: List of operators can be removed since it is encapsulated in the genespace instance
class CartesianProgram(Program):
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator], config: CartesianConfig):
        super().__init__(genome, input_space, operators)
        self.genome = genome
        self.config = config
        self._realization = self._process_nodes()  # Realization nesting of Nodes
        self._str = self.to_string()

    # Call dunder for easy execution
    def __call__(self, input: List[float]) -> float:
        res = self.evaluate(*input)
        return res

    # Dunder describing program
    # Todo: Correct printing of a program -> Maybe for the Program parent class?
    def __str__(self) -> str:
        return self._str

    # Process the nodes
    def _process_nodes(self):

        _genes_per_node = genes_per_node(self.config)

        # Accumulator
        nodes = {
            'all': [],
            'output': []
        }

        # Go over each set of genes and capture node + indices of the output nodes
        for i in range(self.config.n_nodes):

            # Process
            n_index = i * _genes_per_node
            f_index, o_index, start_c_index, stop_c_index = n_index, n_index + 1, n_index + 2, n_index + 1 + self.config.max_node_arity
            operator = self.genome.express_gene(f_index)  # Gene space has list of operators that can be realized
            output = self.genome.express_gene(o_index) == 1  # Translate binary value to boolean
            connections = [int(self.genome.express_gene(j)) for j in range(start_c_index, stop_c_index)]

            # Transform into node abstraction and put into accumulator
            node = Node(operator, output, connections)
            nodes['all'].append(node)

            # Add node to output when indicated
            if output:
                nodes['output'].append(node)

        # Check if there is an output node, otherwise it is an invalid program
        if len(nodes['output']) < 1:
            raise ValueError(f"No output in {self.genome}")

        # Go over a second time for setting the connections and operands
        for node in nodes['all']:
            function = node.function

            ## Cases need to be separate conditional outside connection loop
            # Operator case
            if isinstance(function, Operator):
                for connection in node.connections[:function.n_operands]:
                    connected_node = nodes['all'][connection]
                    node.connected_nodes.append(connected_node)
                    function.operands.append(connected_node)

            # All other cases
            else:
                for connection in node.connections:
                    connected_node = nodes['all'][connection]
                    node.connected_nodes.append(connected_node)

        return nodes

    # Evaluate the realized function
    def evaluate(self, input: ndarray[float]) -> float:

        res = 0
        outputs = self._realization['output']

        # Operator
        def on_operator(node: Node, input: Union[None, ndarray[float]]) -> Operator:
            operands = []
            for operand in node.function.operands:
                operands.append(operand.traverse(input, on_operator, on_inputvar, on_float))
            return node.function(operands)

        # Input variable
        def on_inputvar(node: Node, input: Union[None, ndarray[float]]) -> float:
            return node.function(input)

        # Float
        def on_float(node: Node, input: Union[None, ndarray[float]]) -> float:
            return node.function

        # Sum over all output nodes
        for o in outputs:
            res += o.traverse(input, on_operator, on_inputvar, on_float)

        return res

    # Return string representation of program. When input is not given, placeholder names are used
    def to_string(self, input: Union[None, ndarray[float]] = None) -> list[Any] | Any:

        res = []
        outputs = self._realization['output']

        # Operator
        def on_operator(node: Node, input: Union[None, ndarray[float]]) -> str:
            operands = []
            for operand in node.function.operands:
                operands.append(operand.traverse(input, on_operator, on_inputvar, on_float))
            return node.function.print(operands)

        # Input variable
        def on_inputvar(node: Node, input: Union[None, ndarray[float]]) -> str:
            return node.function.to_string(input)

        # Float
        def on_float(node: Node, input: Union[None, ndarray[float]]) -> str:
            return str(node.function)

        for o in outputs:
            res.append(o.traverse(input, on_operator, on_inputvar, on_float))

        # Sum over results if multiple output nodes are present
        if len(res) == 1:
            return res[0]
        else:
            return SumString(res)


if __name__ == "__main__":
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    # Test for 4 nodes of 4 genes
    c = CartesianConfig()
    c.n_nodes = 4
    gs = generate_cartesian_genome_space(c, input_size)

    genome = Genome(n_genes=len(gs) * c.n_nodes, genome_space=gs)
    print(genome)

    # Test valid program
    genome = Genome(genes=test.SMALL_GENE_1_OUTPUT, genome_space=gs)
    c.n_nodes = int(test.SMALL_GENE_1_OUTPUT.shape[0] / len(gs))
    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c)
    res = prog.evaluate(test.SMALl_INPUT)
    s = prog.to_string()
    print(s)
    s = prog.to_string(test.SMALl_INPUT)
    print(s)
