# +++ CM-GP/program/realization +++
#
# Executable programs as the realization of the genome
#
# 28/11/2024 - Senne Deproost
import os
import sys

from src.cmgp.config import OptimizerConfig

sys.path.insert(0, '../')
from copy import deepcopy, copy

import numpy as np
from typing import Callable, List, Union, Any, Tuple
import gymnasium as gym
from matplotlib.pyplot import connect

import test

from numpy import ndarray

from config import CartesianConfig
from population_weighted import *
from program import Operator, InputVar, SIMPLE_OPERATORS, SIMPLE_OPERATORS_DICT


# Encoding
# gene 0: function of the node
# gene 1 to max_arity -> determined the arity of operator in set with the highest number of operands

# Node as part of the Cartesian graph
class Node:
    def __init__(self,
                 config: CartesianConfig,
                 function_dist: ndarray[float],
                 connection_dist: List[ndarray[float]]):
        self.function_dist, self.connection_dist = function_dist, connection_dist
        self.config = config

    def __str__(self) -> str:
        return f' [ {self.function} | {self.output} | {self.connections} ]'

    # Helper function for traversal
    def traverse(self, input: Union[ndarray[float], None],
                 on_operator: Callable, on_inputvar: Callable, on_float: Callable):

        # Connections
        connections = []

        # Determine connections greedily
        for connection in self.connection_dist:
            connections.append(np.argmax(connection))

        # Functions
        # Go over each possible function (inputs + operators) and traverse using the given weight
        for i, f in self.function_dist:
            # Part of the distribution that is input variables
            if i < self.config.n_inputs:
                on_inputvar(self, index=i, input=input)
            else:
                on_operator(self, index=i, input=input)


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


# Realize program from array
def realize_from_array(genes: np.array,
                       genome_space: List[GeneSpace],
                       config: CartesianConfig,
                       state_space: gym.Space,
                       operators: dict = SIMPLE_OPERATORS_DICT,
                       ) -> Program:
    genome = Genome(genes=genes, genome_space=genome_space)
    realization = CartesianProgram(
        genome=genome,
        input_space=state_space,
        operators=operators,
        config=config
    )
    return realization


def realize_subs_from_array(genes: np.array,
                            genome_space: List[GeneSpace],
                            config: CartesianConfig,
                            state_space: gym.Space,
                            operators: dict = SIMPLE_OPERATORS_DICT,
                            ) -> List[Program]:
    res = []
    for i in range(config.n_nodes):
        g = copy(genes)
        g[-config.n_outputs] = i  # Todo better solution for programs with multiple inputs
        genome = Genome(genes=g, genome_space=genome_space)
        realization = CartesianProgram(
            genome=genome,
            input_space=state_space,
            operators=operators,
            config=config
        )
        res.append(realization)
    return res


# Cartesian graph based program:
# Todo: List of operators can be removed since it is encapsulated in the genespace instance
class CartesianProgram(Program):
    def __init__(self, genome: Genome, input_space: gym.Space, operators: List[Operator], config: CartesianConfig):
        super().__init__(genome, input_space, deepcopy(operators))
        self.genome = genome
        self.n_operators = len(operators)
        self.config = config
        self.input_space = input_space
        self.n_inputs = input_space.shape[0]
        self._realization = self._process_nodes()  # Realization nesting of Nodes
        self._str = self.to_string()


    # Call dunder for easy execution
    def __call__(self, input: ndarray[float]) -> float:
        res = self.evaluate(input)
        return res

    # Dunder describing program
    # Todo: Correct printing of a program -> Maybe for the Program parent class?
    def __str__(self) -> str:
        return self._str

    # Process the nodes
    def _process_nodes(self):
        nodes = []
        roots = []
        node_len = node_length(self.config, self.n_inputs, self.operators)
        last_end = None
        values = self.genome.values

        start_index = 0

        # Nodes
        for i in range(self.config.n_nodes):
            connection_index = start_index + self.n_operators + self.n_inputs
            end_index = connection_index + self.config.n_nodes
            f_v = values [start_index:connection_index]
            c_v = values [connection_index:end_index]
            node = Node(self.config,
                        function_dist=f_v,
                        connection_dist=c_v)
            nodes.append(node)
            last_end = end_index
            start_index += node_len

        # Roots
        root_start = last_end + 1
        for i in range(self.config.n_outputs):
            root_end = root_start + self.config.n_nodes
            roots.append(values[root_start:root_end])
            root_start += node_len

        return nodes


    # Evaluate the realized function
    def evaluate(self, input: ndarray[float]) -> float:
        pass

    # Return string representation of program. When input is not given, placeholder names are used
    def to_string(self, input: Union[None, ndarray[float]] = None) -> list[Any] | Any:
        pass


if __name__ == "__main__":
    space = test.SMALL_OBS_SPACE
    input_size = space.shape[0]

    # Test for 4 nodes of 4 genes
    c = OptimizerConfig()
    c.program.n_nodes = 20

    gs = generate_cartesian_genome_space(c.program, input_size, SIMPLE_OPERATORS_DICT)
    pop = CartesianPopulation(config=c, operators_dict=SIMPLE_OPERATORS_DICT, state_space=space)

    values = pop.individuals[0]
    genome = Genome(genome_space=gs, values=values, pop_index=0)

    prog = CartesianProgram(genome, space, SIMPLE_OPERATORS, c.program)
    res = prog.evaluate(test.SMALl_INPUT)
    s = prog.to_string()
    print(s)
    s = prog.to_string(test.SMALl_INPUT)
    print(s)
