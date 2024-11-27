# +++ CM-GP/program/operators +++
#
# Sets of domain-specific operators
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

from typing import List, Callable, Union
import math
import numpy as np

# Simple operator for Python function
class Operator:
    def __init__(self, name: str, n_operands: int, function: Callable) -> None:
        self.name = name
        self.n_operands = n_operands
        self.function = function

    # Dunder for operator name
    def __str__(self) -> str:
        return self.name

SIMPLE_OPERATORS = [
    Operator(name='+', n_operands=2, function=lambda a, b: a + b),
]

GONIOMETRIC_OPERATORS = [
    Operator(name='sin', n_operands=1, function=lambda a: math.sin(a)),
]

OPERATOR_SETS = [SIMPLE_OPERATORS, GONIOMETRIC_OPERATORS]

if __name__ == '__main__':
    for operator_set in OPERATOR_SETS:
        print(f"Set: {operator_set}")
        for operator in operator_set:
            print(f" Operator: {operator}")
        print("--------------")