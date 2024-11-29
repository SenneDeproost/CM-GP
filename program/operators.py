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
    def __init__(self, name: str, n_operands: int, function: Callable, print=Callable) -> None:
        self.name = name
        self.n_operands = n_operands
        self.function = function
        self.print = print

    # Dunder for operator name
    def __str__(self) -> str:
        return self.name

# Abstraction for input variable ot the program
class InputVar:
    def __init__(self, index):
        self.index = index

    # Lambda that can access the given observation at any index
    def __call__(self) -> callable:
        return lambda input: input[self.index]

    # Dunder for input variable index
    def __str__(self) -> str:
        return f'input_{self.index}'

SIMPLE_OPERATORS = [
    Operator('+', 2, lambda a, b: a + b,
             lambda a, b: f'{a} + {b}'),
    Operator('-', 2, lambda a, b: a - b,
             lambda a, b: f'{a} - {b}'),
    Operator('%', 2, lambda a, b: a % b,
             lambda a, b: f'{a} % {b}'),
    Operator('*', 2, lambda a, b: a * b,
             lambda a, b: f'{a} * {b}'),
    Operator('/', 2, lambda a, b: 999 if -0.01 > b > 0.01 else a / b,
             lambda a, b: f'{a} / {b}'),
    Operator('||', 1, lambda a: abs(a),
             lambda a: f'|{a}|'),
    Operator('exp', 1, lambda a: math.exp(a),
             lambda a: f'exp({a})'),
    Operator('sin', 1, lambda a: math.sin(a),
             lambda a: f'sin({a})'),
    Operator('cos', 1, lambda a: math.cos(a),
             lambda a: f'cos({a})'),
    Operator('log', 1, lambda a: -999 if a < 0.01 else math.log(a),
             lambda a: f'log({a})'),
    Operator('neg', 1, lambda a: -a,
             lambda a: f'neg({a})'),
    Operator('id', 1, lambda a: a,
             lambda a: f'id({a})'),
    Operator('>', 4, lambda a, b, _true, _false: _true if a > b else _false,
             lambda a, b, _true, _false: f'{_true} if {a} > {b} else {_false}'),
    Operator('<', 4, lambda a, b, _true, _false: _true if a < b else _false,
             lambda a, b, _true, _false: f'{_true} if {a} < {b} else {_false}'),
]

OPERATOR_SETS = [SIMPLE_OPERATORS]

if __name__ == '__main__':
    for operator_set in OPERATOR_SETS:
        print(f"Set: {operator_set}")
        for operator in operator_set:
            print(f" Operator: {operator}")
        print("--------------")
