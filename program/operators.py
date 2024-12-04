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
    def __init__(self, name: str, n_operands: int, function: Callable, print: Callable) -> None:
        self.name = name
        self.n_operands = n_operands
        self.function = function
        self.print = print
        self.operands = []

    # Dunder for operator name
    def __str__(self) -> str:
        return self.name

    # Get string representation of operator with operands
    # Dunder for operator name
    def get_string(self, operands: Union[list[str]]) -> str:
        return f'{self.print(operands)}'


    # Dunder for callable
    def __call__(self, input) -> float:
        return self.function(input)

    # Build the lambda
    def build(self, input) -> tuple[Callable, np.ndarray[float]]:
        return self.function, input


# Abstraction for input variable ot the program
class InputVar:
    def __init__(self, index):
        self.index = index

    # Lambda that can access the given observation at any index
    def __call__(self, input) -> callable:
        return input[self.index]

    # Dunder for input variable index
    def __str__(self, input: Union[None, np.ndarray[float]] = None) -> str:
        if input is not None:
            return f'[{input}]'
        else:
            return f'input_{self.index}'


old_SIMPLE_OPERATORS = [
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
SIMPLE_OPERATORS = [
    Operator('+', 2, lambda x: x[0] + x[1],
             lambda x: f'{x[0]} + {x[1]}'),
    Operator('-', 2, lambda x: x[0] - x[1],
             lambda x: f'{x[0]} - {x[1]}'),
    Operator('%', 2, lambda x: x[0] % x[1],
             lambda x: f'{x[0]} % {x[1]}'),
    Operator('*', 2, lambda x: x[0] * x[1],
             lambda x: f'{x[0]} * {x[1]}'),
    Operator('/', 2, lambda x: 999 if -0.01 > x[1] > 0.01 else x[0] / x[1],
             lambda x: f'{x[0]} / {x[1]}'),
    Operator('||', 1, lambda x: abs(x[0]),
             lambda x: f'|{x}|'),
    Operator('exp', 1, lambda x: math.exp(x[0]),
             lambda x: f'exp({x[0]})'),
    Operator('sin', 1, lambda x: math.sin(x[0]),
             lambda x: f'sin({x[0]})'),
    Operator('cos', 1, lambda x: math.cos(x[0]),
             lambda x: f'cos({x[0]})'),
    Operator('log', 1, lambda x: -999 if x[0] < 0.01 else math.log(x[0]),
             lambda x: f'log({x[0]})'),
    Operator('neg', 1, lambda x: -x[0],
             lambda x: f'neg({x[0]})'),
    Operator('id', 1, lambda x: x[0],
             lambda x: f'id({x[0]})'),
    Operator('>', 4, lambda x: x[2] if x[0] > x[1] else x[3],
             lambda x: f'{x[2]} if {x[0]} > {x[1]} else {x[3]}'),
    Operator('<', 4, lambda x: x[2] if x[0] < x[1] else x[3],
             lambda x: f'{x[2]} if {x[0]} < {x[1]} else {x[3]}'),
]

OPERATOR_SETS = [SIMPLE_OPERATORS]

if __name__ == '__main__':
    i = InputVar(0)
    for operator_set in OPERATOR_SETS:
        print(f"Set: {operator_set}")
        for operator in operator_set:
            print(f" Operator: {operator}")
        print("--------------")
