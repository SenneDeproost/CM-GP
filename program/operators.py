# +++ CM-GP/program/operators +++
#
# Sets of domain-specific operators
#
# 16/12/2024 - Senne Deproost

from typing import List, Callable, Union
import math
import numpy as np


# Check if brackets are needed (Bracket Check)
def bc(x: str):
    if isinstance(x, (list, tuple, np.ndarray)):
        if '(' in x:
            return f'({x})'
        else:
            return x
    else:
        return x


# Simple operator for Python function
class Operator:
    def __init__(self, name: str, n_operands: int) -> None:
        self.name = name
        self.n_operands = n_operands
        self.operands = []

    # Dunder for call, used to check given values
    def __call__(self, x: np.ndarray) -> None:
        assert len(x) == self.n_operands, "Incorrect of arguments given to function"

    # Dunder for operator name
    def __str__(self) -> str:
        return self.name


##### OPERATORS

class Add(Operator):

    def __init__(self) -> None:
        self.name = '+'
        self.n_operands = 2

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[0] + x[1]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{x[0]} + {x[1]}'


class Sub(Operator):

    def __init__(self) -> None:
        self.name = '-'
        self.n_operands = 2

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[0] - x[1]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{x[0]} - {x[1]}'


class Mod(Operator):

    def __init__(self) -> None:
        self.name = '%'
        self.n_operands = 2

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[0] % x[1]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{x[0]} % {x[1]}'


class Mul(Operator):

    def __init__(self) -> None:
        self.name = '*'
        self.n_operands = 2

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[0] * x[1]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{bc(x[0])} * {bc(x[1])}'


class Div(Operator):

    def __init__(self) -> None:
        self.name = '/'
        self.n_operands = 2

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return 999 if -0.01 > x[1] > 0.01 else x[0] / x[1]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{bc(x[0])} / {bc(x[1])}'


class Abs(Operator):

    def __init__(self) -> None:
        self.name = '||'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return abs(x[0])

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'|{x[0]}|'


class Exp(Operator):

    def __init__(self) -> None:
        self.name = 'exp'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return math.exp(99) if x[0] > 99 else math.exp(x[0])

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'exp({x[0]})'


class Sin(Operator):

    def __init__(self) -> None:
        self.name = 'sin'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return math.sin(x[0])

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'sin({x[0]})'


class Cos(Operator):

    def __init__(self) -> None:
        self.name = 'cos'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return math.cos(x[0])

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'cos({x[0]})'


class Log(Operator):

    def __init__(self) -> None:
        self.name = 'log'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return -999 if x[0] < 0.01 else math.log(x[0])

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'log({x[0]})'


class Neg(Operator):

    def __init__(self) -> None:
        self.name = 'neg'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return -x[0]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'neg({x[0]})'


class Id(Operator):

    def __init__(self) -> None:
        self.name = 'id'
        self.n_operands = 1

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[0]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'id({x[0]})'


class Grt(Operator):

    def __init__(self) -> None:
        self.name = '>'
        self.n_operands = 4

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[2] if x[0] > x[1] else x[3]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{x[2]} if {x[0]} > {x[1]} else {x[3]}'


class Sml(Operator):

    def __init__(self) -> None:
        self.name = '<'
        self.n_operands = 4

        super().__init__(self.name, self.n_operands)

    def __call__(self, x: np.ndarray[float]) -> float:
        return x[2] if x[0] < x[1] else x[3]

    @staticmethod
    def print(x: np.ndarray[float]) -> str:
        return f'{x[2]} if {x[0]} < {x[1]} else {x[3]}'
