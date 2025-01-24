# +++ CM-GP/program +++
#
# Abstractions for realization
#
# 27/11/2024 - Senne Deproost & Denis Steckelmacher

from .operators import *


# Abstraction for input variable ot the program
class InputVar:
    def __init__(self, index: int) -> None:
        self.index = index

    # Lambda that can access the given observation at any index
    def __call__(self, input: np.ndarray[float]) -> float:
        return input[self.index]

    # Dunder for input variable index
    def __str__(self) -> str:
        return f'input_{self.index}'

    # Get string representation with inputs
    def to_string(self, input: np.ndarray[float]) -> str:
        if input is None:
            return self.__str__()
        else:
            return f'{input[self.index]}'


# Lookup dictionary on the amount of operands

# Dictionary needs to be continuous. If no operator with a certain n_operands exists, add empty list.

# Simple operators

# Operator dict is used to index operators with the amount of operands they use. For population creations mostly.
SIMPLE_OPERATORS_DICT = {
    0: [],
    1: [
        Sin(),
        Cos(),
        Abs(),
        Exp(),
        #Log(),
        Neg(),
        Id()
    ],
    2: [
        Add(),
        Sub(),
        Mul(),
        Div(),
        Mod(),
    ],
    3: [],
    4: [
        Grt(),
        Sml()
    ]}

SIMPLE_OPERATORS = [x for y in SIMPLE_OPERATORS_DICT.values() for x in y]

# Simple functions (SIMPLE OPERATORS with max 2 operands)
SIMPLE_FUNCTIONS_DICT = SIMPLE_OPERATORS_DICT.copy()
del SIMPLE_FUNCTIONS_DICT[4]
del SIMPLE_FUNCTIONS_DICT[3]
SIMPLE_FUNCTIONS = [x for y in SIMPLE_FUNCTIONS_DICT.values() for x in y]
