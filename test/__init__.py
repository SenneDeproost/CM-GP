import unittest
import gymnasium as gym
import numpy as np

SMALL_OBS_SPACE = gym.spaces.Box(low=-20, high=20, shape=(4,))
SMALl_INPUT = np.array([-9, 99, 999, 9999])

## Programs

# id(input_0) + 0.5
SMALL_GENE_1_OUTPUT = np.array([-14, 0.0, 1, 0.69991735,  # [ input_0 | False | [1, 1] ]
                                -11, 0.0, 0, 1.37312973,  # [ id      | False | [0, 1] ]
                                5, 0.0, 0, 0,             # [ 5.0     | False | [0, 0] ]
                                0, 1.0, 1, 2.34511764     # [ +       | True  | [1, 2] ]
                                ])

# ∑[ (id(input_0) + 5.0) , id(input_0) ]
SMALL_GENE_2_OUTPUT = np.array([-14, 0.0, 1, 0.69991735,  # [ input_0 | False | [1, 1] ]
                                -11, 1.0, 0, 1.37312973,  # [ id      | True | [0, 1] ]
                                5, 0.0, 0, 0,             # [ 5.0     | False | [0, 0] ]
                                0, 1.0, 1, 2.34511764     # [ +       | True  | [1, 2] ]
                                ])

# (id(input_0) + 5.0) * input_1
BIG_GENE_1_OUTPUT = np.array([-14, 0.0, 1, 0.69991735,    # [ input_0 | False | [1, 1] ]
                              -11, 0.0, 0, 1.37312973,    # [ id      | False | [0, 1] ]
                              0, 0.0, 1, 3.34511764,      # [ +       | False | [1, 3] ]
                              5, 0.0, 0, 0,               # [ 5.0     | False | [0, 0] ]
                              -3, 1.0, 2.111, 5.364,      # [ *       | True  | [2, 5] ]
                              -15, 0.0, 0, 0,             # [ input_1 | False | [0, 0] ]
                              ])

# ∑[ (id(input_0) + 5.0) * input_1, (id(input_0) + 5.0) ]
BIG_GENE_2_OUTPUT = np.array([-14, 0.0, 1, 0.69991735,    # [ input_0 | False | [1, 1] ]
                              -11, 0.0, 0, 1.37312973,    # [ id      | False | [0, 1] ]
                              0, 1.0, 1, 3.34511764,      # [ +       | True | [1, 3] ]
                              5, 0.0, 0, 0,               # [ 5.0     | False | [0, 0] ]
                              -3, 1.0, 2.111, 5.364,      # [ *       | True  | [2, 5] ]
                              -15, 0.0, 0, 0,             # [ input_1 | False | [0, 0] ]
                              ])

if __name__ == '__main__':
    print(f"Small gene: \n{SMALL_GENE_1_OUTPUT}")
