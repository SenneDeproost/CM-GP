# +++ CM-GP/critic +++
#
# Critic value network abstraction
#
# 16/12/2024 - Senne Deproost

import numpy as np


# Critic value network, containing components for gradient calculation.
class Critic:

    def __init__(self, state_size, action_size):
        pass

    def improve_action(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, state: np.ndarray) -> np.ndarray:
        pass