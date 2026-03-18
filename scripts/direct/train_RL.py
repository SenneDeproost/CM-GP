import sys
sys.path.append('../src/cmgp/')
sys.path.append('../')

from config import RLAgentConfig


class RLTraining:

    def __init__(self, config:RL):
        self.config = config