# +++ CM-GP/buffer +++
#
# Replay/rollout buffers
#
# 23/11/2024 - Senne Deproost
from collections import deque
from random import randint
from typing import List


class EpisodicReplayBuffer:

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = []
        self.index = 0

    def add(self, episode) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append(episode)
        else:
            self.buffer[self.index] = episode
        self.index += 1
        if self.index == self.size - 1:
            self.index = 0

    def sample(self, batch_size: int) -> List:

        #return [self.buffer[self.index - 1]]

        res = []
        for _ in range(batch_size):
            i = randint(0, len(self.buffer) - 1)
            res.append(self.buffer[i])
        return res
