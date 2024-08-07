from postfix_program import Program

dna = [3.37582996, -2.056983, -23.2837208, 0.95573015, -5.83069119, -2.03368705, -11.53087575, -23.46732749 , -2.53473184, -21.91881229]

program = Program(dna)

import gymnasium as gym
import numpy as np

terminated, truncated = False, False

env = gym.make('InvertedPendulum-v4')

obs, _ = env.reset()

while not terminated or truncated:

    action = np.array([program.run_program(inp=obs)])
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(reward)

