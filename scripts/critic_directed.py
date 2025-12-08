import gymnasium as gym
import numpy as np
import math

env = gym.make('InvertedPendulum-v4', render_mode='human')


def log(x) -> float:
    return -999 if x < 0.01 else math.log(x)


p = lambda x: 20*x[1] + 2*x[3]
p = lambda x: x[3] - x[2]
p = lambda x: x[3]*x[1] - math.sin(x[2]) - x[3]*x[1] - x[1] + x[3]
p = lambda x: 2*x[1] - 2*x[0] - math.cos(log(x[1]) - x[1])
p = lambda x: 4*x[1] + 2*x[2] + 2*x[3]
p = lambda x: x[3]*4*x[1] + x[3]


obs, _ = env.reset()
terminated, truncated = False, False
while True:
    act = p(obs)
    act = np.array([act])
    print(obs)
    next_obs, reward, terminated, truncated, info = env.step(act)
    print(reward)
    env.render()

    obs = next_obs

    if terminated or truncated:
        break