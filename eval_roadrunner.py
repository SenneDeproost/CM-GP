import matplotlib.pyplot as plt
import gymnasium as gym
from math import sin
import envs


env = gym.make('RoadrunnerContinuous-v0')


p = [0]
s = [0]


x = [0, 0]
## min(ACCELERATE_↑↑↑, (ACCELERATE_↑↑ + min(ACCELERATE_↑, ((DECELERATE_↓ * x[1]) * x[0]))))
f = lambda x: min(1, (0.5 + min(0.1, ((0.1 * x[1]) * x[0]))))


## (ACCELERATE_↑↑↑ + min((DECELERATE_↓ * ((x[0] if (x[1] * x[0]) > x[1] else x[1]) * x[0])), x[1]))
#f = lambda x: (1 + min((-0.1 * ((x[0] if (x[1] * x[0]) > x[1] else x[1]) * x[0])), x[1]))

## ((ACCELERATE_↑↑↑ * ACCELERATE_↑↑↑) + (min(ACCELERATE_↑, ±2.873033616424472) * (x[1] * x[0])))
f = lambda x: (1 + (0.1 * (x[1] * x[0])))

obs, info = env.reset()

p = [obs[0]]
s = [obs[1]]

while True:

    a = f(obs)
    print(f'Action: {a}')
    obs, reward, terminated, truncated, info = env.step([a])
    p.append(obs[0])
    s.append(obs[1])
    print(f'Next: {obs}, reward: {reward}')
    if terminated:
        print('done')
        break

plt.ylabel('Speed')
plt.xlabel('Position')
plt.plot(p, s)
plt.show()
print('ok')
