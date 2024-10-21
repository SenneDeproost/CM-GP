from simple_envs import SimpleGoalEnvSpeed

env = SimpleGoalEnvSpeed()
env.reset()
env.step((0, 0))
print('done')