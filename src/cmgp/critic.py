# +++ CM-GP/critic +++
#
# Critic value network abstraction
#
# 16/12/2024 - Senne Deproost
from copy import copy

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples

from config import CriticConfig, ExperimentConfig


# Source: https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


# Critic value network, containing components for gradient calculation.
class Critic:

    def __init__(self, env: gym.Env, config: CriticConfig) -> None:
        self.model = QNetwork(env)
        self.target_model = QNetwork(env)
        self.target_model.load_state_dict(self.model.state_dict())  # Ensure same parameters
        self.env = env
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    # For a given batch of actions, improve them using critic gradients
    def improve_actions(self, actions: np.ndarray, states: np.ndarray) -> (np.ndarray, float):

        actions = torch.tensor(actions, requires_grad=True)
        res_actions = copy(actions)
        og_actions = copy(actions)

        deltas = torch.zeros(size=actions.shape, requires_grad=False)
        states = torch.tensor(states, requires_grad=True)

        # Iterate several times over the actions to improve them
        for i in range(self.config.gradient_updates):

            # Reset gradients
            self.model.zero_grad()

            # Make predictions and retrieve gradients
            a = torch.tensor(res_actions, requires_grad=True)
            prediction = self.model(states, a)
            prediction.mean().backward()
            g = a.grad # Gradient of the action towards loss min, which is towards Q max

            # Stop if gradient threshold is met
            # Todo: check if abs is needed inner
            m = torch.abs(og_actions - res_actions).mean()
            if m > self.config.update_threshold:
                print(f'stopped grad update {i} at mean {m}')
                break

            #g = torch.tensor(g, requires_grad=True)
            delta = self.config.update_rate*g
            res_actions = res_actions + delta  # !!!! Minus
            deltas += delta.detach()
            #predictions.append(res_actions[1].detach().numpy()[0])


        #import matplotlib.pyplot as plt
        #y = np.array(predictions)
        #x = np.arange(0, len(predictions), 1)
        #plt.plot(x, y)
        #plt.show()

        res_actions = res_actions.to(dtype=torch.float64)

        #print(f'Delta gradients {deltas.detach().numpy().mean()}')
        return res_actions.detach().numpy(), deltas.detach().numpy()

    # Learn Q values
    def learn_values(self, data: ReplayBufferSamples, next_actions: torch.Tensor) -> (float, float):

        with torch.no_grad():
            q_next_target = self.target_model(data.next_observations, next_actions)
            next_q_value = (data.rewards.flatten() +
                            (1 - data.dones.flatten()) * self.config.gamma * (q_next_target).view(-1))

        q_a_values = self.model(data.observations, data.actions).view(-1)
        q_loss = F.mse_loss(q_a_values, next_q_value)

        # optimize the model
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        return q_loss.item(), q_a_values.detach().numpy()

    def q_to_action_gradients(self, gradients, rewards, states, actions, qs):
        pass



    # Update target network when updating policy
    def update_target(self) -> None:
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    import envs

    env = gym.make("SimpleGoal-v0")
    config = ExperimentConfig()
    critic = Critic(env, config.training.critic)

    rb = ReplayBuffer(
        3000,
        env.observation_space,
        env.action_space,
        'cpu',
        handle_timeout_termination=False,
    )

    obs, _ = env.reset(seed=config.seed)
    actions = []

    for episode in range(20):
        obs, _ = env.reset(seed=config.seed)
        termination, truncation = False, False
        while not termination:
            action = np.random.random((2))
            next_obs, reward, termination, truncation, info = env.step(action)
            rb.add(obs, next_obs, action, reward, termination, info)
            loss = critic.learn_values(rb)
            print(np.mean(np.array(loss)))

    state = np.array([[0.9, 0.9]])
    action = np.array([[0.9, 0.9]])
    r = critic.improve_actions(state, action)
    print(r)
