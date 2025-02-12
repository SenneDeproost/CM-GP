# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import sys
from pprint import pprint

import yaml

sys.path.append('../src/cmgp/')
sys.path.append('../')
import os
import random
import time
from dataclasses import dataclass
import pyrallis

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from stable_baselines3.common.buffers import ReplayBuffer
from cleanrl.cleanrl_utils.buffers import PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

from optimizer import PyGADOptimizer
from config import ExperimentConfig, CriticConfig

import envs
from program import SIMPLE_OPERATORS_DICT
from critic import Critic


def make_env(env_id, seed, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
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
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
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


def get_state_actions(program_optimizers, obs, env, args):
    program_actions = []

    for i, o in enumerate(obs):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        for action_index in range(env.action_space.shape[0]):
            program = program_optimizers[action_index].best_program
            action[action_index] = program(o)

        program_actions.append(action)

    return np.array(program_actions)


def main(config: ExperimentConfig):
    args = config

    run_name = f"{args.env_id}__{args.log.run_name}__{args.seed}__{int(time.time())}"
    if args.log.wandb.track:
        import wandb

        wandb.init(
            project=args.log.wandb.project,
            entity=args.log.wandb.entity,
            group=args.log.wandb.group,
            mode='online',
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, args.seed, 0, args.log.video, run_name)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    n_actions = env.action_space.shape[0]

    env.observation_space.dtype = np.float32
    rb_old = ReplayBuffer(
        args.training.agent.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    rb = PrioritizedReplayBuffer(
        buffer_size=args.training.agent.buffer_size,
        alpha=1,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=1
    )

    critic_config = CriticConfig()
    critic_1 = Critic(env, critic_config)
    critic_2 = Critic(env, critic_config)
    critic_optimizer = optim.Adam(list(critic_1.model.parameters()) + list(critic_2.model.parameters()),
                                  lr=args.training.critic.learning_rate)

    actor = Actor(env).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.training.agent.learning_rate)
    actor_target = Actor(env).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # Actor is a learnable program for each action in the action space
    program_optimizers = [PyGADOptimizer(
        args.training.optimizer,
        SIMPLE_OPERATORS_DICT,  # Todo: change!
        env.observation_space,
        critic=critic_1,
        buffer=rb,
        action_space=env.action_space,
        buffer_batch_size=args.training.agent.actor_batch_size) for _ in range(n_actions)]

    #for action_index in range(env.action_space.shape[0]):
    #    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    n_environment_interactions = 0

    for global_step in range(args.training.timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < args.training.start_learning:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                #action_program = get_state_actions(program_optimizers, obs[None, :], env, args)[0]
                action = actor(torch.Tensor(obs).to(device))
                #action += np.random.normal(loc=action, scale=args.training.agent.exploration_noise)
                action += torch.normal(0, actor.action_scale * args.training.agent.exploration_noise)
                #action = action.clip(env.action_space.low, env.action_space.high)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        n_environment_interactions += 1

        #print(f'Action: {action}')
        #print(f'Reward: {reward}')

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("policy/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("policy/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("policy/env_interactions", n_environment_interactions, global_step)
            writer.add_scalar("policy/reward", reward, global_step)
            writer.add_text('policy/action', str(action), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, action, reward, termination)

        # RESET
        if termination or truncation:
            next_obs, _ = env.reset()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.training.start_learning:
            data = rb.sample(args.training.agent.critic_batch_size, beta=0.5)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(
                    data.actions, device=device) * args.training.agent.policy_noise).clamp(
                    -args.training.agent.noise_clip, args.training.agent.noise_clip
                ) * actor_target.action_scale

                # Go over all observations the buffer provides
                #next_state_actions = get_state_actions(program_optimizers,
                #                                       data.next_observations.detach().numpy(), env, args)
                next_state_actions = (actor_target(data.next_observations) + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]
                )
                next_state_actions = torch.tensor(next_state_actions)
                next_state_actions = (next_state_actions + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]).float()

            q_1_target_values = critic_1.target_model(data.next_observations, next_state_actions)
            q_2_target_values = critic_2.target_model(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(q_1_target_values, q_2_target_values)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.training.critic.gamma * (
                min_qf_next_target).view(-1)

            qf1_a_values = critic_1.model(data.observations, data.actions).view(-1)
            qf2_a_values = critic_2.model(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            critic_optimizer.zero_grad()
            qf_loss.backward()
            critic_optimizer.step()

            # Optimize the actor
            if global_step % args.training.policy_update == 0:

                actor_loss = -critic_1.model(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(
                        args.training.agent.tau * param.data + (1 - args.training.agent.tau) * target_param.data)

                critic_1.update_target()
                critic_2.update_target()

                # Optimize the program
            if global_step % 1000 == 0:
                # New sampling
                #data = rb.sample(args.training.agent.actor_batch_size) # Was a mistake

                #program_actions = get_state_actions(program_optimizers,
                #                                    data.observations.detach().numpy(), env, args)

                #cur_program_actions = np.copy(program_actions)
                #print('BEFORE ACTIONS')
                #pprint(program_actions[0:4])

                # Replay buffer already given to the critic abstraction

                #improved_actions, improved_action_deltas = (
                #    critic_1.improve_actions(program_actions, data.observations.detach().numpy()))

                best_programs = []

                for action_index in range(n_actions):
                    optimizer = program_optimizers[action_index]
                    optimizer.env = env

                    #improved_actions = improved_actions[:, action_index].reshape(-1, 1)
                    max_fit, min_fit, mean_fit = optimizer.fit()
                    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")
                    program_optimizers[action_index] = optimizer
                    best_programs.append(optimizer.best_program)

                # Do validation episode
                obs, _ = env.reset()
                truncation, termination = False, False
                r = 0

                while not termination or not truncation:
                    #action = get_state_actions(program_optimizers, [obs], env, args)[0]
                    prog = best_programs[0]
                    #action = np.array([prog(state) for state in obs]).reshape((-1, 1)) # Needs to be chenged for other environments !
                    #print(action)
                    action = np.array([prog(obs)])
                    action = action.clip(env.action_space.low, env.action_space.high)
                    obs, reward, termination, truncation, info = env.step(action)
                    r += reward
                    if termination or truncation:
                        env.reset()
                        break

                print(f'Validation episodic return = {r}')

                #print('IMPROVED ACTIONS')
                #pprint(improved_actions[0:4])

                # Fit the program optimizers on all the action dimensions
                #states = data.observations.detach().numpy()
                #actions = improved_actions

                #print('Best program:')

                """for action_index in range(n_actions):
                    optimizer = program_optimizers[action_index]

                    #improved_actions = improved_actions[:, action_index].reshape(-1, 1)
                    max_fit, min_fit, mean_fit = optimizer.fit()
                    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")
                    program_optimizers[action_index] = optimizer

                    # Optimizer
                    if global_step % 10 == 0:
                        writer.add_scalar(f'optimizer/action_{action_index}/min_fit', min_fit, global_step)
                        writer.add_scalar(f'optimizer/action_{action_index}/max_fit', max_fit, global_step)
                        writer.add_scalar(f'optimizer/action_{action_index}/mean_fit', mean_fit, global_step)
                        writer.add_text(f'optimizer/best_program/action_{action_index}',
                                        str(optimizer.best_program), global_step)
                """

                # update the target network
                #critic.update_target()

            # Logging
            if global_step % 10 == 0:
                # Critic
                #writer.add_scalar('critic/loss', critic_loss, global_step)
                #writer.add_scalar('critic/mean_delta', improved_action_deltas.mean(), global_step)
                #writer.add_scalar('critic/mean_q_value', q_values.mean(), global_step)
                # Experiment
                writer.add_scalar('experiment/time', time.time() - start_time, global_step)
                writer.add_scalar('experiment/time_per_step', global_step / (time.time() - start_time), global_step)

    env.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)

    if args.config_file is not None:
        c = ExperimentConfig
        with open(args.config_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            pyrallis.load(c, f)

    main(args)
