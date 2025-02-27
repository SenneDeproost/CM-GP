# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import sys
from pprint import pprint

import yaml

from src.cmgp.buffer import EpisodicReplayBuffer

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
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


def get_state_actions(programs, obs, env):
    program_actions = []

    for i, o in enumerate(obs):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        for action_index in range(env.action_space.shape[0]):
            program = programs[action_index]
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
    rb = ReplayBuffer(
        args.training.agent.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    buff = EpisodicReplayBuffer(size=100)
    tmp = []

    critic_config = CriticConfig()
    critic = Critic(env, critic_config)

    # Actor is a learnable program for each action in the action space
    program_optimizers = [PyGADOptimizer(
        args.training.optimizer,
        SIMPLE_OPERATORS_DICT,  # Todo: change!
        observation_space=env.observation_space,
        action_space=env.action_space,
        critic=critic,
        buffer=rb,
        env=env,
        buffer_batch_size=args.training.agent.actor_batch_size) for _ in range(n_actions)]

    #for action_index in range(env.action_space.shape[0]):
    #    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")


    best_programs = [optimizer.best_program for optimizer in program_optimizers]


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
                action = get_state_actions(best_programs, obs[None, :], env)[0]
                action += np.random.normal(loc=action, scale=args.training.agent.exploration_noise) # !!!!!!
                action = action.clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        n_environment_interactions += 1

        #print(f'Action: {action}')
        #print(f'Reward: {reward}')

        tmp.append(obs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("policy/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("policy/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("policy/env_interactions", n_environment_interactions, global_step)
            writer.add_scalar("policy/reward", reward, global_step)
            writer.add_text('policy/action', str(action), global_step)
            buff.add(tmp)
            tmp = []
            # Prevent best program to change during episode
            #best_programs = [optimizer.best_program for optimizer in program_optimizers]

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, action, reward, termination, info)

        # RESET
        if termination or truncation:
            next_obs, _ = env.reset()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.training.start_learning:
            data = rb.sample(args.training.agent.critic_batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(
                    data.actions, device=device) * args.training.agent.policy_noise).clamp(
                    -args.training.agent.noise_clip, args.training.agent.noise_clip
                )

                #clipped_noise = 0 # !!!!!!

                # Go over all observations the buffer provides
                next_state_actions = get_state_actions(best_programs,
                                                       data.next_observations.detach().numpy(), env)
                next_state_actions = torch.tensor(next_state_actions)
                next_state_actions = (next_state_actions + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]).float()

            critic_loss, q_values = critic.learn_values(data, next_state_actions)
            #print(f'Critic prediction loss {critic_loss}')

            # Optimize the program
            if global_step % args.training.policy_update == 0:

                # New sampling
                #data = rb.sample(args.training.agent.actor_batch_size) # Was a mistake

                program_actions = get_state_actions(best_programs,
                                                    data.observations.detach().numpy(), env)

                cur_program_actions = np.copy(program_actions)
                #print('BEFORE ACTIONS')
                #pprint(program_actions[0:4])

                #improved_actions, improved_action_deltas = (
                #    critic.improve_actions(cur_program_actions, data.observations.detach().numpy()))

                #print('IMPROVED ACTIONS')
                #pprint(improved_actions[0:4])

                # Fit the program optimizers on all the action dimensions
                states = data.observations.detach().numpy()
                #actions = improved_actions

                print('Best program:')

                for action_index in range(n_actions):
                    optimizer = program_optimizers[action_index]

                    #improved_actions = improved_actions[:, action_index].reshape(-1, 1)
                    max_fit, min_fit, mean_fit = optimizer.fit()
                    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")
                    program_optimizers[action_index] = optimizer
                    best_programs = [optimizer.best_program for optimizer in program_optimizers]

                    # Optimizer
                    if global_step % 10 == 0:
                        writer.add_scalar(f'optimizer/action_{action_index}/min_fit', min_fit, global_step)
                        writer.add_scalar(f'optimizer/action_{action_index}/max_fit', max_fit, global_step)
                        writer.add_scalar(f'optimizer/action_{action_index}/mean_fit', mean_fit, global_step)
                        writer.add_text(f'optimizer/best_program/action_{action_index}',
                                        str(optimizer.best_program), global_step)

                # update the target network
                critic.update_target()

                #next_obs, _ = env.reset()

            # Logging
            if global_step % 10 == 0:
                # Critic
                writer.add_scalar('critic/loss', critic_loss, global_step)
                #writer.add_scalar('critic/mean_delta', improved_action_deltas.mean(), global_step)
                writer.add_scalar('critic/mean_q_value', q_values.mean(), global_step)
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
