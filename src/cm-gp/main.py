 # docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
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
from config import ExperimentConfig

import envs
from program import SIMPLE_OPERATORS_DICT


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


def get_state_actions(program_optimizers, obs, env, args):
    program_actions = []

    for i, o in enumerate(obs):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        for action_index in range(env.action_space.shape[0]):
            program = program_optimizers[action_index].best_program
            action[action_index] = program(o)

        program_actions.append(action)

    return np.array(program_actions)

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)

    run_name = f"{args.env_id}__{args.log.run_name}__{args.seed}__{int(time.time())}"
    if args.log.wandb.track:
        import wandb

        wandb.init(
            project=args.log.wandb.project,
            entity=args.log.wandb.entity,
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

    # Actor is a learnable program for each action in the action space
    program_optimizers = [PyGADOptimizer(
        args.training.optimizer,
        SIMPLE_OPERATORS_DICT,
        env.observation_space,
    ) for i in range(env.action_space.shape[0])]

    for action_index in range(env.action_space.shape[0]):
        print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")

    qf1 = QNetwork(env).to(device)
    qf2 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf2_target = QNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.training.agent.learning_rate)

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.training.agent.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    for global_step in range(args.training.timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < args.training.start_learning:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = get_state_actions(program_optimizers, obs[None, :], env, args)[0]
                action = np.random.normal(loc=action, scale=args.training.agent.policy_noise)
                print('ACTION', action)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
            data = rb.sample(args.training.agent.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(
                    data.actions, device=device) * args.training.agent.policy_noise).clamp(
                    -args.training.agent.noise_clip, args.training.agent.noise_clip
                )

                # Go over all observations the buffer provides
                next_state_actions = get_state_actions(program_optimizers,
                                                       data.next_observations.detach().numpy(), env, args)
                next_state_actions = torch.tensor(next_state_actions)
                next_state_actions = (next_state_actions + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]).float()

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.training.agent.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            #print(f'Loss critic: {qf1_loss}')

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Optimize the program
            if global_step % args.training.policy_update == 0:
                orig_program_actions = get_state_actions(program_optimizers,
                                                         data.observations.detach().numpy(), env, args)
                cur_program_actions = np.copy(orig_program_actions)
                print('BEFORE ACTIONS', orig_program_actions[0])

                for i in range(500):
                    program_actions = torch.tensor(cur_program_actions, requires_grad=True)

                    program_objective_1 = qf1(data.observations, program_actions).mean()
                    program_objective_2 = qf2(data.observations, program_actions).mean()
                    program_objective = (program_objective_1 + program_objective_2) * 0.5
                    program_objective.backward()

                    with torch.no_grad():
                        cur_program_actions += program_actions.grad.numpy()

                    if np.abs(cur_program_actions - orig_program_actions).mean() > 0.5:
                        break

                # Fit the program optimizers on all the action dimensions
                states = data.observations.detach().numpy()
                actions = cur_program_actions

                print('Best program:')
                writer.add_scalar("losses/program_objective", program_objective.item(), global_step)

                for action_index in range(env.action_space.shape[0]):
                    program_optimizers[action_index].fit(states, actions[:, action_index])
                    print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")

            # update the target network
            tau = args.training.agent.tau
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 10 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()
