# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import sys

import yaml

sys.path.append('../src/cmgp/')
sys.path.append('../')
import pyrallis

from config import ExperimentConfig

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from critic import Critic
from config import CriticConfig

import envs
from config import ExperimentConfig


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

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf2 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf2_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.training.agent.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.training.agent.learning_rate)

    critic_config = CriticConfig()
    critic = Critic(env, critic_config)

    n_actions = env.action_space.shape[0]

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

    n_environment_interactions = 0

    for global_step in range(args.training.timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.training.start_learning:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(torch.Tensor(obs).to(device))
                action += torch.normal(0, actor.action_scale * args.training.agent.exploration_noise)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        n_environment_interactions += 1

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
                clipped_noise = (torch.randn_like(data.actions,
                                                  device=device) * args.training.agent.policy_noise).clamp(
                    -args.training.agent.noise_clip, args.training.agent.noise_clip
                ) * target_actor.action_scale

                next_observations = data.next_observations

                next_state_actions = (target_actor(next_observations) + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]
                )
                qf1_next_target = qf1_target(next_observations, next_state_actions)
                qf2_next_target = qf2_target(next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.training.critic.gamma * (
                    min_qf_next_target).view(-1)

            observations = data.observations

            qf1_a_values = qf1(observations, data.actions).view(-1)
            qf2_a_values = qf2(observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.training.policy_update == 0:
                actor_loss = -qf1(observations, actor(observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(
                        args.training.agent.tau * param.data + (1 - args.training.agent.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.training.critic.tau * param.data + (1 - args.training.critic.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.training.critic.tau * param.data + (1 - args.training.critic.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                #writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
