import random
import time
from copy import copy
from dataclasses import dataclass
import pyrallis

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import types

from stable_baselines3.common.buffers import ReplayBuffer
from tensorflow.compiler.tf2xla.python.xla import self_adjoint_eig
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

from optimizer import PyGADOptimizer
from config import ExperimentConfig, CriticConfig, OptimizerConfig

import envs
from program import SIMPLE_OPERATORS_DICT
from critic import Critic




class CustomOptimizer(PyGADOptimizer):

    def __init__(self, config: OptimizerConfig, operators, space):
        super().__init__(config, operators, space)
        self.env = None

    # New fit function
    def fitness_function(self, _, solution, solution_index) -> float:
        fitness = 0.0

        prog = self.population.realize(solution_index)

        obs, _ = self.env.reset()

        terminated, truncated = False, False

        while not terminated or not truncated:

            action = prog(obs)
            next_obs, reward, terminated, truncated, info = self.env.step([action])

            fitness += reward
            obs = next_obs

            if terminated or truncated:
                break
        return fitness

    def fit(self) -> None:

        self._optim.initial_population = self.raw_population  #self.population.raw_genes()

        # Iterate with optimizer
        #self._optim = self._init_optimizer()
        self.reset_solutions()
        self._optim.run()

        self.raw_population = self._optim.population
        self.population.update(self._optim.population)

        # Set best results
        best_sol, best_fit, best_idx = self._optim.best_solution()
        self.best_solution_index = best_idx
        self.best_fitness = best_fit
        self.best_program = self.population.realize(self.best_solution_index)


def make_env(env_id, seed, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


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
    program_optimizers = [CustomOptimizer(
        args.training.optimizer,
        SIMPLE_OPERATORS_DICT,  # Todo: change!
        env.observation_space,
    ) for i in range(env.action_space.shape[0])]

    for optimizer in program_optimizers:
        optimizer.env = copy(env)

    for action_index in range(env.action_space.shape[0]):
        print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")

    critic_config = CriticConfig()
    critic = Critic(env, critic_config)

    env.observation_space.dtype = np.float32

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    for global_step in range(args.training.timesteps):

        action = get_state_actions(program_optimizers, obs[None, :], env, args)[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)

        #print(f'Action: {action}')
        #print(f'Reward: {reward}')

        for action_index in range(env.action_space.shape[0]):
            optimizer = program_optimizers[action_index]
            optimizer.fit(obs, action)
            print(f"a[{action_index}] = {program_optimizers[action_index].best_program}")
            program_optimizers[action_index] = optimizer

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)



            if global_step % 10 == 0:
                pass
                #writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                ##writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                #writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                #writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                #writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                #writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()
