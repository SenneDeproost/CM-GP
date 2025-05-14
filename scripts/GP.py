import sys

#from scripts.cartpole import terminated

sys.path.append('../src/cmgp/')
sys.path.append('../')

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

from optimizer import PyGADOptimizer, print_fitness
from config import ExperimentConfig, CriticConfig, OptimizerConfig

import envs
from program import SIMPLE_OPERATORS_DICT
from critic import Critic
import yaml


class CustomOptimizer(PyGADOptimizer):

    def __init__(self, config: OptimizerConfig, operators, space):
        super().__init__(config, operators, space)
        self.env = None

    # New fit function
    def fitness_function_(self, _, solution, solution_index) -> float:

        env = copy(self.env)

        fitness = 0.0

        prog = self.population.realize(solution_index)

        obs, _ = env.reset()

        terminated, truncated = False, False

        while not terminated or not truncated:

            action = prog(obs)
            next_obs, reward, terminated, truncated, info = env.step([action])

            fitness += reward
            obs = next_obs
            self.interactions += 1

            if terminated or truncated:
                break

        return fitness

    def fit_old(self) -> None:

        self._optim.initial_population = self.population.individuals  #self.population.raw_genes()

        # Iterate with optimizer
        self._optim = self._init_optimizer()
        #self.reset_solutions()
        self._optim.run()

        #self.raw_population = self._optim.population
        self.population.individuals = self._optim.population

        # Set best results
        best_sol, best_fit, best_idx = self._optim.best_solution(
            pop_fitness=self._optim.last_generation_fitness)  # Recalculates using new interaction
        self.best_solution_index = best_idx
        self.best_fitness = best_fit
        self.best_program = self.population.realize(self.best_solution_index)
        print(f'N_interactions: {self.interactions}')

        print(f'Optimizer says: best program is {self.best_program}')
        print(f'Optimizer says: best fitness is {self.best_fitness}')

    # Run n_generations with the optimizer
    def run_optimizer(self):
        # Run for each generation the whole evolutionary loop.
        for gen in range(self.config.n_generations - 1):
            # If replay buffer is given, sample new experience in each generation
            #if self.buffer is not None:
            #    self.new_sample()

            # Reinit test
            self._init_optimizer()
            self._optim.initial_population = self.population.individuals
            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # The genetic operations on the population
            self._optim.run_select_parents()
            self._optim.run_crossover()
            self._optim.run_mutation()
            self._optim.run_update_population()

            # Calc fitness function
            self._optim.previous_generation_fitness = self._optim.last_generation_fitness.copy()
            self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

            # Print
            self.on_generation(self._optim)

            # Update population with population of optimizer
            self.population.individuals = self._optim.population

    def fit(self, critic_states=None, critic_actions=None) -> (float, float, float):

        # Iterate with optimizer
        self._init_optimizer()
        self._optim.initial_population = self.population.individuals
        self._critic_states = critic_states
        self._critic_actions = critic_actions

        # Sample from buffer if given
        if self.buffer is not None:
            self.new_sample()

        # Calculate initial fitness
        self._optim.last_generation_fitness = self._optim.cal_pop_fitness()

        # Run the optimizer
        self.run_optimizer()

        # Candidate is best from optimizer
        candidate_solution, candidate_fitness, candidate_index = self._optim.best_solution(
            pop_fitness=self._optim.last_generation_fitness)

        candidate_program = self.population.realize(self.best_index)
        candidate_score = self.run_direct_validation(candidate_solution)
        best_program_score = self.run_direct_validation(self.best_solution)

        # Test if candidate performs better than current best in direct interaction

        # Print
        print(f'Candidate is {candidate_program} with fitness {candidate_fitness}')
        #print(f'Best program is {self.best_program} with score {best_program_score}')
        print(f'Best candidate is {candidate_program} with score {candidate_score}')

        #if candidate_score > best_program_score:
        self.best_program = candidate_program
        #print(f"New best program: {self.best_program}")
        #self.population.individuals = np.tile(candidate_solution, np.array([self.config.n_individuals, 1])) # !!!

        return (self._optim.last_generation_fitness.max(),
                self._optim.last_generation_fitness.min(),
                self._optim.last_generation_fitness.mean())


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


def main(config: ExperimentConfig):
    args = config

    run_name = f"{args.env_id}__{args.log.run_name}__{args.seed}__{int(time.time())}"
    if args.log.wandb.track:
        import wandb

        wandb.init(
            project=args.log.wandb.project,
            entity=args.log.wandb.entity,
            sync_tensorboard=True,
            config=vars(args),
            mode='online',
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

    env.observation_space.dtype = np.float32

    start_time = time.time()

    obs, _ = env.reset(seed=args.seed)

    for global_step in range(1000):

        env_interactions = 0

        for optimizer in program_optimizers:
            optimizer.fit()
            p = optimizer.best_program
            env_interactions += optimizer.interactions


        # Validation
        obs, _ = env.reset(seed=args.seed)
        l = 0
        r = 0

        terminated, truncated, = False, False
        while not terminated or truncated:
            l += 1
            #print(f'OBS: {obs}')
            action = get_state_actions(program_optimizers, [obs], env, args)
            obs, reward, terminated, truncated, info = env.step(action[0])
            r += reward
            if terminated or truncated:
                break
        writer.add_scalar("policy/episodic_return", r, global_step)
        writer.add_scalar("policy/episodic_length", l, global_step)
        writer.add_scalar("policy/env_interactions", env_interactions, env_interactions)

    env.close()
    writer.close()


if __name__ == "__main__":
    #exit()
    #if args.config_file is not None:
    #    c = ExperimentConfig
    #    with open(args.config_file, 'r') as f:
    #        data = yaml.load(f, Loader=yaml.SafeLoader)
    #        pyrallis.load(c, f)

    c = ExperimentConfig()
    #c.env_id = 'InvertedPendulum-v4'
    #c.training.optimizer.n_generations = 50

    #args = tyro.cli(c)

    main(c)
