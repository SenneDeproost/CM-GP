import os
from dataclasses import dataclass, field
import pyrallis
from typing import Any, Dict, List, Union

import yaml


@dataclass
class WandbConfig:
    # Wandb track
    track: bool = field(default=False)
    # Wandb project name
    project: str = field(default='test')
    # Wandb tags
    tags: tuple = field(default=tuple())
    # Wandb entity name
    entity: str = field(default=None)

@dataclass
class HuggingFaceConfig:
    # Upload model to huggingface
    upload: bool = field(default=False)
    # Huggingface entity name
    entity: str = field(default=None)

@dataclass
class LogConfig:
    """Configuration for logging"""

    # Run name
    run_name: str = field(default=os.path.basename(__file__)[: -len(".py")])
    # Capture video
    video: bool = field(default=False)
    # Save model locally
    save: bool = field(default=False)

    # Wandb configuration
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # HuggingFace configuration
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)

@dataclass
class CartesianConfig:
    """Configuration for Cartesian graph based programs"""

    # Name of representation
    representation: str = 'Cartesian'
    # Number of nodes in Cartesian graph
    n_nodes: int = field(default=6)
    # Number maximum arity over the set of operators
    max_node_arity: int = field(default=2)

@dataclass
class OptimizerConfig:
    """Configuration for the Genetic Evolution based optimizer"""

    # Configuration for type of program
    program: Union[CartesianConfig] = field(default=CartesianConfig)
    # Number of individuals in population
    n_individuals: int = field(default=10)
    # Number of genomes per individual
    n_genomes: int = field(default=12)
    # Number of generations
    n_generations: int = field(default=10)
    # Number of parents mating
    n_parents_mating: int = field(default=2)
    # Probability of gene mutation
    gene_mutation_prob: float = field(default=0.05)

@dataclass
class AgentConfig:
    """Configuration for the Reinforcement Learning Agent"""

    # Size of replay buffer
    buffer_size: int = field(default=1e6)
    # Learning rate of network optimizer
    learning_rate: float = field(default=3e-4)
    # Discount factor
    gamma: float = field(default=0.99)
    # Target smoothing coefficient
    tau: float = field(default=0.005)
    # Batch size of sample from replay memory
    batch_size: int = field(default=256)
    # Scale of the policy noise
    policy_noise: float = field(default=0.1)
    # Timestep to start learning
    start_learning: int = field(default=100)
    # Frequency of training the policy
    policy_update: int = field(default=128)
    # Noise clip of the Target Policy Smoothing Regularization
    noise_clip: float = field(default=0.5)

@dataclass
class TrainingConfig:
    """Config for a training session"""

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # Agent
    agent: AgentConfig = field(default_factory=AgentConfig)

@dataclass
class ExperimentConfig:
    """Configuration for performing a single experiment"""

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Log
    log: LogConfig = field(default_factory=LogConfig)

    # Seed value
    seed: int = field(default=0)
    # Deterministic
    deterministic: bool = field(default=True)
    # CUDA use
    cuda: bool = field(default=False)

if __name__ == "__main__":
    from pprint import pprint

    # Test ExperimentConfig
    exp = ExperimentConfig()
    print(exp)
    with open('test.yaml', 'w+')as f:
        pyrallis.dump(exp, f)
        f.close()
    exp = ExperimentConfig
    with open('test.yaml', 'r')as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        pprint(data)
        pyrallis.load(exp, f)


    os.remove('test.yaml')