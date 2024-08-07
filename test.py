import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import optuna
import time
import pyrallis

from TD3_program_synthesis_redux import run_synthesis, Args
#from test import run_synthesis


search_space = {
    'num_individuals': tune.randint(100, 1000),
    'num_genes': tune.randint(4, 10),
    'num_generations': tune.randint(3, 50),
    'num_parents_mating': tune.randint(2, 10),
    'keep_parents': tune.randint(2, 4),
    'mutation_percent_genes': tune.choice([5, 10, 15, 20, 25, 40]),
    'keep_elitism': tune.randint(0, 10)
}

def run(search_config):
    config = pyrallis.parse(config_class=Args)
    for k, v in search_config.items():
        config.__dict__[k] = v
    result = run_synthesis(Args)
    return train.report(result)

tuner = tune.Tuner(
    run,
    tune_config=tune.TuneConfig(
        metric="mean_eval",
        mode="max",
        search_alg=OptunaSearch(),
        num_samples=1,
    ),
    param_space=search_space,
)

results = tuner.fit()