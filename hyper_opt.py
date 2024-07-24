import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import optuna
import time

from TD3_program_synthesis import run_synthesis


search_space = {
    'num_individuals': tune.uniform(10, 1000),
    'num_genes': tune.uniform(4, 10),
    'num_generations': tune.uniform(3, 50),
    'num_parents_mating': tune.uniform(2, 100),
    'keep_parents': tune.uniform(2, 100),
    'mutation_percent_genes': tune.choice([5, 10, 15, 20, 25, 40])
}






def evaluate(args):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 0
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost

def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})



#algo = OptunaSearch()
#num_samples = 1000

#tuner = tune.Tuner(
#    objective,
#    tune_config=tune.TuneConfig(
#        metric="mean_loss",
#        mode="min",
#        search_alg=algo,
#        num_samples=num_samples,
#    ),
#    param_space=search_space,
#)
#results = tuner.fit()
