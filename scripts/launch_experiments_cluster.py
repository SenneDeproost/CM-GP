import os
import subprocess
from pprint import pprint
os.chdir('../')

import pyrallis
import yaml
from src.cmgp.config import ExperimentConfig

SEEDS = list(range(0, 400, 10))
SCRIPT = 'CMGP.py'

def run(config_file: str):

    for seed in SEEDS:
        cmd = f'bash run_experiment_cluster.sh {SCRIPT} --config --seed={seed}'
        subprocess.run(cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    files = os.listdir('./configs')
    for file in files:
        run(file)
