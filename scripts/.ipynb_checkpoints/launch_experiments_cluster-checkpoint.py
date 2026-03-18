import os
import subprocess
from pprint import pprint
os.chdir('../')

import pyrallis
import yaml
from src.cmgp.config import ExperimentConfig

#SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = [990]
SCRIPT = 'CMGP.py'

def run(config_file: str):

    for seed in SEEDS:
        cmd = f'bash run_experiment_cluster.sh {SCRIPT} --config --seed={seed}'
        subprocess.run(cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    files = os.listdir('./configs')
    for file in files:
        run(file)
