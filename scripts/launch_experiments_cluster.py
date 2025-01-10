import os
import subprocess
from pprint import pprint
import sys
sys.path.append('../src/cmgp/')
print(os.getcwd())

import pyrallis
import yaml
from config import ExperimentConfig

SEEDS = list(range(0, 40, 10))
SCRIPT = 'CMGP_PyGAD.py'

def run(config_file: str):

    for seed in SEEDS:
        cmd = f'sbatch run_experiment_cluster.sh {SCRIPT} --config_file={config_file} --seed={seed}'
        subprocess.run(cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    files = os.listdir('./configs')
    for file in files:
        file = os.path.join('./configs', file)
        run(file)
