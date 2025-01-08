#!/bin/bash
#SBATCH --job-name=cm-gp
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

args=("$@")

# Load modules
#module load Python/3.11.3-GCCcore-12.3.0
#module load SciPy-bundle/2023.07-gfbf-2023a
#module load PyTorch-bundle/2.3.0-foss-2023a

# Load environment
#source /scratch/brussel/103/vsc10368/_VENVS/cm-gp/bin/activate

# Run experiment
python ${args[*]}