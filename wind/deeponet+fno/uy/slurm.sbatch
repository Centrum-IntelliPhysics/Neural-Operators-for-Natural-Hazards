#!/bin/bash -l
#SBATCH --job-name=deeponet
#SBATCH -t 10:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -A sgoswam4_gpu
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

source $HOME/jax.venv/bin/activate
python3 -u deeponet.py

