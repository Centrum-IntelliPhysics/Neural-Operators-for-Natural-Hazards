#!/bin/bash -l
#SBATCH --job-name=deeponet
#SBATCH -t 10:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -A sgoswam4_gpu
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

source $HOME/jax.venv/bin/activate
cd /home/sgoswam4/data_sgoswam4/Natural_hazards/Earthquake/deeponet+fno/
python3 -u deeponet.py
