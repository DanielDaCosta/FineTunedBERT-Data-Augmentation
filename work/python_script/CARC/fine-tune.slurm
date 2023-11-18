#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --account=jonmay_231

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate hw3

# run
cd ..
python3 main.py --train --eval