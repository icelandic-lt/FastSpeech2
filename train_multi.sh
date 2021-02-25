#!/bin/sh

#SBATCH --mem=16G
#SBATCH --time=3200
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2063518/full-train.log

source /home/${USER}/.bashrc
source activate fs2

python train.py --restore_step=70000
