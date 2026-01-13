#!/bin/bash
#SBATCH --job-name="tra-IL"
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=18GB
#SBATCH --time=0-1:0:0

module purge

source $HOME/venvs/nni-venv/bin/activate

python3 -m src.training.train_transformer --transformer_json /scratch/p302242/lunar-transformer-il/configs/baseline_transformer.json --train_json /scratch/p302242/lunar-transformer-il/configs/training_config_baseline.json

deactivate