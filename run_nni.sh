#!/bin/bash
#SBATCH --job-name="nni-IL"
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=18GB
#SBATCH --time=3-0:0:0

module purge

source $HOME/venvs/nni-venv/bin/activate

python3 -m src.hpo.nni_main --search_space_json /scratch/p302242/lunar-transformer-il/configs/hpo_search_space_noise.json --config_json /scratch/p302242/lunar-transformer-il/configs/nni_config.json

deactivate