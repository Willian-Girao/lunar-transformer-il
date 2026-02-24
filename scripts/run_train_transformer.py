import os, sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--env',
    type=str,
    choices=['lunar', 'cheetah'],
    required=True,
    help='Model type: expert or transformer.'
)
args = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

os.system(f'"{sys.executable}" -m src.training.train_transformer --env {args.env} --transformer_json baseline_cheetah_transformer.json --train_json training_config_baseline.json')