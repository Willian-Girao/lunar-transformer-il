import os, sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--model',
    type=str,
    choices=['expert', 'transformer'],
    required=True,
    help='Model type: expert or transformer.'
)
args = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

if args.model == 'transformer':
    os.system(f'"{sys.executable}" -m src.evaluation.within_seed_variability --test_json xp_across_within_seed_var.json')
else:
    os.system(f'"{sys.executable}" -m src.expert.within_seed_variability --test_json xp_across_within_seed_var.json')