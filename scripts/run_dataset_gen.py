# Transforms expert's episodes raw data into an custom Dataset (root/data/processed).
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

if args.env == 'lunar':
    os.system(f'"{sys.executable}" -m src.data.gen_lander_dataset --dataset_gen_json lander_dataset_gen.json')
else:
    os.system(f'"{sys.executable}" -m src.data.gen_cheetah_dataset --dataset_gen_json cheetah_dataset_gen.json')