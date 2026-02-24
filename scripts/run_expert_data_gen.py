# Experts will control the environments so that episode (raw) data can be
# exported to file  (root/data/raw).
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

nb_episodes = 1000

# Generates the training data
if args.env == 'lunar':
    os.system(f'"{sys.executable}" -m src.data.gen_lander_expert_data --max_steps 400 --nb_episodes {nb_episodes}')
else:
    os.system(f'"{sys.executable}" -m src.data.gen_cheetah_expert_data --nb_episodes {nb_episodes}')