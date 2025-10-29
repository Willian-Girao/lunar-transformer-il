# Experts will control the environments so that episode (raw) data can be
# exported to file  (root/data/raw).
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

os.system(f'"{sys.executable}" -m src.data.gen_lander_expert_data --max_steps 400 --nb_episodes 1000')
