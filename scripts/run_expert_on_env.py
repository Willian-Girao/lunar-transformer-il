# Experts will control the environments so that episode (raw) data can be
# exported to file  (root/data/raw).
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

os.system(f'"{sys.executable}" -m src.expert.run_expert_example --seed 1002')