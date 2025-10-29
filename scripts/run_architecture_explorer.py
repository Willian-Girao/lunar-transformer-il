import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

os.system(f'"{sys.executable}" -m src.hpo.explore_baseline_architecture --baseline_architecture_space baseline_architecture_space.json --train_json training_config_baseline.json')