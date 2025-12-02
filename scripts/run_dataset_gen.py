# Transforms expert's episodes raw data into an custom Dataset (root/data/processed).
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # get the absolute path to the project root.
os.chdir(project_root) # change working directory to project root.

os.system(f'"{sys.executable}" -m src.data.gen_lander_dataset --dataset_gen_json lander_dataset_gen.json')