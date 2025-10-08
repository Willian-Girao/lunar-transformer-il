import pickle, torch, os, re
import numpy as np
from src.models.datasets.LanderDataset import LanderDataset
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--training_seq_len', type=int, help='Number of sequential state vectors and actions interleaved to create an input sequence. Must be an even integer greater than 2.', default=12)
args = parser.parse_args()

if args.training_seq_len <= 2 or args.training_seq_len % 2 != 0:
    raise ValueError(f'--training_seq_len must be an even integer greater than 2 (got {args.training_seq_len}).')

# Load the raw data
# -------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
dataset_name = 'gymnasium-ActorCritic-LunarLander-1000.pkl'
path = os.path.join(project_root, 'data', 'raw', dataset_name)

with open(path, 'rb') as file:
    lunarland_expert_data = pickle.load(file)

nb_train_episodes = None
match = re.search(r"-(\d+)\.pickle$", dataset_name)
if match:
    nb_train_episodes = int(match.group(1))

training_seq_len = args.training_seq_len

# Normalize state space vectors
# -------------------------------------------

continuous_states = lunarland_expert_data['X'][..., :6]  # take first 6 features

# Compute mean and std across all samples and timesteps (flatten N and T)
mean = continuous_states.mean(axis=(0, 1), keepdims=True)  # shape (1,1,6)
std  = continuous_states.std(axis=(0, 1), keepdims=True)

# Normalize in-place
lunarland_expert_data['X'][..., :6] = (continuous_states - mean) / std

# Compute sample weights from rewards
# -------------------------------------------

# shift to make min reward = 0
rewards_shifted = lunarland_expert_data['rewards'] - lunarland_expert_data['rewards'].min()
# avoid all-zero weights
if rewards_shifted.max() > 0:
    weights = rewards_shifted / rewards_shifted.max()
else:
    weights = np.ones_like(rewards_shifted)

# Instantiate and save dataset
# -------------------------------------------
print('Exporting to file...')
dataset = LanderDataset(
    states=lunarland_expert_data['X'],
    actions=lunarland_expert_data['Y'],
    padding_start=lunarland_expert_data['padding_idxs'],
    weights=weights,
    seq_len=training_seq_len)

file_name = dataset_name.replace('.pkl', f'-{training_seq_len}.pt')
path = os.path.join(project_root, 'data', 'processed', file_name)
torch.save(dataset, path)
print(f'... exported to: {path}')