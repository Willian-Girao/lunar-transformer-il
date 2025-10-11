# -*- coding: utf-8 -*-
def main():
    import pickle, torch, os, re
    import numpy as np
    from src.data.LanderDataset import LanderDataset
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--training_seq_len', type=int, help='Number of sequential state vectors and actions interleaved to create an input sequence. Must be an even integer greater than 2.', default=12)
    parser.add_argument("--normalize", action="store_false", help='Whether or not the first 6 dimensions of the state space vectors are normalized.')
    parser.add_argument("--overlapping_seqs", action="store_false", help='Whether or not training sequences are created using a sliding window or are completely disjoint.')
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

    # Normalize state space vectors
    # -------------------------------------------
    mean = None
    std = None
    if args.normalize:
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
        seq_len=args.training_seq_len,
        normalized=args.normalize,
        mean=mean,
        std=std,
        overlapping_seqs=args.overlapping_seqs)
    
    seq_type = 'os' if args.overlapping_seqs else 'ds'

    file_name = dataset_name.replace('.pkl', f'-{args.training_seq_len}-{seq_type}.pt')
    path = os.path.join(project_root, 'data', 'processed', file_name)
    torch.save(dataset, path)
    print(f'... exported to: {path}')

if __name__ == "__main__":
    main()