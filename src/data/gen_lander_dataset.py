# -*- coding: utf-8 -*-
def main():
    import pickle, torch, os, re
    import numpy as np
    from src.data.LanderDataset import LanderDataset
    from src.utils.json_handling import json_2_dict
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_gen_json', type=str, help='Configuration .json file describing the parameters for dataset generation.')
    args = parser.parse_args()

    config = json_2_dict(os.path.join(project_root, 'configs', args.dataset_gen_json))

    if config['training_seq_len'] <= 2 or config['training_seq_len'] % 2 != 0:
        training_seq_len = config['training_seq_len']
        raise ValueError(f'--training_seq_len must be an even integer greater than 2 (got {training_seq_len}).')

    # Load the raw data
    # -------------------------------------------
    path = os.path.join(project_root, 'data', 'raw', config['expert_data_file'])

    with open(path, 'rb') as file:
        lunarland_expert_data = pickle.load(file)

    # Normalize state space vectors
    # -------------------------------------------
    mean = None
    std = None
    if config['normalize']:
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
        seq_len=config['training_seq_len'],
        normalized=config['normalize'],
        mean=mean,
        std=std,
        overlapping_seqs=config['overlapping_seqs'])
    
    seq_type = 'os' if config['overlapping_seqs'] else 'ds'

    os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
    
    file_name = config['expert_data_file'].replace('.pkl', f'-{config['training_seq_len']}-{seq_type}.pt')
    path = os.path.join(project_root, 'data', 'processed', file_name)
    torch.save(dataset, path)
    print(f'... exported to: {path}')

if __name__ == "__main__":
    main()