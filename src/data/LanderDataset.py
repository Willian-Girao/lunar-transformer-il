# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class LanderDataset(Dataset):
    def __init__(self, states, actions, padding_start, weights, seq_len, normalized, mean=None, std=None, overlapping_seqs=True):
        """
        states: an array with shape (nb_episods, nb_steps, state_dim)
        actions: an array with shape (nb_episods, nb_steps,)
        padding_start: an array marking at which index the padding for each episode should start
        """
        self.states = []
        self.actions = []
        self.labels = []
        self.token_mask = []
        self.weights = []
        
        self.seq_len = seq_len
        self.normalized = normalized
        self.mean = mean
        self.std = std
        self.overlapping_seqs = overlapping_seqs
        self.nb_episodes = states.shape[0]

        if seq_len % 2 != 0:
            raise ValueError(f'The argument "seq_len" has to be even (received {seq_len}).')
        
        stride = seq_len // 2 # to interleaving states and actions.
        stride_step = 1 if overlapping_seqs else stride # overlapping or disjoint windows to create sequences chunks.

        # Creating training sequences and labels (actions) for each individual episode.
        for idx, episode in enumerate(states):
            # index withing both `actions` and `states` where elements in the array were placed as padding.
            pad_idx = padding_start[idx]
            for i in range(0, len(episode) - stride, stride_step):
                # a sequence can not be composed of only padded elements (tho padding might start somewhere within this sequence).
                if i < pad_idx:
                    # take sub-sequence of length `stride`
                    state_seq = episode[i:i+stride]         # (stride, state_dim)
                    action_seq = actions[idx][i:i+stride]   # (stride,)

                    # elements in the state/action sequences created as padding will not be attended to.
                    mask = [1 if j < pad_idx+1 else 0 for j in range(i, i+seq_len)]

                    if len(state_seq) == stride and len(action_seq) == stride:
                        self.states.append(state_seq)           # (stride, state_dim)
                        self.actions.append(action_seq[:-1])    # (stride-1,)
                        self.labels.append(action_seq[-1])      # predict next action
                        self.token_mask.append(mask[:-1])       # last element is `mask` is related to label so we drop it
                        self.weights.append(weights[idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        states_seq = torch.tensor(self.states[idx], dtype=torch.float32)
        actions_seq = torch.tensor(self.actions[idx], dtype=torch.long) # NOTE: embedding expects classes as indices.
        label = torch.tensor(self.labels[idx], dtype=torch.long)        # NOTE: CrossEntropy expects classes as indices.
        mask = torch.tensor(self.token_mask[idx], dtype=torch.long)
        reward = torch.tensor(self.weights[idx], dtype=torch.long)

        # the embedding layer of the transformer will turn
        # `states_seq` and `actions_seq` into a single input sequence.
        return states_seq, actions_seq, label, mask, reward
    
    def get_classes_weights(self):
        class_counts = torch.bincount(torch.tensor(self.labels))
        class_weights = 1.0 / class_counts.float()
        return class_weights