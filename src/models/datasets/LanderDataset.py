import torch
from torch.utils.data import Dataset
import numpy as np

class LanderDataset(Dataset):
    def __init__(self, states, actions, padding_start, weights, seq_len, normalized):
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
        self.normalized = normalized

        if seq_len % 2 != 0:
            raise ValueError(f'The argument "seq_len" has to be even (received {seq_len}).')

        # Interleaving states and actions.
        stride = seq_len // 2

        # Creating training sequences and labels (actions) for each individual episode.
        for idx, episode in enumerate(states):
            for i in range(0, len(episode) - stride):
                # index withing both `actions` and `states` where elements in the array were placed as padding.
                pad_idx = padding_start[idx]

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