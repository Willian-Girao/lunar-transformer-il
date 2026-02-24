"""
https://github.com/DishankJ/Soft-Actor-Critic-PyTorch/tree/main
"""
import os
import torch
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)
        self.max_size = max_size
    def push(self, transition: Transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        idx = torch.randint(len(self.memory), (batch_size,))
        transitions = Transition(*zip(*[self.memory[i] for i in idx]))
        return transitions
    def clear(self):
        self.memory.clear()

def plot_learning_curve(rewards):
    running_avg = np.zeros(len(rewards))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(rewards[max(0, i - 100):(i + 1)])
    plt.plot([i+1 for i in range(len(rewards))], running_avg)
    plt.title("Running Average")
    plt.savefig('saved/sac.png')
    plt.show()