import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import digitize_returns

def quantize_returns(returns, num_classes):
    min_returns, max_returns = np.min(returns), np.max(returns)
    bins = np.linspace(min_returns, max_returns, num_classes + 1)
    quantized_returns = digitize_returns(returns, bins)
    return quantized_returns, bins


class ReturnBuffer:
    def __init__(self, num_classes, train_val_split=[0.9, 0.1]):
        self.num_classes = num_classes  # implictly assume if num_classes > 1, then quantize=False

        self.num_episodes_stored = 0
        self.num_transitions_stored = 0

        self.states = []
        self.returns = []

        self.train_val_split = train_val_split

    def add_episode(self, states, returns):
        self.states.extend(states)
        self.returns.extend(returns)
        self.num_episodes_stored += 1
        self.num_transitions_stored += len(states)

    def clear(self):
        del self.states[:]
        del self.returns[:]

    def get_dataloader(self, batch_size):
        states = np.array(self.states)
        # returns = np.array(self.returns).reshape((-1, self.num_classes))
        returns = np.array(self.returns).reshape(-1, 1)

        if self.num_classes > 1:
            # quantize=True implicitly
            returns, bins = quantize_returns(returns, self.num_classes)

        states = torch.from_numpy(states).float()
        returns = torch.from_numpy(returns).float()

        dataset = TensorDataset(states, returns)
        train_dataset, val_dataset = random_split(dataset, self.train_val_split)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        # we need to return bins to use it for test time examples
        # TODO: have a default for bins.
        return train_dataloader, val_dataloader, bins

    def get_input_stats(self):
        states = np.array(self.states)
        inp_mean, inp_std = np.mean(states, 0), np.std(states, 0)
        return inp_mean, inp_std
