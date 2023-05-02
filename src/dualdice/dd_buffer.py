import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class DualDICEBuffer:
    def __init__(self, action_dim, train_val_split=[1.0, 0.0]):
        self.action_dim = action_dim

        self.num_episodes_stored = 0
        self.num_transitions_stored = 0

        self.states = []
        self.h_actions = []
        self.pi_actions = []
        self.returns = []

        self.train_val_split = train_val_split

    def add_episode(self, states, pi_actions, returns):
        self.states.extend(states)
        self.pi_actions.extend(pi_actions)
        self.returns.extend(returns)
        self.num_episodes_stored += 1
        self.num_transitions_stored += len(states)

    def clear(self):
        del self.states[:]
        del self.h_actions[:]
        del self.pi_actions[:]
        del self.returns[:]

    def get_dataloader(self, batch_size):
        states = np.array(self.states)
        h_actions = np.array(self.h_actions).reshape((-1, self.action_dim))
        pi_actions = np.array(self.pi_actions).reshape((-1, self.action_dim))
        returns = np.array(self.returns).reshape((-1, 1))

        X_h = torch.from_numpy(
            np.concatenate((states, h_actions, returns), -1)
        ).float()
        X_pi = torch.from_numpy(
            np.concatenate((states, pi_actions, np.zeros_like(returns)), -1)
        ).float()  # we concatenate 0 for \pi to indicate we don't use the returns here

        dataset = TensorDataset(X_h, X_pi)
        train_dataset, val_dataset = random_split(dataset, self.train_val_split)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        if self.train_val_split[1] > 0:
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader

    def get_input_stats(self, normalize_returns_only):
        states = np.array(self.states)
        h_actions = np.array(self.h_actions).reshape((-1, self.action_dim))
        pi_actions = np.array(self.pi_actions).reshape((-1, self.action_dim))
        actions = np.concatenate((h_actions, pi_actions), 0)
        returns = np.array(self.returns).reshape((-1, 1))

        if normalize_returns_only:
            state_mean, state_std = np.zeros(states.shape[1]), np.ones(states.shape[1])
            action_mean, action_std = np.zeros(actions.shape[1]), np.ones(actions.shape[1])
        else:
            state_mean, state_std = np.mean(states, 0), np.std(states, 0)
            action_mean, action_std = np.mean(actions, 0), np.std(actions, 0)
        return_mean, return_std = np.mean(returns, 0), np.std(returns, 0)

        inp_mean = np.concatenate((state_mean, action_mean, return_mean), 0)
        inp_std = np.concatenate((state_std, action_std, return_std), 0)
        return inp_mean, inp_std
