import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class DualDICEBuffer:
    def __init__(self, state_dim, action_dim, train_val_split=[1.0, 0.0]):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.num_episodes_stored = 0
        self.num_transitions_stored = 0

        self.states = []
        self.h_actions = []
        self.pi_actions = []
        self.returns = []

        # Stores return samples to estimate the second term in the DD loss. These samples can in principle
        # come from an arbitrary distribution.
        self.psi_returns = []

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
        del self.psi_returns[:]

    def get_dataloader(self, batch_size):
        if isinstance(self.state_dim, int):
            states = torch.from_numpy(np.array(self.states)).reshape(
                (-1, self.state_dim)
            )
        else:
            states = torch.from_numpy(np.array(self.states)).reshape(
                (-1, *self.state_dim)
            )
        h_actions = torch.from_numpy(np.array(self.h_actions)).reshape(
            (-1, self.action_dim)
        )
        pi_actions = torch.from_numpy(np.array(self.pi_actions)).reshape(
            (-1, self.action_dim)
        )
        returns = torch.from_numpy(np.array(self.returns)).reshape((-1, 1))
        psi_returns = torch.from_numpy(np.array(self.psi_returns)).reshape(
            (-1, 1)
        )

        dataset = TensorDataset(
            states,
            h_actions,
            returns,
            pi_actions,
            psi_returns,
        )
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
            state_mean, state_std = np.zeros(states.shape[1]), np.ones(
                states.shape[1]
            )
            action_mean, action_std = np.zeros(actions.shape[1]), np.ones(
                actions.shape[1]
            )
        else:
            state_mean, state_std = np.mean(states, 0), np.std(states, 0)
            action_mean, action_std = np.mean(actions, 0), np.std(actions, 0)
        return_mean, return_std = np.mean(returns, 0), np.std(returns, 0)

        return (
            state_mean,
            state_std,
            action_mean,
            action_std,
            return_mean,
            return_std,
        )
