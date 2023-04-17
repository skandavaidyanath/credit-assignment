import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class DualDICEBuffer:
    def __init__(self, action_dim, train_val_split=[0.9, 0.1]):
        self.action_dim = action_dim

        self.num_episodes_stored = 0
        self.num_transitions_stored = 0

        self.states = []
        self.h_actions = []
        self.pi_actions = []

        self.train_val_split = train_val_split

    def add_episode(self, states, h_actions, pi_actions):
        self.states.extend(states)
        self.h_actions.extend(h_actions)
        self.pi_actions.extend(pi_actions)
        self.num_episodes_stored += 1
        self.num_transitions_stored += len(states)

    def clear(self):
        del self.states[:]
        del self.h_actions[:]
        del self.pi_actions[:]

    def get_dataloader(self, batch_size):
        states = np.array(self.states)
        h_actions = np.array(self.h_actions).reshape((-1, self.action_dim))
        pi_actions = np.array(self.pi_actions).reshape((-1, self.action_dim))

        X_h = torch.from_numpy(np.concatenate((states, h_actions), -1)).float()
        X_pi = torch.from_numpy(
            np.concatenate((states, pi_actions), -1)
        ).float()

        dataset = TensorDataset(X_h, X_pi)
        train_dataset, val_dataset = random_split(dataset, self.train_val_split)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        return train_dataloader, val_dataloader

    def get_batch(self, batch_size):
        if batch_size > self.num_transitions_stored:
            print(
                "Warning: tried updating hca model without enough transitions in buffer!"
            )
        states = np.array(self.states)
        h_actions = np.array(self.h_actions).reshape((-1, self.action_dim))
        pi_actions = np.array(self.pi_actions).reshape((-1, self.action_dim))
        inp_h = np.concatenate((states, h_actions), -1)
        inp_pi = np.concatenate((states, pi_actions), -1)
        actions = np.array(self.actions).reshape((-1, self.action_dim))

        size = states.shape[0]
        inds = np.random.choice(size, size=batch_size, replace=False)
        return inp_h[inds], inp_pi[inds]

    def get_input_stats(self):
        states = np.array(self.states)
        h_actions = np.array(self.h_actions).reshape((-1, self.action_dim))
        pi_actions = np.array(self.pi_actions).reshape((-1, self.action_dim))
        inp_h = np.concatenate((states, h_actions), -1)
        inp_pi = np.concatenate((states, pi_actions), -1)
        inp_h_mean, inp_h_std = np.mean(inp_h, 0), np.std(inp_h, 0)
        inp_pi_mean, inp_pi_std = np.mean(inp_pi, 0), np.std(inp_pi, 0)
        return inp_h_mean, inp_h_std, inp_pi_mean, inp_pi_std
