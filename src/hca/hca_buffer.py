import datetime
import os
import pickle
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def calculate_mc_returns(rewards, terminals, gamma):
    """
    Calculates MC returns
    Duplicated from ppo_algo.py.
    """
    batch_size = len(rewards)
    returns = [0 for _ in range(batch_size)]
    returns[batch_size - 1] = rewards[batch_size - 1]
    for t in reversed(range(batch_size - 1)):
        returns[t] = rewards[t] + returns[t + 1] * gamma * (1 - terminals[t])

    return returns


class HCABuffer:
    def __init__(self, exp_name, action_dim, train_val_split=[0.9, 0.1]):
        self.action_dim = action_dim
        self.num_episodes_stored = 0
        self.num_transitions_stored = 0
        self.states = []
        self.actions = []
        self.returns = []

        # Can probably remove this but leaving it in for now
        self.checkpoint_path = f"hca_data/{exp_name}_"
        self.checkpoint_path += (
            f"{datetime.datetime.now().replace(microsecond=0)}"
        )

        self.train_val_split = train_val_split

    def add_episode(
        self, episode_states, episode_actions, episode_rewards, gamma
    ):
        rewards = np.array(episode_rewards).reshape(-1, 1)
        episode_returns = list(
            np.array(
                calculate_mc_returns(rewards, np.zeros_like(rewards), gamma)
            ).flatten()
        )

        self.states.extend(episode_states)
        self.actions.extend(episode_actions)
        self.returns.extend(episode_returns)
        self.num_episodes_stored += 1
        self.num_transitions_stored += rewards.shape[0]

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.returns[:]

    def get_dataloader(self, batch_size):
        states = np.array(self.states)
        returns = np.array(self.returns).reshape((-1, 1))
        X = torch.from_numpy(np.concatenate((states, returns), -1)).float()

        y = torch.from_numpy(
            np.array(self.actions).reshape((-1, self.action_dim))
        )

        dataset = TensorDataset(X, y)
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
        returns = np.array(self.returns).reshape((-1, 1))
        inp_data = np.concatenate((states, returns), -1)
        actions = np.array(self.actions).reshape((-1, self.action_dim))

        size = states.shape[0]
        inds = np.random.choice(size, size=batch_size, replace=False)
        return inp_data[inds], actions[inds]

    def save_data(self, num_actions):
        states = np.array(self.states)
        returns = np.array(self.returns).reshape((-1, 1))
        inp_data = np.concatenate((states, returns), -1)
        save_dict = {
            "x": inp_data,
            "y": np.array(self.actions),
            "num_acts": num_actions,
        }
        filename = str(self.num_episodes_stored) + "_eps"

        with open(self.checkpoint_path + "/" + filename + ".pkl", "wb") as f:
            pickle.dump(save_dict, f)