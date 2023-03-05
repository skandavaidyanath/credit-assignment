from utils import flatten, estimate_montecarlo_returns_adv, estimate_gae
import numpy as np
import torch

class RolloutBuffer:
    '''
        Rollout Buffer: Each element is a list of lists
        where inner lists corresponds to a single trajectory
    '''

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.terminals = []
        self.hindsight_logprobs = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.rewards[:]
        del self.terminals[:]
        del self.hindsight_logprobs[:]

    def prep_buffer(self, gamma, lamda, adv_type):
        # the last element in self.values is the final step value; remove this and use to compute gae.
        buffer_states = np.array(self.states, dtype=np.float32).squeeze()
        buffer_actions = np.array(self.actions, dtype=np.float32).squeeze()
        buffer_logprobs = np.array(self.logprobs, dtype=np.float32).squeeze()
        buffer_rewards = np.array(self.rewards, dtype=np.float32).squeeze()
        buffer_terminals = np.array(self.terminals, dtype=np.float32).squeeze()
        buffer_values = np.array(self.values, dtype=np.float32).squeeze()

        if adv_type == "gae":
            advantages, returns = estimate_gae(
                gamma, lamda, buffer_rewards, buffer_values, buffer_terminals
            )
        elif adv_type == "mc":
            advantages, returns = estimate_montecarlo_returns_adv(
                gamma, buffer_rewards, buffer_values, buffer_terminals
            )
        else:
            raise NotImplementedError

        # What to return: states, actions, logprobs, values, advantages, returns.
        return buffer_states, buffer_actions, buffer_logprobs, buffer_values, advantages, returns

    def generate_batches(self, gamma, lamda, batch_size, adv_type, device):
        buffer_states, buffer_actions, buffer_logprobs, buffer_values, buffer_advantages, buffer_returns = \
            self.prep_buffer(gamma, lamda, adv_type)
        buffer_size = buffer_states.shape[0]
        indices = np.random.permutation(buffer_size)
        start_idx = 0
        while start_idx < buffer_size:
            batch_inds = indices[start_idx : start_idx + batch_size]
            batch_states = torch.from_numpy(buffer_states[batch_inds]).to(device)
            batch_actions = torch.from_numpy(buffer_actions[batch_inds]).to(device)
            batch_logprobs = torch.from_numpy(buffer_logprobs[batch_inds]).flatten().to(device)
            batch_values = torch.from_numpy(buffer_values[batch_inds]).flatten().to(device)
            batch_advantages = torch.from_numpy(buffer_advantages[batch_inds]).flatten().to(device)
            batch_returns = torch.from_numpy(buffer_returns[batch_inds]).flatten().to(device)
            yield batch_states, batch_actions, batch_logprobs, batch_values, batch_advantages, batch_returns
            start_idx += batch_size

    def __len__(self):
        return len(self.states)
