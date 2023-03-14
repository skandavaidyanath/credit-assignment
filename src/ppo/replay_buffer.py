from utils import estimate_montecarlo_returns_adv, estimate_gae
import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout Buffer: Each element is a list of lists
    where inner lists corresponds to a single trajectory
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.rewards[:]
        del self.dones[:]

    def prep_buffer(self, gamma, lamda, adv_type, normalize_adv=True):
        # the last element in self.values is the final step value; remove this and use to compute gae.
        buffer_states = np.array(self.states, dtype=np.float32).squeeze()
        buffer_actions = np.array(self.actions, dtype=np.float32).squeeze()
        buffer_logprobs = np.array(self.logprobs, dtype=np.float32).squeeze()
        buffer_values = np.array(self.values, dtype=np.float32).squeeze()
        buffer_rewards = np.array(self.rewards, dtype=np.float32).squeeze()
        buffer_dones = np.array(self.dones, dtype=np.float32).squeeze()

        if adv_type == "gae":
            advantages, returns = estimate_gae(
                gamma,
                lamda,
                buffer_rewards,
                buffer_values,
                buffer_dones,
                normalize_adv=normalize_adv,
            )
        elif adv_type == "mc":
            advantages, returns = estimate_montecarlo_returns_adv(
                gamma,
                buffer_rewards,
                buffer_values,
                buffer_dones,
                normalize_adv=normalize_adv,
            )
        else:
            raise NotImplementedError

        # What to return: states, actions, logprobs, values, advantages, returns, None
        # The last None is for hca_ratios
        return (
            buffer_states,
            buffer_actions,
            buffer_logprobs,
            buffer_values,
            advantages,
            returns,
        )

    def generate_batches(self, gamma, lamda, batch_size, adv_type, device):
        (
            buffer_states,
            buffer_actions,
            buffer_logprobs,
            buffer_values,
            buffer_advantages,
            buffer_returns,
        ) = self.prep_buffer(gamma, lamda, adv_type)
        buffer_size = buffer_states.shape[0]
        indices = np.random.permutation(buffer_size)
        start_idx = 0
        while start_idx < buffer_size:
            batch_inds = indices[start_idx : start_idx + batch_size]
            batch_states = torch.from_numpy(buffer_states[batch_inds]).to(
                device
            )
            batch_actions = torch.from_numpy(buffer_actions[batch_inds]).to(
                device
            )
            batch_logprobs = (
                torch.from_numpy(buffer_logprobs[batch_inds])
                .flatten()
                .to(device)
            )
            batch_values = (
                torch.from_numpy(buffer_values[batch_inds]).flatten().to(device)
            )
            batch_advantages = (
                torch.from_numpy(buffer_advantages[batch_inds])
                .flatten()
                .to(device)
            )
            batch_returns = (
                torch.from_numpy(buffer_returns[batch_inds])
                .flatten()
                .to(device)
            )
            yield batch_states, batch_actions, batch_logprobs, batch_values, batch_advantages, batch_returns
            start_idx += batch_size

    def __len__(self):
        return len(self.states)


class RolloutBufferHCA(RolloutBuffer):
    def __init__(self, hindsight_model, hindsight_ratio_clip_val=None):
        super(RolloutBufferHCA, self).__init__()
        self.hindsight_logprobs = []
        self.hindsight_model = hindsight_model
        self.hindsight_ratio_clip_val = hindsight_ratio_clip_val
        self.hindsight_ratios = None

    def clear(self):
        super(RolloutBufferHCA, self).clear()
        del self.hindsight_logprobs[:]
        self.hindsight_ratios = None

    def prep_buffer(self, gamma, lamda, adv_type, normalize_adv=True):

        # lamda and adv_type are not used here.

        buffer_states = np.array(self.states, dtype=np.float32).squeeze()
        buffer_actions = np.array(self.actions, dtype=np.float32).squeeze()
        buffer_logprobs = np.array(self.logprobs, dtype=np.float32).squeeze()
        buffer_rewards = np.array(self.rewards, dtype=np.float32).squeeze()
        buffer_dones = np.array(self.dones, dtype=np.float32).squeeze()
        buffer_values = np.array(self.values, dtype=np.float32).squeeze()

        _, returns = estimate_montecarlo_returns_adv(
            gamma, buffer_rewards, None, buffer_dones
        )

        hindsight_logprobs = self.hindsight_model.get_hindsight_logprobs(
            buffer_states, returns, buffer_actions
        )
        hindsight_logprobs = hindsight_logprobs.detach().numpy()
        hindsight_ratios = np.exp(buffer_logprobs - hindsight_logprobs)
        if self.hindsight_ratio_clip_val:
            hindsight_ratios = np.clip(hindsight_ratios,
                                       a_max=self.hindsight_ratio_clip_val
                                       )

        # set this to be a class variable so we can access it from outside
        # if required
        self.hindsight_ratios = hindsight_ratios

        advantages = (1 - hindsight_ratios) * returns
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-7
            )

        # What to return: states, actions, logprobs, values, advantages, returns, hca_ratios
        return (
            buffer_states,
            buffer_actions,
            buffer_logprobs,
            buffer_values,
            advantages,
            returns,
        )
