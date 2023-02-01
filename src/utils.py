import numpy as np
import torch
import gridworld
import pickle
import datetime
import os


class HCABuffer:
    def __init__(self, exp_name):
        self.num_episodes_stored = 0
        self.states = []
        self.actions = []
        self.returns = []

        self.checkpoint_path = f"hca_data/{exp_name}_"
        self.checkpoint_path += (
            f"{datetime.datetime.now().replace(microsecond=0)}"
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def add_episode(self, episode_states, episode_actions, episode_rewards, gamma):
        episode_return = np.sum(episode_rewards)
        rewards = np.array(episode_rewards).reshape(-1, 1)
        episode_returns = list(np.array(calculate_mc_returns(rewards, np.zeros_like(rewards), gamma)).flatten())

        self.states.extend(episode_states)
        self.actions.extend(episode_actions)
        self.returns.extend(episode_returns)
        self.num_episodes_stored += 1

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


def flatten(x):
    """
    Flattens a list of lists into a numpy array
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return np.array(out, dtype=np.float32).squeeze()


def tensor_flatten(x):
    """
    Flattens a list of lists of tensor with gradients to torch tensor with gradients
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return torch.stack(out).squeeze()


def calculate_mc_returns(rewards, terminals, gamma):
    """
    Calculates MC returns
    Duplicated from ppo.py.
    """
    batch_size = len(rewards)
    returns = [0 for _ in range(batch_size)]
    returns[batch_size - 1] = rewards[batch_size - 1]
    for t in reversed(range(batch_size - 1)):
        returns[t] = rewards[t] + returns[t + 1] * gamma * (1 - terminals[t])

    return returns


def get_human_hindsight_logprobs(
    episode_rewards, policy_logprobs, total_reward, max_steps
):
    """
    Returns the return-conditioned hindsight logprobs
    from the episode rewards using the total episode reward
    and max steps in the environment. Also uses the policy logprobs,
    Return a numpy array that is the size of the total episode length.
    The probability choices here are fairly arbitrary...
    The -1 added to the reward mapping is for the step we take
    """
    hindsight_logprobs = []
    for i, reward in enumerate(episode_rewards):
        # agent stepped into fire; assign high probability to actions which transitioned into fire, and low probability
        # to actions that got a diamond.
        if total_reward < -max_steps:
            if reward == gridworld.REWARD_MAPPING["*"] - 1:
                hindsight_logprobs.append(np.log(0.01))
            elif reward == gridworld.REWARD_MAPPING["F"] - 1:
                hindsight_logprobs.append(np.log(0.99))
            else:
                # don't want to change the weighting for this
                hindsight_logprobs.append(np.inf)
        # Not very likely that agent stepped into fire, even less likely that agent picked up a diamond.
        # Very likely that all the agent was transition into empty tiles.
        elif total_reward == -max_steps:
            if reward == gridworld.REWARD_MAPPING["*"] - 1:
                hindsight_logprobs.append(np.log(0.01))
            elif reward == gridworld.REWARD_MAPPING["F"] - 1:
                hindsight_logprobs.append(np.log(0.1))
            else:
                hindsight_logprobs.append(np.log(0.99))
        # Agent has to have picked up at least one diamond; transition into fire isn't super likely, but possible.
        elif -max_steps < total_reward <= 0:
            if reward == gridworld.REWARD_MAPPING["*"] - 1:
                hindsight_logprobs.append(np.log(0.99))
            elif reward == gridworld.REWARD_MAPPING["F"] - 1:
                hindsight_logprobs.append(np.log(0.05))
            else:
                # don't want to change the weighting for this
                hindsight_logprobs.append(np.inf)
        # Agent must have picked up at least one diamond; transitions into fire are highly unlikely.
        elif total_reward > 0:
            if reward == gridworld.REWARD_MAPPING["*"] - 1:
                hindsight_logprobs.append(np.log(0.99))
            elif reward == gridworld.REWARD_MAPPING["F"] - 1:
                hindsight_logprobs.append(np.log(0.01))
            else:
                # don't want to change the weighting for this
                hindsight_logprobs.append(np.inf)

    return np.array(hindsight_logprobs, dtype=np.float32)


def get_hindsight_logprobs(h_model, states, returns, actions):
    inputs = []
    for state, g in zip(states, returns):
        inputs.append(np.concatenate([state, [g]]))
    inputs = torch.Tensor(inputs).reshape(len(inputs), -1).float()  # B x D
    h_values = h_model.get_hindsight_values(
        inputs, torch.Tensor(actions).long()
    )
    return h_values.detach().tolist()
