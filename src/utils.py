import numpy as np
import torch
from gridworld.gridworld_env import GridWorld
import gym

try:
    import d4rl
except:
    print("D4RL is not installed!")
try:
    import lorl_env
except:
    print("Lorl env is not installed!")

from wrappers.lorl import LorlWrapper
from wrappers.mujoco import MujocoWrapper


def get_env(args):
    if args.env.type == "d4rl":
        env = gym.make(args.env.name)
    elif args.env.type == "gridworld":
        env = GridWorld(args.env.puzzle_path, sparse=args.env.sparse)
    elif args.env.type == "lorl":
        env = LorlWrapper(
            gym.make(args.env.name),
            task=args.env.task,
            use_state=args.env.use_state,
            normalize=args.env.normalize,
        )
    elif args.env.type == "mujoco":
        env = gym.make(args.env.name)
        env = MujocoWrapper(env)
    else:
        raise NotImplementedError
    return env


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


def get_hindsight_logprobs(h_model, states, returns, actions):
    inputs = []
    for state, g in zip(states, returns):
        inputs.append(np.concatenate([state, [g]]))
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).reshape(len(inputs), -1).float()  # B x D
    h_values = h_model.get_hindsight_values(
        inputs, torch.from_numpy(np.array(actions))
    )
    return h_values.detach().tolist()


def estimate_montecarlo_returns_adv(
    gamma, rewards, values, dones, normalize_adv=True
):
    # Monte Carlo estimate of returns
    batch_size = len(rewards)
    returns = np.zeros(batch_size)
    returns[batch_size - 1] = rewards[batch_size - 1]

    for t in reversed(range(batch_size - 1)):
        returns[t] = rewards[t] + returns[t + 1] * gamma * (1 - dones[t])

    advantages = None
    if values is not None:
        advantages = returns - values

        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-7
            )
        advantages = advantages.astype(np.float32)
    return advantages, returns.astype(np.float32)


def estimate_gae(gamma, lamda, rewards, values, dones, normalize_adv=True):
    # GAE estimates of Advantage
    batch_size = len(rewards)
    advantages = np.zeros(batch_size, dtype=np.float32)
    advantages[batch_size - 1] = (
        rewards[batch_size - 1] - values[batch_size - 1]
    )
    for t in reversed(range(batch_size - 1)):
        delta = (
            rewards[t] + (gamma * values[t + 1] * (1 - dones[t])) - values[t]
        )
        advantages[t] = delta + (
            gamma * lamda * advantages[t + 1] * (1 - dones[t])
        )

    returns = advantages + values
    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    return advantages, returns
