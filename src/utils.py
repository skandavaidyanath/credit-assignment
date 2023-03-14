import numpy as np
import torch
import gridworld
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


# def validate_model(model, val_dataloader, continuous):
#     results = []
#     for states, actions in val_dataloader:
#         preds, dists = model(states)
#         if continuous:
#             log_probs = dists.log_prob(actions)
#             results.append(log_probs.mean().item())
#         else:
#             preds = preds.argmax(-1)
#             results.append(torch.sum(preds == actions) / len(preds))
#     return results


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
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).reshape(len(inputs), -1).float()  # B x D
    h_values = h_model.get_hindsight_values(
        inputs, torch.from_numpy(np.array(actions))
    )
    return h_values.detach().tolist()


def estimate_montecarlo_returns_adv(gamma, rewards, values, terminals, normalize_adv=True):
    # Monte Carlo estimate of returns
    batch_size = len(rewards)
    returns = np.zeros(batch_size)
    returns[batch_size - 1] = rewards[batch_size - 1]

    for t in reversed(range(batch_size - 1)):
        returns[t] = rewards[t] + returns[t + 1] * gamma * (
                1 - terminals[t]
        )

    advantages = None
    if values:
        advantages = returns - values

        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-7
            )
        advantages = advantages.astype(np.float32)
    return advantages, returns.astype(np.float32)

def estimate_gae(gamma, lamda, rewards, values, terminals, normalize_adv=True):
    # GAE estimates of Advantage
    batch_size = len(rewards)
    advantages = np.zeros(batch_size, dtype=np.float32)
    advantages[batch_size - 1] = (
        rewards[batch_size - 1] - values[batch_size - 1]
    )
    for t in reversed(range(batch_size - 1)):
        delta = (
            rewards[t]
            + (gamma * values[t + 1] * (1 - terminals[t]))
            - values[t]
        )
        advantages[t] = delta + (
            gamma * lamda * advantages[t + 1] * (1 - terminals[t])
        )

    returns = advantages + values
    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-7
        )
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    return advantages, returns