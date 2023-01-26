import numpy as np
import torch
import gridworld


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


def get_hindsight_logprobs(
    episode_rewards, policy_logprobs, total_reward, max_steps
):
    """
    Returns the return-conditioned hindsight logprobs
    from the episode rewards using the total episode reward
    and max steps in the environment. Also uses the policy logprobs,
    Return a numpy array that is the size of the total episode length.
    The probability choices here are fairly arbitrary...
    """
    hindsight_logprobs = []
    for i, reward in enumerate(episode_rewards):
        if total_reward < -max_steps:
            if reward == gridworld.REWARD_MAPPING["*"]:
                hindsight_logprobs.append(np.log(0.01))
            elif reward == gridworld.REWARD_MAPPING["F"]:
                hindsight_logprobs.append(np.log(0.99))
            else:
                hindsight_logprobs.append(
                    policy_logprobs[i].detach().cpu().item()
                )
        elif total_reward == -max_steps:
            if reward == gridworld.REWARD_MAPPING["*"]:
                hindsight_logprobs.append(np.log(0.1))
            elif reward == gridworld.REWARD_MAPPING["F"]:
                hindsight_logprobs.append(np.log(0.1))
            else:
                hindsight_logprobs.append(np.log(0.99))
        elif -max_steps < total_reward <= 0:
            if reward == gridworld.REWARD_MAPPING["*"]:
                hindsight_logprobs.append(np.log(0.99))
            elif reward == gridworld.REWARD_MAPPING["F"]:
                hindsight_logprobs.append(np.log(0.05))
            else:
                hindsight_logprobs.append(
                    policy_logprobs[i].detach().cpu().item()
                )
        elif total_reward > 0:
            if reward == gridworld.REWARD_MAPPING["*"]:
                hindsight_logprobs.append(np.log(0.99))
            elif reward == gridworld.REWARD_MAPPING["F"]:
                hindsight_logprobs.append(np.log(0.01))
            else:
                hindsight_logprobs.append(
                    policy_logprobs[i].detach().cpu().item()
                )

    return np.array(hindsight_logprobs, dtype=np.float32)
