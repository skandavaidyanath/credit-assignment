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
