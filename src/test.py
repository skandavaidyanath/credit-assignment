import numpy as np


def estimate_hca_discounted_returns(rewards, logprobs, hindsight_logprobs):
    """All of the args should be numpy arrays"""
    assert len(rewards) == len(logprobs) == len(hindsight_logprobs)
    ret = []
    for ep_rew, ep_logprobs, ep_hindsight_logprobs in zip(rewards, logprobs, hindsight_logprobs):
        ep_hindsight_logprobs = np.array(ep_hindsight_logprobs)
        ratios = np.exp(ep_logprobs - ep_hindsight_logprobs)
        gammas = 1 - ratios
        T = len(ratios)
        ep_returns = []
        for t in range(T):
            curr_gamma = np.array([gammas[t]**i for i in range(T - t)])
            discounted_return = (curr_gamma * ep_rew[t:]).sum()
            ep_returns.append(discounted_return)
        ret.append(ep_returns)
    # flatten?
    return ret

if __name__ == '__main__':
    rewards = [np.random.rand(3, ) for _ in range(2)]
    logprobs = [np.random.rand(3, ) for _ in range(2)]
    hindsight_logprobs = [np.random.rand(3, ) for _ in range(2)]
    a = estimate_hca_discounted_returns(rewards, logprobs, hindsight_logprobs)
    print(a)