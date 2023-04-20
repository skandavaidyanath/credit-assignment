import numpy as np
import gym


def eval(env, agent, args):
    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
    else:
        continuous = False

    # logging
    total_rewards, total_successes = [], []
    ep_lens = []

    for episode in range(1, args.training.num_eval_eps + 1):
        state = env.reset()
        current_ep_reward = 0
        done = False

        rewards = []
        ep_len = 0

        while not done:
            # select action with policy
            action, _ = agent.select_action(state, greedy=True)
            if continuous:
                action = action.numpy().flatten()
                action = action.clip(
                    env.action_space.low, env.action_space.high
                )
            else:
                action = action.item()

            # Step in env
            state, reward, done, info = env.step(action)

            # saving reward and terminals
            rewards.append(float(reward))

            current_ep_reward += reward
            ep_len += 1

        total_rewards.append(current_ep_reward)
        total_successes.append(info.get("success", 0.0))
        ep_lens.append(ep_len)

    return np.mean(total_rewards), np.mean(total_successes), np.mean(ep_lens)
