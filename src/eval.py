import numpy as np
import gym

def eval(env, agent, num_eval_eps=32):
    print("=========== Evaluating ============")
    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
    else:
        continuous = False

    # logging
    total_rewards, total_successes = [], []

    for episode in range(1, num_eval_eps + 1):
        state = env.reset()
        current_ep_reward = 0
        done = False

        rewards = []

        while not done:
            # select action with policy
            action = agent.select_action(state, greedy=True)
            if continuous:
                action = action.numpy().flatten()
            else:
                action = action.item()

            # Step in env
            state, reward, done, info = env.step(action)

            # saving reward and terminals
            rewards.append(float(reward))

            current_ep_reward += reward

        total_rewards.append(current_ep_reward)
        total_successes.append(info["success"])

    print("\t Average eval returns: ", np.mean(total_rewards))
    print("\t Average eval success: ", np.mean(total_successes))
    print("======= Finished Evaluating =========")

    return np.mean(total_rewards), np.mean(total_successes)
