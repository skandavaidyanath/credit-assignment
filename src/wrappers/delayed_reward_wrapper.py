import gym


class DelayedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.returns = 0.0

    def reset(self):
        self.returns = 0.0
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.returns += reward
        returned_reward = self.returns if done else 0.0
        return state, returned_reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
