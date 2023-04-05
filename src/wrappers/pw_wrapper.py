import gym
import numpy as np


class PushWorldWrapper(gym.Wrapper):
    """
    PushWorld processing wrapper.
    1) Gets output in the old gym "done" format
    2) Allows the option to use images or puzzle state
    3) Allows modifications to info dict if required
    """

    def __init__(pw_env, use_state=False):
        super(PushWorldWrapper, self).__init__(env)

        self.env = env
        self.use_state = use_state

    def reset(self):
        observation, info = self.env.reset()
        if self.use_state:
            return self._get_state()
        else:
            return observation

    def step(self, action):
        assert action in self.action_space
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self.use_state:
            return self._get_state(), reward, done, info
        else:
            return observation, reward, done, info

    def _get_state(self):
        return np.array(self.env._current_state)
