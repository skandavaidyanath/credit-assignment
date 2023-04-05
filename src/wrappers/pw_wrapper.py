import gym
import numpy as np


class PushWorldWrapper(gym.Wrapper):
    """
    PushWorld processing wrapper.
    1) Gets output in the old gym "done" format
    2) Allows the option to use images or puzzle state
    3) Allows modifications to info dict if required
    """

    def __init__(self, pw_env, use_state=False):
        super(PushWorldWrapper, self).__init__(pw_env)

        self.env = pw_env
        self.use_state = use_state
        if self.use_state:
            state = self.reset()
            bound = np.ones_like(state) * np.inf
            self.observation_space = gym.spaces.Box(low=-1.0*np.inf, high=np.inf, shape=(len(state), ))

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
        state = np.array(self.env._current_state)
        return state.flatten()
