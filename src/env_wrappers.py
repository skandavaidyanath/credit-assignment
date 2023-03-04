import gym 

class HalfCheetahWrapper(gym.Wrapper):
    def __init__(self, env):
        super(HalfCheetahWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["terminal_state"] = False
        if done:
            info["timeout"] = True
        else:
            info["timeout"] = False
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
