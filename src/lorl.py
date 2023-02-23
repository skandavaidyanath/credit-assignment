import gym
import numpy as np
import copy
import os
import pickle
from tqdm import tqdm
import cv2

TASKS = [
    "open drawer",
    "close drawer",
    "turn faucet right",
    "turn faucet left",
    "move black mug right",
    "move white mug down",
]


def save_im(im, name):
    """
    Save an image
    """
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, im.astype(np.uint8))


def lorl_gt_reward(qpos, initial, task):
    """
    Measure true task progress for different instructions
    """
    if task == "open drawer":
        dist = initial[14] - qpos[14]
        s = dist > 0.02
    elif task == "close drawer":
        dist = qpos[14] - initial[14]
        s = dist > 0.02
    elif task == "turn faucet right":
        dist = initial[13] - qpos[13]
        s = dist > np.pi / 10
    elif task == "turn faucet left":
        dist = qpos[13] - initial[13]
        s = dist > np.pi / 10
    elif task == "move black mug right":
        dist = initial[11] - qpos[11]
        s = dist > 0.02
    elif task == "move white mug down":
        dist = qpos[10] - initial[10]
        s = dist > 0.02
    return dist, s


class LorlWrapper(gym.Wrapper):
    """
    Lorl processing wrapper.
    1) Adds String command to observations
    2) Preprocess states
    """

    def __init__(self, env, use_state=True, max_steps=20, normalize=False):
        super(LorlWrapper, self).__init__(env)

        self.env = env
        self.use_state = use_state
        self.max_steps = max_steps
        self.normalize = normalize
        
        self.state_dim = 15 if use_state else (3, 64, 64)
        self.act_dim = env.action_space.shape[0]

        if normalize:
            if self.use_state:
                stats = pickle.load(open("static/lorl_state_stats.pkl", "rb"))
                self.state_mean, self.state_std = stats["mean"], stats["std"]
            else:
                print("Image observations are normalized by default. Setting mean and std to 0 and 1 respectively.")
                self.state_mean = np.zeros(self.state_dim)
                self.state_std = np.ones(self.state_dim)

        if isinstance(self.state_dim, tuple):
            # Image inputs
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.state_dim,
                dtype=np.float32,
            )
        else:
            # State inputs
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32,
            )

        self.action_space = env.action_space

        self.cur_step = 0
        self.task = None
        self.initial_state = None

    def reset(self, task, render=False, **kwargs):
        if render:
            render_path = kwargs["render_path"]

        env = self.env
        im, _ = env.reset()
        self.cur_step = 0

        if task not in TASKS:
            raise ValueError(f"Unknown task! Choose from {TASKS}")

        self.task = task

        # Initialize state for different tasks
        if task == "open drawer":
            env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
        elif task == "close drawer":
            env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
        elif task == "turn faucet right":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi / 5, np.pi / 5)
        elif task == "turn faucet left":
            env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi / 5, np.pi / 5)
        elif task == "move black mug right":
            env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
        elif task == "move white mug down":
            env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
            env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)

        if "mug" in task:
            env._reset_hand(pos=[-0.1, 0.55, 0.1])
        else:
            env._reset_hand(pos=[0, 0.45, 0.1])

        for _ in range(50):
            env.sim.step()

        reset_state = copy.deepcopy(env.sim.data.qpos[:])
        env.sim.data.qpos[:] = reset_state
        env.sim.data.qacc[:] = 0
        env.sim.data.qvel[:] = 0
        env.sim.step()
        self.initial_state = copy.deepcopy(env.sim.data.qpos[:])

        if render:
            # Initialize goal image for initial state
            if task == "open drawer":
                env.sim.data.qpos[14] = -0.15
            elif task == "close drawer":
                env.sim.data.qpos[14] = 0.0
            elif task == "turn faucet right":
                env.sim.data.qpos[13] -= np.pi / 5
            elif task == "turn faucet left":
                env.sim.data.qpos[13] += np.pi / 5
            elif task == "move black mug right":
                env.sim.data.qpos[11] -= 0.1
            elif task == "move white mug down":
                env.sim.data.qpos[10] += 0.1

            env.sim.step()
            gim = env._get_obs()[:, :, :3]

            # Reset inital state
            env.sim.data.qpos[:] = reset_state
            env.sim.data.qacc[:] = 0
            env.sim.data.qvel[:] = 0
            env.sim.step()

            im = env._get_obs()[:, :, :3]
            initim = im
            save_im(
                (initim * 255.0).astype(np.uint8),
                render_path + f"/initialim_{task}.jpg",
            )
            save_im(
                (gim * 255.0).astype(np.uint8),
                render_path + f"gim_{task}.jpg",
            )

        cur_state = self.get_state(im)

        return cur_state

    def step(self, action):
        assert action in self.action_space
        self.cur_step += 1

        im, _, _, info = self.env.step(action)
        dist, s = lorl_gt_reward(
            self.env.sim.data.qpos[:], self.initial_state, self.task
        )

        reward = 0
        success = 0
        if s:
            success = 1
            reward = dist*1000  # just increasing reward magnitude to get better gradients

        done = s or (self.cur_step >= self.max_steps)

        info.update({"success": success})
        return self.get_state(im), reward, done, info

    def get_state(self, obs):
        """Returns the preprocessed observation"""

        if self.use_state:
            state = self.env.sim.data.qpos[:]
        else:
            state = np.moveaxis(obs, 2, 0)  # make H,W,C to C,H,W
        if self.normalize:
            state = (state - self.state_mean) / self.state_std
        return state

    def get_image(self, h=1024, w=1024):
        """
        Get image to visualize
        """
        obs = self.sim.render(h, w, camera_name="cam0") / 255.0
        im = np.flip(obs, 0).copy()
        return (im[:, :, :3] * 255.0).astype(np.uint8)


if __name__ == "__main__":
    import lorl_env
    import gym

    env = gym.make("LorlEnv-v0")
    wrapped_env = LorlWrapper(env, normalize=True)
    print(wrapped_env.reset("open drawer"))
