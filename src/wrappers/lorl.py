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
    "open drawer and move black mug right",
    "move black mug right and close drawer",
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
    elif task == "open drawer and move black mug right":
        dist1 = initial[14] - qpos[14]
        dist2 = initial[11] - qpos[11]
        s = dist1 > 0.02 and dist2 > 0.02
        dist = dist1 + dist2
    elif task == "move white mug down and close drawer":
        dist1 = qpos[10] - initial[10]
        dist2 = qpos[14] - initial[14]
        s = dist1 > 0.02 and dist2 > 0.02
        dist = dist1 + dist2

    return dist, s


class LorlWrapper(gym.Wrapper):
    """
    Lorl processing wrapper.
    1) Implements reset() based on the passed task
    2) Implements get_state() and get_image() methods
    3) Normalized state if specified
    """

    def __init__(
        self,
        env,
        task,
        use_state=True,
        reward_multiplier=1000,
        binary_reward=False,
        max_steps=20,
        normalize=True,
    ):
        super(LorlWrapper, self).__init__(env)

        self.env = env

        if task not in TASKS:
            raise ValueError(f"Unknown task! Choose from {TASKS}")

        self.task = task

        self.use_state = use_state
        self.reward_multiplier = reward_multiplier
        self.binary_reward = binary_reward
        self.max_steps = max_steps
        self.normalize = normalize

        self.state_dim = 15 if use_state else (3, 64, 64)
        self.act_dim = env.action_space.shape[0]

        if normalize:
            if self.use_state:
                stats = pickle.load(
                    open("static/lorl_offline_state_stats.pkl", "rb")
                )
                # stats = pickle.load(open("static/lorl_state_stats.pkl", "rb"))
                self.state_mean, self.state_std = stats["mean"], stats["std"]
            else:
                print(
                    "Image observations are normalized by default. Setting mean and std to 0 and 1 respectively."
                )
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
        self.initial_state = None

    def reset(self, render=False, **kwargs):
        if render:
            render_path = kwargs["render_path"]

        env = self.env
        im, _ = env.reset()
        self.cur_step = 0

        task = self.task

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
        success = int(s)

        done = s or (self.cur_step >= self.max_steps)

        if done:
            if self.binary_reward:
                reward = int(s)
            else:
                reward = (
                    dist * self.reward_multiplier
                )  # just increasing reward magnitude to get better gradients

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
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = gym.make("LorlEnv-v0")

    def make_env(task):
        def _init():
            wrapped_env = LorlWrapper(gym.make("LorlEnv-v0"), task)
            # Important: use a different seed for each environment
            # wrapped_env.seed(seed + rank)
            return wrapped_env

        return _init

    env_fns = [make_env("open drawer")] * 3
    vec_env = DummyVecEnv(env_fns)
    # print(vec_env.reset())

#     def get_mean_std(env, use_state=True, steps=10000):
#         """
#         Calculate mean and std of Lorl env states if
#         we haven't already calculated them
#         """
#         if use_state:
#             if os.path.isfile("static/lorl_state_stats.pkl"):
#                 x = pickle.load(open("static/lorl_state_stats.pkl", "rb"))
#             else:
#                 states = []
#                 obs, _ = env.reset()
#                 states.append(env.sim.data.qpos[:])
#                 for _ in tqdm(range(steps)):
#                     action = env.action_space.sample()
#                     obs, reward, done, info = env.step(action)
#                     states.append(env.sim.data.qpos[:])
#                     if done:
#                         obs, _ = env.reset()
#                         states.append(env.sim.data.qpos[:])
#                 states = np.array(states)
#                 x = {"mean": states.mean(0), "std": states.std(0)}
#                 os.makedirs("static/", exist_ok=True)
#                 pickle.dump(x, open("static/lorl_state_stats.pkl", "wb"))

#         else:
#             if os.path.isfile("static/lorl_img_stats.pkl"):
#                 x = pickle.load(open("static/lorl_img_stats.pkl", "rb"))
#             else:
#                 states = []
#                 obs, _ = env.reset()
#                 states.append(np.moveaxis(obs, 2, 0))
#                 for _ in tqdm(range(steps)):
#                     action = env.action_space.sample()
#                     obs, reward, done, info = env.step(action)
#                     states.append(np.moveaxis(obs, 2, 0))
#                     if done:
#                         obs, _ = env.reset()
#                         states.append(np.moveaxis(obs, 2, 0))
#                 states = np.array(states)
#                 x = {"mean": states.mean(0), "std": states.std(0)}
#                 os.makedirs("static/", exist_ok=True)
#                 pickle.dump(x, open("static/lorl_img_stats.pkl", "wb"))

#         return x["mean"], x["std"]

#     print(get_mean_std(env))
