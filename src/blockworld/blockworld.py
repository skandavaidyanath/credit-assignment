## Code and environment adapted from https://github.com/Stanford-ILIAD/PantheonRL/blob/master/pantheonrl/envs/blockworldgym/blockworld.py

## modified version of BlockWorld for a single-agent setting.


import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from blockworld.blockworld_utils import (
    generate_random_world,
    place,
    gravity,
    matches,
)


class BlockWorldSingleEnv(gym.Env):
    """
    Single agent block-world environment
    Trying to reach the goal grid
    """

    BLUE = 1
    RED = 2

    def __init__(self, gridlen, num_blocks, num_colors, blocksize):
        super().__init__()
        self.gridlen = gridlen
        self.num_blocks = num_blocks
        self.num_colors = num_colors
        self.blocksize = blocksize  ## actual blocksize is blocksize x 1
        if self.blocksize == 1:
            self.num_orientations = 1
        else:
            self.num_orientations = 2
        gridformat = [self.num_colors + 1] * self.gridlen * self.gridlen * 2

        self.observation_space = gym.spaces.MultiDiscrete(gridformat)
        self.action_space = gym.spaces.Discrete(
            self.gridlen * self.num_orientations * self.num_colors
        )

    def get_state(self):
        return np.concatenate(
            [self.grid.copy().flatten(), self.goal.copy().flatten()]
        )

    def reset(self):
        while True:
            ## sometimes the random gridworld geenration gets stuck in impossible situations where it is unable to add all blocks
            ## in this case we need to re-generate the grid
            self.goal = generate_random_world(
                self.gridlen, self.num_blocks, self.num_colors, self.blocksize
            )
            if self.goal is not None:
                break
        self.grid = np.zeros((self.gridlen, self.gridlen))
        self.num_steps = 0

        return self.get_state()

    def step(self, action):
        self.num_steps += 1
        done = False
        reward = 0
        is_success = False
        if self.num_steps == self.num_blocks:
            done = True
        x, orientation, color = self.decode_action(action)

        y = gravity(self.grid, orientation, x, self.blocksize)
        if y != -1:
            place(self.grid, x, y, color, orientation, self.blocksize)
        if done:
            reward, is_success = self.get_reward()
        return (
            self.get_state(),
            reward,
            done,
            {"success": is_success},
        )

    def get_reward(self):
        # we use F1 score which is 2 * precision * recall / (precision + recall)
        # also = 2 * truepos / (selected + relevant)
        truepos = matches(self.grid, self.goal)
        selected = np.count_nonzero(self.grid)
        relevant = np.count_nonzero(self.goal)
        f_score = 2 * truepos / (selected + relevant)
        return f_score, (f_score == 1.0)

    def render(self, filename):
        cmap = colors.ListedColormap(["white", "blue", "red"])
        plt.pcolor(self.grid[::-1], cmap=cmap, edgecolors="k", linewidths=3)
        plt.savefig(f"{filename}")
        plt.close()

    def encode_action(self, x, orientation, color):
        assert (
            x < self.gridlen
            and orientation < self.num_orientations
            and color
            in [BlockWorldSingleEnv.BLUE, BlockWorldSingleEnv.RED][
                : self.num_colors
            ]
        )
        color -= 1
        action = x
        action *= self.num_orientations
        action += orientation
        action *= self.num_colors
        action += color
        assert action in self.action_space
        return action

    def decode_action(self, action_int):
        assert action_int in self.action_space
        color = action_int % self.num_colors
        color += 1
        assert (
            color
            in [BlockWorldSingleEnv.BLUE, BlockWorldSingleEnv.RED][
                : self.num_colors
            ]
        )
        action_int = action_int // self.num_colors
        orientation = action_int % self.num_orientations
        assert orientation < self.num_orientations
        action_int = action_int // self.num_orientations
        x = action_int
        assert x < self.gridlen
        action_int = action_int // self.gridlen
        assert action_int == 0
        return x, orientation, color


if __name__ == "__main__":
    env = BlockWorldSingleEnv(
        gridlen=4, num_blocks=5, num_colors=2, blocksize=2
    )
    print(env.reset())
