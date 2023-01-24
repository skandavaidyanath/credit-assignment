import numpy as np
import gym
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

from tabulate import tabulate

TILE_MAPPING = {".": 0, "S": 1, "G": 2, "*": 3, "R": 4, "F": 5, "A": 6}
INVERSE_TILE_MAPPING = {0: ".", 1: "S", 2: "G", 3: "*", 4: "R", 5: "F", 6: "A"}
ACTION_MAPPING = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}
REWARD_MAPPING = {"F": -100, "*": +20}


class GridWorld(Env):
    def __init__(self, filename, sparse=True, max_steps=50):
        gridmap = open(filename, "r").read()

        self.sparse = sparse
        self.max_steps = max_steps
        self.current_steps = 0

        # process gridworld
        self.process(gridmap)
        self.current_map = self.gridmap.copy()

        # set agent to start state
        self.start_location = np.where(self.gridmap == TILE_MAPPING["S"])
        self.start_location = np.array(
            [self.start_location[0][0], self.start_location[1][0]]
        )
        self.agent_location = self.start_location.copy()

        # set episode reward to 0
        self.episode_reward = 0

        # observation and action spaces
        self.observation_space = MultiDiscrete(7 * np.ones(self.R * self.C))
        self.action_space = Discrete(4)

    def process(self, gridmap):
        rows = gridmap.split("\n")
        tiles = [row.split() for row in rows]
        tiles = [[TILE_MAPPING[elem] for elem in row] for row in tiles]

        self.gridmap = np.array(tiles)
        self.R, self.C = self.gridmap.shape

    def get_state(self):
        map = self.current_map.copy()
        map[self.agent_location[0], self.agent_location[1]] = TILE_MAPPING["A"]
        return map.flatten()

    def reset(self):
        self.current_map = self.gridmap.copy()
        # set agent to start state
        self.agent_location = self.start_location.copy()
        self.current_steps = 0
        self.episode_reward = 0
        return self.get_state()

    def next_state(self, action):
        next_state = self.agent_location + ACTION_MAPPING[action]
        if (0 <= next_state[0] < self.R) and (0 <= next_state[1] < self.C):
            self.agent_location = next_state
            return self.current_map[next_state[0], next_state[1]]
        return self.current_map[self.agent_location[0], self.agent_location[1]]

    def step(self, action):
        assert action in self.action_space
        next_state = self.next_state(action)
        self.current_steps += 1

        done = self.current_steps >= self.max_steps

        if next_state in [TILE_MAPPING["."], TILE_MAPPING["S"]]:
            reward = -1
        if next_state == TILE_MAPPING["R"]:
            reward = -1
            self.agent_location = self.start_location
        if next_state == TILE_MAPPING["F"]:
            reward = REWARD_MAPPING["F"]
            done = True
        if next_state == TILE_MAPPING["*"]:
            # remove the diamond from the location
            self.current_map[
                self.agent_location[0], self.agent_location[1]
            ] = TILE_MAPPING["."]
            reward = REWARD_MAPPING["*"] - 1
        if next_state == TILE_MAPPING["G"]:
            done = True
            reward = -1

        if self.sparse:
            self.episode_reward += reward
            reward = 0
            if done:
                return (
                    self.get_state(),
                    self.episode_reward,
                    done,
                    {"success": 0},
                )

        return self.get_state(), reward, done, {"success": 0}

    def render(self):
        out = [
            INVERSE_TILE_MAPPING[self.current_map[i][j]]
            for i in range(self.R)
            for j in range(self.C)
        ]
        out = np.array(out).reshape(self.R, self.C)
        out[self.agent_location[0], self.agent_location[1]] = "A"
        print(tabulate(out, tablefmt="fancy_grid"))


if __name__ == "__main__":
    g = GridWorld("maps/test.txt", sparse=False)
