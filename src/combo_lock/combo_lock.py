# Code adapted from https://github.com/microsoft/Intrepid/

import numpy as np
import math
import gym
from gym import spaces

# Generate Hadamhard matrix of atleast a given size using Sylvester's method
def generated_hadamhard_matrix(lower_bound):

    dim = 1
    h = np.array([[1.0]], dtype=float)

    while dim < lower_bound:
        h = np.block([[h, h], [h, -h]])
        dim = 2 * dim

    # Trim the columns of the matrix to match the lower bound
    return h[:, :lower_bound]


# Size of the smallest Hadamhard matrix which is greater than lower bound, as generated by Sylvester's method.
def get_sylvester_hadamhard_matrix_dim(lower_bound):
    return int(math.pow(2, math.ceil(math.log(lower_bound, 2))))


class DiabolicalCombinationLock(gym.Env):

    env_name = "diabcombolock"
    NONE, BERNOULLI, GAUSSIAN, HADAMHARD, HADAMHARDG = range(5)

    def __init__(self, config):
        """
        :param config: Configuration of the environment
        """
        self.noise_type = self.get_noise(config.env.noise_type)

        self.horizon = config.env.horizon
        self.swap = config.env.swap_prob
        self.num_actions = config.env.num_actions
        self.optimal_reward = config.env.optimal_reward
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(config.training.seed)
        self.anti_shaping_reward = config.env.anti_shaping_reward
        self.anti_shaping_reward2 = config.env.anti_shaping_reward2

        assert (
            self.anti_shaping_reward * 0.5
            < self.optimal_reward * self.optimal_reward_prob
        ), (
            "Anti shaping reward shouldn't exceed optimal reward which is %r"
            % (self.optimal_reward * self.optimal_reward_prob)
        )

        assert self.num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, self.num_actions))
        self.spawn_prob = config.env.spawn_prob

        self.opt_a = self.rng.randint(
            low=0, high=self.num_actions, size=self.horizon
        )
        self.opt_b = self.rng.randint(
            low=0, high=self.num_actions, size=self.horizon
        )

        if self.noise_type == DiabolicalCombinationLock.NONE:

            # No noise is added and we return the underlying state directly which is
            # the state type and the time
            self.dim = 2

        elif self.noise_type == DiabolicalCombinationLock.GAUSSIAN:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 4

        elif self.noise_type == DiabolicalCombinationLock.BERNOULLI:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = (
                self.horizon + 4 + self.horizon
            )  # Add noise of length horizon

        elif self.noise_type == DiabolicalCombinationLock.HADAMHARD:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif self.noise_type == DiabolicalCombinationLock.HADAMHARDG:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        ############

        self.curr_state = None  # Current state
        self.timestep = -1  # Current time step
        self._eps_return = 0.0  # Current episode return

        #######

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dim,)
        )
        self.action_space = spaces.Discrete(self.num_actions)

    def reset(self):
        """
        :return:
            obs: Agent observation. No assumption made on the structure of observation.
        """

        self.curr_state = self.start()
        self.timestep = 0
        obs = self.make_obs(self.curr_state)

        self._eps_return = 0.0

        return obs

    def step(self, action):
        """
        :param action:
        :return:
            obs:        Agent observation. No assumption made on the structure of observation.
            reward:     Reward received by the agent. No Markov assumption is made.
            done:       True if the episode has terminated and False otherwise.
            info:       Dictionary containing relevant information such as latent state, etc.
        """

        horizon = self.get_horizon()

        if self.curr_state is None or self.timestep < 0:
            raise AssertionError("Environment not reset")

        if self.timestep > horizon:
            raise AssertionError(
                "Cannot take more actions than horizon %d" % horizon
            )

        new_state = self.transition(self.curr_state, action)
        recv_reward = self.reward(self.curr_state, action, new_state)
        obs = self.make_obs(new_state)

        self.curr_state = new_state
        self.timestep += 1

        self._eps_return += recv_reward

        done = self.timestep == horizon

        info = {}

        return obs, recv_reward, done, info

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        raise NotImplementedError

    @staticmethod
    def get_noise(noise_type_str):

        if noise_type_str == "none":
            return DiabolicalCombinationLock.NONE
        elif noise_type_str == "bernoulli":
            return DiabolicalCombinationLock.BERNOULLI

        elif noise_type_str == "gaussian":
            return DiabolicalCombinationLock.GAUSSIAN

        elif noise_type_str == "hadamhard":
            return DiabolicalCombinationLock.HADAMHARD

        elif noise_type_str == "hadamhardg":
            return DiabolicalCombinationLock.HADAMHARDG

        else:
            raise AssertionError("Unhandled noise type %r" % noise_type_str)

    def is_episodic(self):
        """
        :return: Return True or False, True if the environment is episodic and False otherwise.
        """
        return True

    def get_env_name(self):
        return self.env_name

    def get_actions(self):
        return self.actions

    def get_num_actions(self):
        return self.num_actions

    def get_horizon(self):
        return self.horizon

    def transition(self, x, a):

        b = self.rng.binomial(1, self.swap)

        if x[0] == 0 and a == self.opt_a[x[1]]:
            if b == 0:
                return 0, x[1] + 1
            else:
                return 1, x[1] + 1
        if x[0] == 1 and a == self.opt_b[x[1]]:
            if b == 0:
                return 1, x[1] + 1
            else:
                return 0, x[1] + 1
        else:
            return 2, x[1] + 1

    def make_obs(self, x):

        if self.noise_type == DiabolicalCombinationLock.NONE:

            v = np.array(x)

        elif self.noise_type == DiabolicalCombinationLock.BERNOULLI:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4 :] = self.rng.binomial(1, 0.5, self.horizon)

        elif self.noise_type == DiabolicalCombinationLock.GAUSSIAN:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

        elif self.noise_type == DiabolicalCombinationLock.HADAMHARD:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = np.matmul(self.hadamhard_matrix, v)

        elif self.noise_type == DiabolicalCombinationLock.HADAMHARDG:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
            v = np.matmul(self.hadamhard_matrix, v)

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        return v

    def start(self):
        # Start stochastically in one of the two live states
        toss_value = self.rng.binomial(1, self.spawn_prob)

        if toss_value == 0:
            return 0, 0
        elif toss_value == 1:
            return 1, 0
        else:
            raise AssertionError(
                "Toss value can only be 1 or 0. Found %r" % toss_value
            )

    def reward(self, x, a, next_x):

        # If the agent reaches the final live states then give it the optimal reward.
        if (x == (0, self.horizon - 1) and a == self.opt_a[x[1]]) or (
            x == (1, self.horizon - 1) and a == self.opt_b[x[1]]
        ):
            return self.optimal_reward * self.rng.binomial(
                1, self.optimal_reward_prob
            )

        # If reaching the dead state for the first time then give it a small anti-shaping reward.
        # This anti-shaping reward is anti-correlated with the optimal reward.
        if x is not None and next_x is not None:
            if x[0] != 2 and next_x[0] == 2:
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif x[0] != 2 and next_x[0] != 2:
                return -self.anti_shaping_reward2 / (self.horizon - 1)

        return 0


if __name__ == "__main__":
    from collections import namedtuple

    Config = namedtuple("Config", ["env", "training"])
    Env = namedtuple(
        "Env",
        [
            "horizon",
            "swap_prob",
            "spawn_prob",
            "noise_type",
            "num_actions",
            "optimal_reward",
            "anti_shaping_reward",
            "anti_shaping_reward2",
        ],
    )
    Training = namedtuple("Training", ["seed"])

    training = Training(seed=0)
    env = Env(
        horizon=50,
        swap_prob=0.0,
        spawn_prob=0.5,
        noise_type="none",
        num_actions=5,
        optimal_reward=100,
        anti_shaping_reward=5,
        anti_shaping_reward2=5,
    )
    config = Config(env=env, training=training)

    diab_env = DiabolicalCombinationLock(config)

    print(diab_env.reset())
