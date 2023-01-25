import numpy as np


class ReplayBuffer:
    """Simple DQN Replay Buffer to store (s, a, r, s') tuples."""
    def __init__(self, state_dim, act_dim, max_buffer_size=1e6):
        self.states = np.empty((max_buffer_size, state_dim), dtype=np.float32)
        self.actions = np.empty((max_buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_buffer_size, 1), dtype=np.float32)
        self.next_states = np.empty((max_buffer_size, state_dim), dtype=np.float32)
        self.dones = np.empty((max_buffer_size, 1), dtype=np.float32)

        self.size = 0
        self.curr_ind = 0
        self.max_buffer_size = int(max_buffer_size)

    def add_transition(self, state, action, reward, next_state, done):
        self.states[self.curr_ind] = state
        self.actions[self.curr_ind] = action
        self.rewards[self.curr_ind] = np.array(reward)
        self.next_states[self.curr_ind] = next_state
        self.dones[self.curr_ind] = np.array(float(done))

        self.size = min(self.size + 1, self.max_buffer_size)
        self.curr_ind = (self.curr_ind + 1) % self.max_buffer_size

    def sample(self, batch_size):
        if batch_size > self.size:
            raise Exception("Not enough transitions in buffer to satisfy the provided batch size!")
        inds = np.random.choice(self.size, size=batch_size, replace=False)
        return self.states[inds], self.actions[inds], self.rewards[inds], self.next_states[inds], self.dones[inds]


if __name__ == '__main__':
    buffer = ReplayBuffer(2, 1, 10)
    for _ in range(100):
        state = np.random.random((2, ))
        action = np.random.random((1, ))
        reward = np.random.random((1, ))[0]
        next_state = np.random.random((2, ))
        done = 0.0
        buffer.add_transition(state, action, reward, next_state, done)

    s, a, r, ns, d = buffer.sample(5)
