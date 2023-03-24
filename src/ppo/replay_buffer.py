class RolloutBuffer:
    '''
        Rollout Buffer: Each element is a list of lists
        where inner lists corresponds to a single trajectory
    '''

    def __init__(self):
        self.states = []
        self.messages = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.returns = []
        self.terminals = []
        self.hindsight_logprobs = []

    def clear(self):
        del self.states[:]
        del self.messages[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.returns[:]
        del self.terminals[:]
        del self.hindsight_logprobs[:]

    def __len__(self):
        return len(self.states)
