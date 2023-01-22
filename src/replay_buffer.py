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
        self.terminals = []
    
    def clear(self):
        del self.states[:]
        del self.messages[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]
        
    def __len__(self):
        return len(self.states)
    