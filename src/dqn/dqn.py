import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers=2, hidden_size=64, exploration="eps-greedy", eps=0.1):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim

        layers = []
        inp_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(inp_dim, hidden_size))
            layers.append(nn.Tanh())
            inp_dim = hidden_size
        layers.append(nn.Linear(inp_dim, action_dim))
        self.q_net = nn.Sequential(*layers)

        self.exploration = exploration
        self.eps = eps

    def forward(self, state):
        q_vals = self.q_net(state)
        return q_vals

    def act(self, state, greedy=False):
        q_vals = self.forward(state).reshape(-1, self.action_dim)
        greedy_acts = torch.argmax(q_vals, dim=-1)
        batch_size = q_vals.shape[0]
        if greedy:
            return greedy_acts
        if self.exploration == "eps-greedy":
            random_acts = torch.distributions.Categorical(logits=torch.ones_like(q_vals)).sample()
            take_random = (torch.rand(batch_size) < self.eps).int()
            acts = take_random * random_acts + (1 - take_random) * greedy_acts
            return acts
        else:
            raise NotImplementedError


if __name__ == '__main__':
    q_net = QNetwork(5, 10)
    states = torch.randn((4, 5))
    a = q_net.act(states)
    b = 2



