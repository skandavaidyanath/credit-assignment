import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCriticLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
        activation_fn="tanh",
    ):
        """
        Default Actor Critic with a LSTM Critic function instead of FFN
        """
        super(ActorCriticLSTM, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        if activation_fn == "tanh":
            activation_fn = nn.Tanh()
        elif activation_fn == "relu":
            activation_fn = nn.ReLU()
        else:
            raise NotImplementedError()

        assert (
            n_layers > 0
        ), "Must have at least one hidden layer for LSTM critic model!"

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(activation_fn)
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation_fn)

        if continuous:
            layers.append(nn.Linear(hidden_size, action_dim))
        else:
            layers.append(nn.Linear(hidden_size, action_dim))
            layers.append(
                nn.Softmax(dim=-1),
            )
        self.actor = nn.Sequential(*layers)

        self.critic = nn.Sequential(
            nn.LSTM(
                input_dim,
                hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            ),
            nn.Linear(hidden_size, 1),
        )

    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete policy!")
        else:
            return torch.exp(self.log_std)

    def forward(self, state):
        if self.continuous:
            action_mean = self.actor(state)
            std = torch.diag(self.std)
            dist = MultivariateNormal(action_mean, scale_tril=std)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        return dist

    def act(self, state, greedy=False):
        dist = self.forward(state)
        if greedy:
            action = dist.mode
        else:
            action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        dist = self.forward(state)
        # For Single Action Environments.
        if self.continuous and self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
