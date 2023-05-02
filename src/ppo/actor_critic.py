import torch
import torch.nn as nn
from utils import model_init
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, continuous=False,
                 cnn_base=None, n_layers=2, hidden_size=64,
                 activation_fn="tanh"):
        super(ActorCritic, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        if continuous:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim), requires_grad=True
            )

        if cnn_base:
            # if a CNN base is passed in, then use that instead of an MLP.
            assert isinstance(cnn_base, CNNBase)
            assert cnn_base.hidden_size == hidden_size
            self.encoder = cnn_base
        else:
            if activation_fn == "tanh":
                activation_fn = nn.Tanh()
            elif activation_fn == "relu":
                activation_fn = nn.ReLU()
            else:
                raise NotImplementedError()

            if n_layers == 0:
                self.encoder = nn.Sequential(*[])
                hidden_size = input_dim
            else:
                layers = []
                for i in range(n_layers):
                    if i == 0:
                        layers.append(nn.Linear(input_dim, hidden_size))
                        layers.append(activation_fn)
                    else:
                        layers.append(nn.Linear(hidden_size, hidden_size))
                        layers.append(activation_fn)
                self.encoder = nn.Sequential(*layers)

        # Critic and actor heads. CNN base models use a orthogonal initialization.
        if cnn_base:
            critic_init_ = lambda m: model_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
            actor_init_ = lambda m: model_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        else:
            critic_init_ = lambda m: m
            actor_init_ = lambda m: m

        self.critic = critic_init_(nn.Linear(hidden_size, 1))
        if continuous:
            self.actor = actor_init_(nn.Linear(hidden_size, action_dim))
        else:
            self.actor = nn.Sequential(
                actor_init_(nn.Linear(hidden_size, action_dim)), nn.Softmax(dim=-1)
            )


    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete policy!")
        else:
            return torch.exp(self.log_std)


    def forward(self, state):
        encoding = self.encoder(state)
        if self.continuous:
            action_mean = self.actor(encoding)
            std = torch.diag(self.std)
            dist = MultivariateNormal(action_mean, scale_tril=std)
        else:
            action_probs = self.actor(encoding)
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
        # state_values = self.critic(state)
        encoding = self.encoder(state)
        state_values = self.critic(encoding)

        return action_logprobs, state_values, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()
        self.hidden_size = hidden_size

        init_ = lambda m: model_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn_base = nn.Sequential(
            init_(nn.Conv2d(in_channels=num_inputs,
                            out_channels=32, kernel_size=8,
                            stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                            out_channels=64, kernel_size=4,
                            stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=64,
                            out_channels=32,
                            kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        self.train()

    def forward(self, inputs):
        x = self.cnn_base(inputs / 255.0)
        return x
