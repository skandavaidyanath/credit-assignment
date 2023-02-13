import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F


class HCAModel(nn.Module):
    """
    HCA model to predict action probabilities conditioned on returns and state
    """

    def __init__(self, state_dim, action_dim, continuous=False, n_layers=2,
                 hidden_size=64, activation_fn='tanh', dropout_p=None):
        super(HCAModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        if activation_fn == "tanh":
            activation_cls = nn.Tanh
        elif activation_fn == "relu":
            activation_cls = nn.ReLU
        else:
            raise NotImplementedError

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(state_dim, hidden_size))
                layers.append(activation_cls())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation_cls())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
        if not layers:
            layers.append(nn.Linear(state_dim, action_dim))
        else:
            layers.append(nn.Linear(hidden_size, action_dim))
        if not continuous:
            layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

        if continuous:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim), requires_grad=True
            )
        else:
            self.log_std = None

    def forward(self, inputs):
        """
        forward pass a bunch of inputs into the model
        """
        out = self.net(inputs)
        if self.continuous:
            std = torch.diag(self.std)
            dist = MultivariateNormal(out, scale_tril=std)
        else:
            dist = Categorical(out)
        return out, dist

    def train_step(self, states, actions, optimizer, device):
        states = states.to(device)
        actions = actions.to(device)
        preds, dist = self.forward(states)
        if self.continuous:
            loss = F.gaussian_nll_loss(preds, actions, dist.variance)
        else:
            loss = F.cross_entropy(preds, actions.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_hindsight_values(self, inputs, actions):
        """
        get the hindsight values for a batch of actions
        """
        out, dist = self.forward(inputs)
        if self.continuous:  # B x A
            log_probs = dist.log_prob(actions).reshape(-1, 1)
            return log_probs.exp()
        else:
            actions = actions.reshape(-1, 1).long()
            return out.gather(1, actions)  # B,

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)

    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete policy!")
        else:
            return torch.exp(self.log_std)
