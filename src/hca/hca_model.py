import torch
import torch.nn as nn


class HCAModel(nn.Module):
    """
    HCA model to predict action probabilities conditioned on returns and state
    """

    def __init__(self, state_dim, action_dim, n_layers=2, hidden_size=64):
        super(HCAModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(state_dim, hidden_size))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Tanh())
        if not layers:
            layers.append(nn.Linear(state_dim, action_dim))
        else:
            layers.append(nn.Linear(hidden_size, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        forward pass a bunch of inputs into the model
        """
        return self.net(inputs)

    def train_step(self, states, actions, optimizer, loss_fn, device):
        states = states.to(device)
        actions = actions.to(device).flatten()
        preds = self.forward(states)
        loss = loss_fn(preds, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_hindsight_values(self, inputs, actions):
        """
        get the hindsight values for a batch of actions
        """
        out = self.forward(inputs)  # B x A
        actions = actions.reshape(-1, 1)
        return out.gather(1, actions)  # B,

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)
