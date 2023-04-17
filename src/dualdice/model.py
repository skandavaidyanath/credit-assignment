import numpy as np

import torch
import torch.nn as nn


class DualDICE(nn.Module):
    """
    Class that solves the DualDICE optimization
    problem and approximates the density ratio
    \pi/ h
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        f="square",
        n_layers=2,
        hidden_size=64,
        activation_fn="relu",
        dropout_p=0,
        batch_size=64,
        lr=3e-4,
        device="cpu",
        normalize_inputs=True,
    ):

        super(DualDICE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.f = f
        self.normalize_inputs = normalize_inputs

        if activation_fn == "tanh":
            activation = nn.Tanh
        elif activation_fn == "relu":
            activation = nn.ReLU
        else:
            raise NotImplementedError

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(state_dim + action_dim, hidden_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))

        if not layers:
            layers.append(nn.Linear(state_dim + action_dim, 1))
        else:
            layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.input_h_mean = torch.zeros((state_dim,), device=device)
        self.input_h_std = torch.ones((state_dim,), device=device)
        self.input_pi_mean = torch.zeros((state_dim,), device=device)
        self.input_pi_std = torch.ones((state_dim,), device=device)

    def _f(self, x):
        """
        Can implement different convex functions here like
        square or p-norm
        """
        if self.f == "square":
            return 0.5 * x**2
        else:
            raise NotImplementedError

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def update_norm_stats(self, h_mean, h_std, pi_mean, pi_std, refresh=True):
        if refresh:  # re-calculate stats each time we train model
            self.input_h_mean = torch.from_numpy(h_mean).to(self.device)
            self.input_h_std = torch.from_numpy(h_std).to(self.device)
            self.input_pi_mean = torch.from_numpy(pi_mean).to(self.device)
            self.input_pi_std = torch.from_numpy(pi_std).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        """
        forward pass a bunch of inputs into the model
        """
        if self.normalize_inputs:
            inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-6)

        out = self.net(inputs)  # B x 1

        return out

    def _get_batch(self, buffer):
        states = np.concatenate(buffer.states, 0)  # B, D_s
        actions = np.concatenate(buffer.actions, 0).reshape(-1, 1)  # B, D_a
        X = np.concatenate((states, actions), -1)  # B, D_s + D_a
        return torch.from_numpy(X).to(self.device)

    def update(self, buffer):
        train_dataloader, val_dataloader = buffer.get_dataloader(
            self.batch_size
        )

        losses = []

        for h_state_actions, pi_state_actions in train_dataloader:
            loss = self.train_step(h_state_actions, pi_state_actions)
            losses.append(loss)

        results = {"dd_train_loss": np.mean(losses)}

        val_results = self.validate(val_dataloader)
        results.update(val_results)
        return results

    def train_step(self, h_state_actions, pi_state_actions):
        h_state_actions = h_state_actions.to(self.device)
        pi_state_actions = pi_state_actions.to(self.device)

        h_preds = self.forward(h_state_actions)
        pi_preds = self.forward(pi_state_actions)

        loss = self._f(h_preds) - pi_preds

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validate(self, val_dataloader):
        losses = []
        for h_state_actions, pi_state_actions in val_dataloader:
            h_state_actions = h_state_actions.to(self.device)
            pi_state_actions = pi_state_actions.to(self.device)

            h_preds = self.forward(h_state_actions)
            pi_preds = self.forward(pi_state_actions)

            loss = self._f(h_preds) - pi_preds
            losses.append(loss.item())

        return {"dd_val_loss": np.mean(losses)}

    def get_density_ratios(self, states, actions):
        """
        get the hindsight values for a batch of state-actions
        """
        states = states.to(self.device)  # B, D_s
        actions = actions.to(self.device)  # B, D_a
        state_actions = torch.cat([states, actions], dim=-1)  # B, D_s + D_a
        ratios = self.forward(state_actions)  # B, 1
        return ratios

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)
