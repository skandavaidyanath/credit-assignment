import numpy as np

import torch
import torch.nn as nn
import warnings

from utils import weight_reset


class DualDICE(nn.Module):
    """
    Class that solves the DualDICE optimization
    problem and approximates the density ratio
    \pi/ h
    Model is S x A x R -> C
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        f="square",
        c=1,
        n_layers=2,
        hidden_size=64,
        activation_fn="relu",
        dropout_p=0,
        batch_size=64,
        lr=3e-4,
        device="cpu",
        normalize_inputs=False,
        normalize_return_inputs=False,
        max_grad_norm=None,
    ):
        super(DualDICE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.f = f
        self.c = c
        # standardize the entire input (state, action, return).
        self.normalize_inputs = normalize_inputs
        # standardize the return portion of the input.
        self.normalize_return_inputs = normalize_return_inputs or normalize_inputs
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device)

        layers = []

        if activation_fn == "tanh":
            activation = nn.Tanh
        elif activation_fn == "relu":
            activation = nn.ReLU
        else:
            raise NotImplementedError

        if n_layers == 0:
            hidden_size = state_dim + action_dim + 1
        for i in range(n_layers):
            if i == 0:
                layers.append(
                    nn.Linear(state_dim + action_dim + 1, hidden_size)  # +1 for return
                )
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))

        layers.append(nn.Linear(hidden_size, 1))
        # this makes the output positive and within a stable range
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size

        self.state_mean = torch.zeros((state_dim,), device=device)
        self.state_std = torch.ones((state_dim,), device=device)
        self.action_mean = torch.zeros((action_dim,), device=device)
        self.action_std = torch.ones((action_dim,), device=device)
        self.return_mean = torch.zeros((1,), device=device)
        self.return_std = torch.ones((1,), device=device)

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
            elif isinstance(layer, torch.nn.Sequential):
                layer.apply(weight_reset)

    def update_norm_stats(
        self,
        state_mean,
        state_std,
        action_mean,
        action_std,
        return_mean,
        return_std,
        refresh=True,
    ):
        if refresh:  # re-calculate stats each time we train model
            self.state_mean = torch.from_numpy(state_mean).to(self.device).float()
            self.state_std = torch.from_numpy(state_std).to(self.device).float()
            self.action_mean = torch.from_numpy(action_mean).to(self.device).float()
            self.action_std = torch.from_numpy(action_std).to(self.device).float()
            self.return_mean = torch.from_numpy(return_mean).to(self.device).float()
            self.return_std = torch.from_numpy(return_std).to(self.device).float()
        else:
            raise NotImplementedError

    def forward(self, states, actions, returns):
        """
        forward pass a bunch of inputs into the model
        """
        if self.normalize_return_inputs:
            returns = (returns - self.return_mean) / (self.return_std + 1e-6)
        if self.normalize_inputs:
            states = (states - self.state_mean) / (self.state_std + 1e-6)
            actions = (actions - self.action_mean) / (self.action_std + 1e-6)

        inputs = torch.concat([states, actions, returns], dim=-1).float()
        out = self.net(inputs)  # B x 1
        out = out * self.c

        return out

    def update(self, buffer):
        train_dataloader, val_dataloader = buffer.get_dataloader(self.batch_size)

        losses = []

        for states, h_a, h_r, pi_a, pi_r in train_dataloader:
            loss = self.train_step(states, h_a, h_r, pi_a, pi_r)
            losses.append(loss)

        results = {"dd_train_loss": np.mean(losses)}

        if val_dataloader is not None:
            val_results = self.validate(val_dataloader)
            results.update(val_results)
        return results

    def train_step(self, states, h_a, h_r, pi_a, pi_r):
        states = states.to(self.device)
        h_a = h_a.to(self.device)
        h_r = h_r.to(self.device)
        pi_a = pi_a.to(self.device)
        pi_r = pi_r.to(self.device)

        h_preds = self.forward(states, h_a, h_r)
        pi_preds = self.forward(states, pi_a, pi_r)

        loss = torch.mean(self._f(h_preds)) - torch.mean(pi_preds)

        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        # if get_grad_norm(self.net) > 100.0 and not self.max_grad_norm:
        #     warnings.warn(
        #         "DD model grad norm is over 100 but is not being clipped!"
        #     )

        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validate(self, val_dataloader):
        losses = []
        for states, h_a, h_r, pi_a, pi_r in val_dataloader:
            states = states.to(self.device)
            h_a = h_a.to(self.device)
            h_r = h_r.to(self.device)
            pi_a = pi_a.to(self.device)
            pi_r = pi_r.to(self.device)

            h_preds = self.forward(states, h_a, h_r)
            pi_preds = self.forward(states, pi_a, pi_r)

            loss = torch.mean(self._f(h_preds)) - torch.mean(pi_preds)
            losses.append(loss.item())

        return {"dd_val_loss": np.mean(losses)}

    @torch.no_grad()
    def get_density_ratios(self, states, actions, returns):
        """
        get the hindsight values for a batch of state-actions
        """
        states = states.to(self.device)  # B, D_s
        actions = actions.to(self.device)  # B, D_a
        returns = returns.to(self.device)  # B, 1
        ratios = self.forward(states, actions, returns)  # B, 1
        return ratios

    def save(self, checkpoint_path, args):
        torch.save(
            {
                "model": self.net.state_dict(),
                "args": args,
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "return_mean": self.return_mean,
                "return_std": self.return_std,
            },
            checkpoint_path,
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint["model"])
        self.state_mean = checkpoint["state_mean"]
        self.state_std = checkpoint["state_std"]
        self.action_mean = checkpoint["action_mean"]
        self.action_std = checkpoint["action_std"]
        self.return_mean = checkpoint["return_mean"]
        self.return_std = checkpoint["return_std"]
