import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dualdice.return_buffer import digitize_returns
from utils import weight_reset, get_grad_norm
import warnings


class ReturnPredictor(nn.Module):
    """
    Class that learns to predit the probability
    of return R under policy \pi given a state s
    """

    def __init__(
        self,
        state_dim,
        quantize=False,
        num_classes=10,  # this is used only when quantize is True
        n_layers=2,
        hidden_size=64,
        activation_fn="relu",
        dropout_p=0,
        batch_size=64,
        lr=3e-4,
        device="cpu",
        normalize_inputs=True,
        normalize_targets=False,
        max_grad_norm=None,
    ):

        super(ReturnPredictor, self).__init__()

        self.state_dim = state_dim

        self.quantize = quantize
        if not quantize:
            num_classes = 1
        self.num_classes = num_classes

        self.normalize_inputs = normalize_inputs
        self.normalize_targets = normalize_targets
        self.max_grad_norm = max_grad_norm

        if activation_fn == "tanh":
            activation = nn.Tanh
        elif activation_fn == "relu":
            activation = nn.ReLU
        else:
            raise NotImplementedError

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(state_dim, hidden_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation())
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))

        if not layers:
            layers.append(nn.Linear(state_dim, num_classes))
        else:
            layers.append(nn.Linear(hidden_size, num_classes))

        if not quantize:
            self.log_std = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.log_std = None

        self.net = nn.Sequential(*layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Normalization statistics for input, used only if self.normalize = True
        self.input_mean = torch.zeros((state_dim,), device=device)
        self.input_std = torch.ones((state_dim,), device=device)

        # Normalization statistics for target, used only if self.normalize_targets = True
        self.target_mean = torch.zeros((1,), device=device)
        self.target_std = torch.zeros((1,), device=device)

        # used to quantize returns during test time
        self.bins = None

    @property
    def std(self):
        if self.quantize:
            raise ValueError("Calling std() on quantized Return predictor!")
        else:
            return torch.exp(self.log_std)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif isinstance(layer, torch.nn.Sequential):
                layer.apply(weight_reset)

    def update_norm_stats(self, mean, std, refresh=True):
        if refresh:  # re-calculate stats each time we train model
            self.input_mean = torch.from_numpy(mean).to(self.device).float()
            self.input_std = torch.from_numpy(std).to(self.device).float()
        else:
            raise NotImplementedError

    def update_target_stats(self, mean, std, refresh=True):
        if refresh:
            self.target_mean = torch.from_numpy(mean).to(self.device)
            self.target_std = torch.from_numpy(std).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        """
        forward pass a bunch of inputs into the model
        """
        if self.normalize_inputs:
            inputs = (inputs - self.input_mean) / (self.input_std + 1e-6)

        out = self.net(inputs)  # B x 1

        return out

    def update(self, buffer):
        train_dataloader, val_dataloader, self.bins = buffer.get_dataloader(
            self.batch_size
        )

        losses, metrics = [], []

        for states, returns in train_dataloader:
            loss, metric = self.train_step(states, returns)
            losses.append(loss)
            metrics.append(metric)

        if self.quantize:
            results = {
                "ret_train_loss": np.mean(losses),
                "ret_train_acc": np.mean(metrics),
            }

        else:
            results = {
                "ret_train_loss": np.mean(losses),
                "ret_train_logprobs": np.mean(metrics),
            }

        if val_dataloader is not None:
            val_results = self.validate(val_dataloader)
            results.update(val_results)
        return results

    def train_step(self, states, returns):
        states = states.to(self.device)
        returns = returns.to(self.device)
        if self.normalize_targets and not self.quantize:
            returns = (returns - self.target_mean) / (self.target_std + 1e-6)
        preds = self.forward(states)

        if self.quantize:
            # return targets are already in the correct class form
            returns = returns.flatten().long()
            loss = F.cross_entropy(preds, returns)
            preds = preds.argmax(-1)
            metric = torch.sum(preds == returns) / len(preds)
        else:
            # return targets are real numbers
            std = self.std.to(self.device)
            dists = Normal(preds, std)
            loss = F.gaussian_nll_loss(preds, returns, dists.variance)
            metric = dists.log_prob(returns).mean()

        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.max_grad_norm
            )

        # if get_grad_norm(self.net) > 100.0 and not self.max_grad_norm:
        #     warnings.warn("Return model grad norm is over 100 but is not being clipped!")

        self.optimizer.step()

        return loss.item(), metric.item()

    @torch.no_grad()
    def validate(self, val_dataloader):
        losses, metrics = [], []

        for states, returns in val_dataloader:
            states = states.to(self.device)
            returns = returns.to(self.device)

            preds = self.forward(states)

            if self.quantize:
                returns = returns.flatten().long()
                loss = F.cross_entropy(preds, returns).item()
                preds = preds.argmax(-1)
                accuracy = torch.sum(preds == returns) / len(preds)
                losses.append(loss)
                metrics.append(accuracy)
            else:
                if self.normalize_targets:
                    returns = (returns - self.target_mean) / (
                        self.target_std + 1e-6
                    )

                dists = Normal(preds, self.std)
                loss = F.gaussian_nll_loss(
                    preds, returns, dists.variance
                ).item()
                log_probs = dists.log_prob(returns)
                mean_logprobs = log_probs.mean().item()
                losses.append(loss)
                metrics.append(mean_logprobs)

        if self.quantize:
            return {
                "ret_val_loss": np.mean(losses),
                "ret_val_acc": np.mean(metrics),
            }
        else:
            return {
                "ret_val_loss": np.mean(losses),
                "ret_val_logprobs": np.mean(metrics),
            }

    @torch.no_grad()
    def get_return_probs(self, states, returns):
        """
        get the probability of returns for a batch of states
        """
        states = states.to(self.device)  # B, D_s
        returns = returns.to(self.device)  # B, 1
        preds = self.forward(states)  # B, num_classes
        if self.quantize:
            quantized_returns = digitize_returns(returns, self.bins)
            return_probs = torch.gather(preds, -1, quantized_returns)
        else:
            if self.normalize_targets:
                returns = (returns - self.target_mean) / (
                    self.target_std + 1e-6
                )

            # returns are real numbers
            std = self.std.to(self.device)
            dists = Normal(preds, std)
            return_probs = torch.exp(dists.log_prob(returns))
        return return_probs

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)
