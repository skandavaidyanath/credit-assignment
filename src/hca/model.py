import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
import numpy as np
from utils import weight_reset, get_grad_norm
import warnings

class HCAModel(nn.Module):
    """
    HCA model to predict action probabilities conditioned on returns and state
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
        activation_fn="relu",
        dropout_p=0,
        batch_size=64,
        lr=3e-4,
        device="cpu",
        normalize_inputs=True,
        max_grad_norm=None,
        weight_training_samples=False,
        noise_std=None,
    ):
        super(HCAModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim if continuous else 1
        self.continuous = continuous
        self.normalize_inputs = normalize_inputs
        self.max_grad_norm = max_grad_norm
        self.noise_std = noise_std

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
            layers.append(nn.Linear(state_dim, action_dim))
        else:
            layers.append(nn.Linear(hidden_size, action_dim))

        self.net = nn.Sequential(*layers).to(device)

        if continuous:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim), requires_grad=True
            )
        else:
            self.log_std = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.weight_training_samples = weight_training_samples

        self.input_mean = torch.zeros((state_dim,), device=device)
        self.input_std = torch.ones((state_dim,), device=device)

    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete HCA function!")
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
            self.input_mean = torch.from_numpy(mean).to(self.device)
            self.input_std = torch.from_numpy(std).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, inputs, add_noise=False):
        """
        forward pass a bunch of inputs into the model
        """
        if self.normalize_inputs:
            inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-6)

        out = self.net(inputs)
        if self.noise_std and add_noise:
            # print(out.abs().max())
            noise = torch.normal(
                mean=0.0, std=self.noise_std, size=out.shape
            ).to(out.device)
            out += noise
        if self.continuous:
            std = torch.diag(self.std)
            dist = MultivariateNormal(out, scale_tril=std)
        else:
            dist = Categorical(logits=out)
        return out, dist

    def update(self, buffer):
        train_dataloader, val_dataloader = buffer.get_dataloader(
            self.batch_size, weight_samples=self.weight_training_samples
        )

        losses, metrics = [], []
        entropy_stats = {
            "entropy_min": [],
            "entropy_max": [],
            "entropy_mean": [],
            "entropy_std": [],
        }

        for states, actions in train_dataloader:
            loss, metric, ent_stat = self.train_step(states, actions)
            for k, v in ent_stat.items():
                entropy_stats[k].append(v)

            losses.append(loss)
            metrics.append(metric)

        if self.continuous:
            results = {
                "hca_train_loss": np.mean(losses),
                "hca_train_logprobs": np.mean(metrics),
            }
        else:
            results = {
                "hca_train_loss": np.mean(losses),
                "hca_train_acc": np.mean(metrics),
            }
        entropy_stats = {
            "hca_train_" + k: np.mean(v) for k, v in entropy_stats.items()
        }
        results.update(entropy_stats)

        if val_dataloader is not None:
            val_results = self.validate(val_dataloader)
            results.update(val_results)
        return results

    def train_step(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        preds, dists = self.forward(states)

        entropy = dists.entropy()
        entropy_stats = {
            "entropy_mean": entropy.mean().detach().item(),
            "entropy_std": entropy.std().detach().item(),
            "entropy_max": entropy.max().detach().item(),
            "entropy_min": entropy.min().detach().item(),
        }

        if self.continuous:
            loss = F.gaussian_nll_loss(preds, actions, dists.variance)
            metric = dists.log_prob(actions).mean()
        else:
            actions = actions.flatten()
            loss = F.cross_entropy(preds, actions)
            preds = preds.argmax(-1)
            metric = torch.sum(preds == actions) / len(preds)

        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.max_grad_norm
            )

        if get_grad_norm(self.net) > 100.0 and not self.max_grad_norm:
            warnings.warn("Hindsight model grad norm is over 100 but is not being clipped!")

        self.optimizer.step()
        return loss.item(), metric.item(), entropy_stats

    @torch.no_grad()
    def validate(self, val_dataloader):
        losses, metrics = [], []
        for states, actions in val_dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            preds, dists = self.forward(states, add_noise=True)
            if self.continuous:
                loss = F.gaussian_nll_loss(
                    preds, actions, dists.variance
                ).item()
                log_probs = dists.log_prob(actions)
                mean_logprobs = log_probs.mean().item()
                losses.append(loss)
                metrics.append(mean_logprobs)
            else:
                actions = actions.flatten()
                loss = F.cross_entropy(preds, actions).item()
                preds = preds.argmax(-1)
                accuracy = torch.sum(preds == actions) / len(preds)
                losses.append(loss)
                metrics.append(accuracy)
        if self.continuous:
            return {
                "hca_val_loss": np.mean(losses),
                "hca_val_logprobs": np.mean(metrics),
            }
        else:
            return {
                "hca_val_loss": np.mean(losses),
                "hca_val_acc": np.mean(metrics),
            }

    @torch.no_grad()
    def get_hindsight_logprobs(self, inputs, actions):
        """
        get the hindsight values for a batch of actions
        """
        inputs = inputs.to(self.device)
        actions = actions.to(self.device)
        out, dist = self.forward(inputs, add_noise=True)
        if self.continuous:  # B x A
            log_probs = dist.log_prob(actions).reshape(-1, 1)
            return log_probs
        else:
            log_probs = dist.log_prob(actions)
            return log_probs

    @torch.no_grad()
    def get_actions(self, inputs, sample=True):
        """
        samples/ gets argmax actions from the hindsight
        function as if it were a policy.
        returns the actions as a list.
        """
        inputs = inputs.to(self.device)
        out, dist = self.forward(inputs)
        if sample:
            actions = dist.sample()
        else:
            actions = dist.mode()
        if self.continuous:
            actions = actions.reshape(-1, self.action_dim).tolist()
        else:
            actions = actions.flatten().tolist()
        return actions

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)
