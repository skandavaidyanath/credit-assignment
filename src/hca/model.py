import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
import numpy as np
import warnings

from utils import weight_reset, get_grad_norm, model_init
from arch.cnn import CNNBase


class HCAModel(nn.Module):
    """
    HCA model to predict action probabilities conditioned on returns and state
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        cnn_base=None,
        n_layers=2,
        hidden_size=64,
        activation_fn="relu",
        dropout_p=0,
        batch_size=64,
        lr=3e-4,
        device="cpu",
        normalize_inputs=False,
        normalize_return_inputs_only=False,
        max_grad_norm=None,
        weight_training_samples=False,
        noise_std=None,
    ):
        super(HCAModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim if continuous else 1
        self.continuous = continuous
        self.normalize_inputs = normalize_inputs or normalize_return_inputs_only
        self.normalize_return_inputs_only = normalize_return_inputs_only
        self.max_grad_norm = max_grad_norm
        self.noise_std = noise_std

        layers = []
        if cnn_base is not None:
            # if a CNN base is passed in, then use that instead of an MLP.
            assert isinstance(cnn_base, CNNBase)
            assert cnn_base.hidden_size == hidden_size
            self.cnn = cnn_base
            final_layer_init_ = lambda m: model_init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
            )
            assert state_dim == hidden_size + 1  # +1 for the return
        else:
            self.cnn = nn.Sequential(*[])

            if activation_fn == "tanh":
                activation = nn.Tanh
            elif activation_fn == "relu":
                activation = nn.ReLU
            else:
                raise NotImplementedError

            if n_layers == 0:
                hidden_size = state_dim
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

            final_layer_init_ = lambda m: m

        layers.append(final_layer_init_(nn.Linear(hidden_size, action_dim)))

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

        self.state_mean = torch.zeros((state_dim,), device=device)
        self.state_std = torch.ones((state_dim,), device=device)
        self.return_mean = torch.zeros((1,), device=device)
        self.return_std = torch.ones((1,), device=device)

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

    def update_norm_stats(
        self, state_mean, state_std, return_mean, return_std, refresh=True
    ):
        if refresh:  # re-calculate stats each time we train model
            self.state_mean = (
                torch.from_numpy(state_mean).to(self.device).float()
            )
            self.state_std = torch.from_numpy(state_std).to(self.device).float()
            self.return_mean = (
                torch.from_numpy(return_mean).to(self.device).float()
            )
            self.return_std = (
                torch.from_numpy(return_std).to(self.device).float()
            )
        else:
            raise NotImplementedError

    def forward(self, states, returns, add_noise=False):
        """
        forward pass a bunch of inputs into the model
        """
        if self.normalize_inputs:
            # if self.normalize_return_inputs_only==True, then the non-return input mean and std will be 0 and 1 resp.
            states = (states - self.state_mean) / (self.state_std + 1e-6)
            returns = (returns - self.return_mean) / (self.return_std + 1e-6)

        embeds = self.cnn(states)
        inputs = torch.concat([embeds, returns], dim=-1)
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

        for states, returns, actions in train_dataloader:
            loss, metric, ent_stat = self.train_step(states, returns, actions)
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

    def train_step(self, states, returns, actions):
        states = states.to(self.device)
        returns = returns.to(self.device)
        actions = actions.to(self.device)
        preds, dists = self.forward(states, returns)

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

        # if get_grad_norm(self.net) > 100.0 and not self.max_grad_norm:
        #     warnings.warn("Hindsight model grad norm is over 100 but is not being clipped!")

        self.optimizer.step()
        return loss.item(), metric.item(), entropy_stats

    @torch.no_grad()
    def validate(self, val_dataloader):
        losses, metrics = [], []
        for states, returns, actions in val_dataloader:
            states = states.to(self.device)
            returns = returns.to(self.device)
            actions = actions.to(self.device)
            preds, dists = self.forward(states, returns)
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
    def get_hindsight_logprobs(self, states, returns, actions):
        """
        get the hindsight values for a batch of actions
        """
        states = states.to(self.device)
        returns = returns.to(self.device)
        actions = actions.to(self.device)
        out, dist = self.forward(states, returns)
        if self.continuous:  # B x A
            log_probs = dist.log_prob(actions).reshape(-1, 1)
            return log_probs
        else:
            log_probs = dist.log_prob(actions)
            return log_probs

    @torch.no_grad()
    def get_actions(self, states, returns, sample=True):
        """
        samples/ gets argmax actions from the hindsight
        function as if it were a policy.
        returns the actions as a list.
        """
        states = states.to(self.device)
        returns = returns.to(self.device)
        out, dist = self.forward(states, returns)
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
