import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
import numpy as np


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
    ):
        super(HCAModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
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

        # if not continuous:
        #     layers.append(nn.Softmax(dim=-1))

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

    def forward(self, inputs):
        """
        forward pass a bunch of inputs into the model
        """
        out = self.net(inputs)
        if self.continuous:
            std = torch.diag(self.std)
            dist = MultivariateNormal(out, scale_tril=std)
        else:
            dist = Categorical(logits=out)
        return out, dist

    def _get_hca_batch(self, buffer):
        states = np.concatenate(buffer.states, 0)  # B, D
        returns = np.concatenate(buffer.returns, 0).reshape(-1, 1)  # B, 1
        X = np.concatenate((states, returns), -1)
        return torch.from_numpy(X).to(self.device)

    def update(self, buffer):
        train_dataloader, val_dataloader = buffer.get_dataloader(
            self.batch_size
        )

        losses, metrics = [], []
        for states, actions in train_dataloader:
            loss, metric = self.train_step(states, actions)
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

        val_results = self.validate(val_dataloader)
        results.update(val_results)
        return results

    def train_step(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        preds, dists = self.forward(states)

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
        self.optimizer.step()
        return loss.item(), metric.item()

    def validate(self, val_dataloader):
        losses, metrics = [], []
        for states, actions in val_dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            preds, dists = self.forward(states)
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

    def get_hindsight_values(self, inputs, actions):
        """
        get the hindsight values for a batch of actions
        """
        inputs = inputs.to(self.device)
        actions = actions.to(self.device)
        out, dist = self.forward(inputs)
        if self.continuous:  # B x A
            log_probs = dist.log_prob(actions).reshape(-1, 1)
            return log_probs.exp()
        else:
            log_probs = dist.log_prob(actions)
            # actions = actions.reshape(-1, 1).long()
            # return out.gather(1, actions)  # B,
            return log_probs.exp()

    def save(self, checkpoint_path, args):
        torch.save(
            {"model": self.net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.net.load_state_dict(checkpoint)
