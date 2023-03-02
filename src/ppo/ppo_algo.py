import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
        activation_fn="tanh"
    ):
        super(ActorCritic, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        if activation_fn == "tanh":
            activation_fn = nn.Tanh()
        elif activation_fn == "relu":
            activation_fn = nn.ReLU()
        else:
            raise NotImplementedError()

        if continuous:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim), requires_grad=True
            )

        if n_layers == 0:
            self.encoder = nn.Sequential(*[])
            hidden_size = state_dim
        else:
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layers.append(nn.Linear(state_dim, hidden_size))
                    layers.append(activation_fn)
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(activation_fn)
            self.encoder = nn.Sequential(*layers)

        # actor
        if continuous:
            self.actor = nn.Linear(hidden_size, action_dim)
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Linear(hidden_size, 1)

    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete policy!")
        else:
            return torch.exp(self.log_std)

    def forward(self, state, return_value=False):
        encoding = self.encoder(state)
        if self.continuous:
            action_mean = self.actor(encoding)
            std = torch.diag(self.std)
            dist = MultivariateNormal(action_mean, scale_tril=std)
        else:
            action_probs = self.actor(encoding)
            dist = Categorical(action_probs)
        state_value = None
        if return_value:
            state_value = self.critic(encoding)
        return dist, state_value

    def act(self, state, greedy=False, return_value=False):
        dist, value = self.forward(state, return_value=return_value)
        if value:
            value = value.detach()

        if greedy:
            action = dist.mode
        else:
            action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), value

    def evaluate(self, state, action):
        dist, state_values = self.forward(state, return_value=True)
        # For Single Action Environments.
        if self.continuous and self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, continuous, device, args):

        self.continuous = continuous
        self.device = device

        self.gamma = args.env.gamma
        self.lamda = args.agent.lamda
        self.entropy_coeff = args.agent.entropy_coeff
        self.value_loss_coeff = args.agent.value_loss_coeff
        self.adv = args.agent.adv
        self.eps_clip = args.agent.eps_clip
        self.clip_range_vf = args.agent.clip_range_vf
        self.ppo_epochs = args.agent.ppo_epochs
        self.minibatch_size = args.agent.minibatch_size
        self.max_grad_norm = args.agent.max_grad_norm

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            continuous=continuous,
            n_layers=args.agent.n_layers,
            hidden_size=args.agent.hidden_size,
            activation_fn=args.agent.activation_fn
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, value = self.policy.act(state, greedy=greedy, return_value=not greedy)
        if value:
            value = value.cpu().item()

        return action.cpu(), action_logprob.cpu().item(), value

    def estimate_hca_advantages(self, mc_returns, logprobs, hindsight_logprobs):
        # Estimate advantages according to Return-conditioned HCA
        # A(s, a) = (1 - \frac{\pi(a | s)}{h(a | s, G_s)})G_s
        hindsight_ratios = torch.exp(
            logprobs.detach() - hindsight_logprobs.detach()
        )
        hindsight_ratio_mean = hindsight_ratios.mean().item()
        hindsight_ratio_max = hindsight_ratios.max().item()
        hindsight_ratio_min = hindsight_ratios.min().item()
        hindsight_ratio_std = hindsight_ratios.std().item()

        hindsight_stats = {
            "min": hindsight_ratio_min,
            "max": hindsight_ratio_max,
            "mean": hindsight_ratio_mean,
            "std": hindsight_ratio_std,
        }

        advantages = (1 - hindsight_ratios) * mc_returns
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )

        return advantages.to(self.device), hindsight_stats

    def estimate_montecarlo_returns(self, rewards, terminals):
        # Monte Carlo estimate of returns
        batch_size = len(rewards)
        returns = np.zeros(batch_size)
        returns[batch_size - 1] = rewards[batch_size - 1]
        for t in reversed(range(batch_size - 1)):
            returns[t] = rewards[t] + returns[t + 1] * self.gamma * (
                1 - terminals[t]
            )

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        return returns.to(self.device)

    def estimate_gae(self, rewards, values, terminals):
        # GAE estimates of Advantage
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)
        advantages[batch_size - 1] = (
            rewards[batch_size - 1] - values[batch_size - 1]
        )
        for t in reversed(range(batch_size - 1)):
            delta = (
                rewards[t]
                + (self.gamma * values[t + 1] * (1 - terminals[t]))
                - values[t]
            )
            advantages[t] = delta + (
                self.gamma * self.lamda * advantages[t + 1] * (1 - terminals[t])
            )

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )
        return advantages.to(self.device)

    def update(self, buffer):
        total_losses, action_losses, value_losses, entropies = [], [], [], []
        hca_ratio_mins, hca_ratio_maxes, hca_ratio_means, hca_ratio_stds = (
            [],
            [],
            [],
            [],
        )

        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            mb_total_losses, mb_action_losses, mb_value_losses, mb_entropies = [], [], [], []

            # iterate over big batch in smaller minibatches
            for minibatch in buffer.generate_batches(self.gamma, self.lamda, self.minibatch_size, self.adv, self.device):
                # Evaluating old actions and values
                states, actions, old_logprobs, old_values, advantages, returns = minibatch
                logprobs, new_values, entropy = self.policy.evaluate(states, actions)

                # Compute Surrogate Loss
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = (
                        torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                        * advantages
                )

                action_loss = -torch.min(surr1, surr2).mean()
                # TODO: implement clipped value loss
                if self.clip_range_vf:
                    new_values = new_values.flatten()
                    values_pred = old_values + torch.clamp(
                        new_values - old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                else:
                    values_pred = new_values.flatten()
                value_loss = self.MseLoss(values_pred, returns)
                loss = (
                        action_loss
                        + self.value_loss_coeff * value_loss
                        - self.entropy_coeff * entropy
                )
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # clip gradient if applicable
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mb_total_losses.append(loss.detach().cpu())
                mb_action_losses.append(action_loss.detach().cpu())
                mb_value_losses.append(value_loss.detach().cpu())
                mb_entropies.append(entropy.detach().cpu())

            # TODO: hca advantages into new buffer setup
            # else:
            #     # hca adv
            #     advantages, hca_info = self.estimate_hca_advantages(
            #         returns, logprobs, hindsight_logprobs
            #     )
            #     hca_ratio_mins.append(hca_info["min"])
            #     hca_ratio_maxes.append(hca_info["max"])
            #     hca_ratio_means.append(hca_info["mean"])
            #     hca_ratio_stds.append(hca_info["std"])
            #

            total_losses.append(np.mean(mb_total_losses))
            action_losses.append(np.mean(mb_action_losses))
            value_losses.append(np.mean(mb_value_losses))
            entropies.append(np.mean(mb_entropies))

        if self.adv != "hca":
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                np.mean(value_losses),
                np.mean(entropies),
                {},
            )
        else:
            hca_stats_dict = {
                "max": np.max(hca_ratio_maxes),
                "min": np.min(hca_ratio_mins),
                "mean": np.mean(hca_ratio_means),
                "std": np.mean(hca_ratio_stds),
            }
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                0,
                np.mean(entropies),
                hca_stats_dict,
            )

    def save(self, checkpoint_path, args):
        torch.save(
            {"policy": self.policy.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.policy.load_state_dict(checkpoint)
