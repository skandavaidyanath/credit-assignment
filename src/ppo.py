import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

from utils import flatten


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
    ):
        super(ActorCritic, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

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
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.Tanh())
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


class PPO:
    def __init__(self, state_dim, action_dim, lr, continuous, device, args):

        self.continuous = continuous
        self.device = device

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.entropy_coeff = args.entropy_coeff
        self.value_loss_coeff = args.value_loss_coeff
        self.adv = args.adv
        self.eps_clip = args.eps_clip
        self.ppo_epochs = args.ppo_epochs

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            continuous=continuous,
            n_layers=args.n_layers,
            hidden_size=args.hidden_size,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, greedy=greedy)

        return action.detach().cpu(), action_logprob.detach().cpu().item()

    def estimate_hca_advantages(self, mc_returns, logprobs, hindsight_logprobs):
        # Estimate advantages according to Return-conditioned HCA
        # A(s, a) = (1 - \frac{\pi(a | s)}{h(a | s, G_s)})G_s
        hindsight_ratios = torch.exp(
            logprobs.detach() - hindsight_logprobs.detach()
        )
        advantages = (1 - hindsight_ratios) * mc_returns
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )

        return advantages.to(self.device)

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
        batch_states = flatten(buffer.states)
        batch_actions = flatten(buffer.actions)
        batch_logprobs = flatten(buffer.logprobs)
        batch_rewards = flatten(buffer.rewards)
        batch_terminals = flatten(buffer.terminals)
        hindsight_logprobs = flatten(buffer.hindsight_logprobs)

        if self.adv != "gae":
            returns = self.estimate_montecarlo_returns(
                batch_rewards, batch_terminals
            )

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.from_numpy(batch_states))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.from_numpy(batch_actions))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.from_numpy(batch_logprobs))
            .detach()
            .to(self.device)
        )
        hindsight_logprobs = (
            torch.squeeze(torch.from_numpy(hindsight_logprobs))
            .detach()
            .to(self.device)
        )

        total_losses, action_losses, value_losses, entropies = [], [], [], []

        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            if self.adv == "gae":
                advantages = self.estimate_gae(
                    batch_rewards, state_values.detach(), batch_terminals
                )
                returns = advantages + state_values.detach()
                returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            elif self.adv == "mc":
                advantages = returns - state_values.detach()
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-7
                )
            else:
                # hca adv
                advantages = self.estimate_hca_advantages(
                    returns, logprobs, hindsight_logprobs
                )

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages
            )

            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values, returns)

            # final loss of clipped objective PPO
            loss = (
                action_loss
                + self.value_loss_coeff * value_loss
                - self.entropy_coeff * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_losses.append(loss.detach().cpu().item())
            action_losses.append(action_loss.detach().cpu().item())
            value_losses.append(value_loss.detach().cpu().item())
            entropies.append(dist_entropy.detach().cpu().item())

        if self.adv != "hca":
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                np.mean(value_losses),
                np.mean(entropies),
            )
        else:
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                0,
                np.mean(entropies),
            )

    def save(self, checkpoint_path, args):
        torch.save(
            {"policy": self.policy.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.policy.load_state_dict(checkpoint)
