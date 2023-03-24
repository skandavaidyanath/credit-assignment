import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

from utils import flatten, unflatten, sigmoid, normalized_atan


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
        activation_fn="tanh",
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

        self.gamma = args.env.gamma
        self.lamda = args.agent.lamda
        self.entropy_coeff = args.agent.entropy_coeff
        self.value_loss_coeff = args.agent.value_loss_coeff
        self.adv = args.agent.adv
        self.eps_clip = args.agent.eps_clip
        self.ppo_epochs = args.agent.ppo_epochs
        self.max_grad_norm = args.agent.max_grad_norm

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            continuous=continuous,
            n_layers=args.agent.n_layers,
            hidden_size=args.agent.hidden_size,
            activation_fn=args.agent.activation_fn,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

        self.gamma_temp = args.agent.get("gamma_temp", None)
        self.smoothing_fn = args.agent.get("smoothing_fn", None)

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, greedy=greedy)

        return action.detach().cpu(), action_logprob.detach().cpu().item()

    def estimate_hca_discounted_returns(
        self, rewards, logprobs, hindsight_logprobs, temp=0.25, normalize=True
    ):
        """All of the args should be numpy arrays"""
        assert len(rewards) == len(logprobs) == len(hindsight_logprobs)
        returns = []
        for ep_rew, ep_logprobs, ep_hindsight_logprobs in zip(
            rewards, logprobs, hindsight_logprobs
        ):
            ep_hindsight_logprobs = np.array(ep_hindsight_logprobs)
            ratios = np.exp(ep_logprobs - ep_hindsight_logprobs)
            gammas = 1 - ratios
            gammas = sigmoid(gammas, temp=temp)
            T = len(ratios)
            ep_returns = []
            for t in range(T):
                curr_gamma = np.array([gammas[t] ** i for i in range(T - t)])
                discounted_return = (curr_gamma * ep_rew[t:]).sum()
                ep_returns.append(discounted_return)
            returns.append(ep_returns)
        returns = torch.from_numpy(flatten(returns))

        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        return returns.to(self.device)

    def estimate_hca_advantages(
        self, mc_returns, logprobs, hindsight_logprobs, smoothing_fn
    ):
        # Estimate advantages according to Return-conditioned HCA
        # A(s, a) = (1 - \frac{\pi(a | s)}{h(a | s, G_s)})G_s

        smoothing_type = None if smoothing_fn is None else smoothing_fn[0]

        if smoothing_type == "exp":
            hindsight_ratios = torch.exp(
                logprobs.detach() - torch.exp(hindsight_logprobs).detach()
            )
            smoothed_hca = 1 - hindsight_ratios
        else:
            hindsight_ratios = torch.exp(
                logprobs.detach() - hindsight_logprobs.detach()
            )
            hca_terms = 1 - hindsight_ratios
            if smoothing_type is None:
                smoothed_hca = hca_terms
            elif smoothing_type == "tanh":
                a, b, c = (
                    smoothing_fn[1],
                    smoothing_fn[2],
                    smoothing_fn[3],
                )
                smoothed_hca = a * torch.tanh(c * hca_terms) + b
            elif smoothing_type == "atan":
                a, b, c = (
                    smoothing_fn[1],
                    smoothing_fn[2],
                    smoothing_fn[3],
                )

                smoothed_hca = a * normalized_atan(c * hca_terms) + b
            else:
                raise NotImplementedError

        advantages = smoothed_hca * mc_returns
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )

        smoothed_hca_mean = smoothed_hca.mean().item()
        smoothed_hca_max = smoothed_hca.max().item()
        smoothed_hca_min = smoothed_hca.min().item()
        smoothed_hca_std = smoothed_hca.std().item()

        hindsight_stats = {
            "min": smoothed_hca_min,
            "max": smoothed_hca_max,
            "mean": smoothed_hca_mean,
            "std": smoothed_hca_std,
        }

        return advantages.to(self.device), hindsight_stats

    def estimate_montecarlo_returns(self, rewards, terminals, normalize=True):
        # Monte Carlo estimate of returns
        batch_size = len(rewards)
        returns = np.zeros(batch_size)
        returns[batch_size - 1] = rewards[batch_size - 1]
        for t in reversed(range(batch_size - 1)):
            returns[t] = rewards[t] + returns[t + 1] * self.gamma * (
                1 - terminals[t]
            )
        returns = torch.tensor(returns, dtype=torch.float32)
        if normalize:
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
            # normalized by default
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
        hca_ratio_mins, hca_ratio_maxes, hca_ratio_means, hca_ratio_stds = (
            [],
            [],
            [],
            [],
        )

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
                # Don't normalize returns for GAE since we're adding
                # the value to the reward.
                # We don't want the value to be normalized while the reward
                # is not normalized
                # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            elif self.adv == "mc":
                # here both the returns and values are normalized
                # so this shouldb be okay
                advantages = returns - state_values.detach()
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-7
                )
            elif self.adv == "hca":
                # hca adv
                # normalizing the MC returns seems to help stability
                # and performance here
                advantages, hca_info = self.estimate_hca_advantages(
                    returns, logprobs, hindsight_logprobs, self.smoothing_fn
                )
                hca_ratio_mins.append(hca_info["min"])
                hca_ratio_maxes.append(hca_info["max"])
                hca_ratio_means.append(hca_info["mean"])
                hca_ratio_stds.append(hca_info["std"])
            elif self.adv == "mc-hca-gamma":
                # trying by normalizing returns here
                # could change later if required
                assert self.gamma_temp is not None
                unflattened_logprobs = unflatten(
                    logprobs.detach().numpy(), buffer.rewards
                )
                returns = self.estimate_hca_discounted_returns(
                    buffer.rewards,
                    unflattened_logprobs,
                    buffer.hindsight_logprobs,
                    temp=self.gamma_temp,
                    normalize=True,
                )
                advantages = returns
                # TODO: VF or not?
                # TODO: what is the VF trained on? i.e. which gamma

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
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
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
