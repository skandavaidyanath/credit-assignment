import torch
import torch.nn as nn
import numpy as np

from utils import flatten, normalized_atan
from arch.actor_critic import ActorCritic


class PPO:
    def __init__(
        self, state_dim, action_dim, lr, continuous, device, args, cnn_base=None
    ):

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
            cnn_base=cnn_base,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

        self.smoothing_fn = args.agent.get("smoothing_fn", None)

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            if len(state.shape) == 3:
                # image input expand batch dim
                state = np.expand_dims(state, 0)
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, greedy=greedy)

        return action.detach().cpu(), action_logprob.detach().cpu().item()

    def estimate_hca_advantages(
        self,
        mc_returns,
        logprobs=None,
        hindsight_logprobs=None,
        smoothing_fn=None,
        hindsight_ratios=None,
    ):
        # Estimate advantages according to Return-conditioned HCA
        # A(s, a) = (1 - \frac{\pi(a | s)}{h(a | s, G_s)})G_s

        if hindsight_logprobs is not None:
            assert (
                hindsight_ratios is None
            ), "Provide only one of hindsight logporbs or ratios"
            smoothing_type = None if smoothing_fn is None else smoothing_fn[0]

            if smoothing_type == "exp":
                # 1 - p/e^x
                hindsight_ratios = torch.exp(
                    logprobs.detach() - torch.exp(hindsight_logprobs).detach()
                )
                smoothed_hca = 1 - hindsight_ratios
            elif smoothing_type == "fancy_exp":
                # p - (1+p)/e^x
                p = torch.exp(logprobs.detach())
                h = torch.exp(hindsight_logprobs.detach())
                smoothed_hca = p - (1 + p) / torch.exp(h)
            else:
                hindsight_ratios = torch.exp(
                    logprobs.detach() - hindsight_logprobs.detach()
                )
                hca_terms = 1 - hindsight_ratios
                if smoothing_type is None:
                    smoothed_hca = hca_terms
                elif smoothing_type == "clip":
                    # clip(hca_terms, min=a, max=b)
                    a, b = smoothing_fn[1], smoothing_fn[2]
                    smoothed_hca = torch.clamp(hca_terms, min=a, max=b)
                elif smoothing_type == "tanh":
                    # a * tanh(c * (1 - p/h)) + b
                    a, b, c = (
                        smoothing_fn[1],
                        smoothing_fn[2],
                        smoothing_fn[3],
                    )
                    smoothed_hca = a * torch.tanh(c * hca_terms) + b
                elif smoothing_type == "atan":
                    # a * norm_atan(c * (1 - p/h)) + b
                    a, b, c = (
                        smoothing_fn[1],
                        smoothing_fn[2],
                        smoothing_fn[3],
                    )

                    smoothed_hca = a * normalized_atan(c * hca_terms) + b
                else:
                    raise NotImplementedError

            advantages = smoothed_hca * mc_returns

            smoothed_hca_mean = smoothed_hca.mean().item()
            smoothed_hca_max = smoothed_hca.max().item()
            smoothed_hca_min = smoothed_hca.min().item()
            smoothed_hca_std = smoothed_hca.std().item()

            hindsight_stats = {
                "ca_stat_type": "smoothed_hca_ratio",
                "min": smoothed_hca_min,
                "max": smoothed_hca_max,
                "mean": smoothed_hca_mean,
                "std": smoothed_hca_std,
            }

        elif hindsight_ratios is not None:
            assert (
                logprobs is None
                and hindsight_logprobs is None
                and smoothing_fn is None
            ), "Provide only one of hindsight logprobs or ratios"
            hca_terms = 1 - hindsight_ratios
            advantages = hca_terms * mc_returns

            hca_mean = hca_terms.mean().item()
            hca_max = hca_terms.max().item()
            hca_min = hca_terms.min().item()
            hca_std = hca_terms.std().item()

            hindsight_stats = {
                "ca_stat_type": "hca_ratio",
                "min": hca_min,
                "max": hca_max,
                "mean": hca_mean,
                "std": hca_std,
            }
        else:
            raise ValueError(
                "Provide at least one of hindsight logprobs or ratios"
            )

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )
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

        advantages = torch.tensor(advantages, dtype=torch.float32).to(
            self.device
        )
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )
        return advantages.to(self.device), returns.to(self.device)

    def update(self, buffer):
        batch_states = flatten(buffer.states)
        batch_actions = flatten(buffer.actions)
        batch_logprobs = flatten(buffer.logprobs)
        batch_rewards = flatten(buffer.rewards)
        batch_terminals = flatten(buffer.terminals)
        hindsight_logprobs = flatten(buffer.hindsight_logprobs)
        hindsight_ratios = flatten(buffer.hindsight_ratios)

        if self.adv != "gae":
            # normalized by default
            returns = self.estimate_montecarlo_returns(
                batch_rewards, batch_terminals
            )

        # convert list to tensor
        # removed the torch.squeezes from here.
        # shouldn't be required with the new flatten function.
        old_states = (
            torch.from_numpy(batch_states)
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.from_numpy(batch_actions)
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.from_numpy(batch_logprobs)
            .detach()
            .to(self.device)
        )
        hindsight_logprobs = (
            torch.from_numpy(hindsight_logprobs)
            .detach()
            .to(self.device)
        )
        hindsight_ratios = (
            torch.from_numpy(hindsight_ratios)
            .detach()
            .to(self.device)
        )
        
        total_losses, action_losses, value_losses, entropies = [], [], [], []

        ca_stats_mins, ca_stats_maxes, ca_stats_means, ca_stats_stds = (
            [],
            [],
            [],
            [],
        )
        ca_stat_type = None

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
                advantages, returns = self.estimate_gae(
                    batch_rewards, state_values.detach(), batch_terminals
                )
                # returns = advantages + state_values.detach()
                # Don't normalize returns for GAE since we're adding
                # the value to the reward.
                # We don't want the value to be normalized while the reward
                # is not normalized
                # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            elif self.adv == "mc":
                # here both the returns and values are normalized
                # so this should be okay
                advantages = returns - state_values.detach()
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-7
                )
            elif self.adv == "hca":
                # hca adv
                # normalizing the MC returns seems to help stability
                # and performance here
                if (
                    hindsight_logprobs is not None
                    and len(hindsight_logprobs) > 0
                ):
                    advantages, ca_stats = self.estimate_hca_advantages(
                        returns,
                        logprobs=logprobs,
                        hindsight_logprobs=hindsight_logprobs,
                        smoothing_fn=self.smoothing_fn,
                        hindsight_ratios=None,
                    )
                elif hindsight_ratios is not None:
                    advantages, ca_stats = self.estimate_hca_advantages(
                        returns,
                        logprobs=None,
                        hindsight_logprobs=None,
                        smoothing_fn=None,
                        hindsight_ratios=hindsight_ratios,
                    )
                else:
                    raise ValueError(
                        "Unexpected error: One of logprobs or ratios should not be None in buffer!"
                    )

                ca_stat_type = ca_stats["ca_stat_type"]
                ca_stats_mins.append(ca_stats["min"])
                ca_stats_maxes.append(ca_stats["max"])
                ca_stats_means.append(ca_stats["mean"])
                ca_stats_stds.append(ca_stats["std"])

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

        if "hca" not in self.adv:
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                np.mean(value_losses),
                np.mean(entropies),
                {},
            )
        else:
            ca_stats_dict = {
                "ca_stat_type": ca_stat_type,
                "max": np.max(ca_stats_maxes),
                "min": np.min(ca_stats_mins),
                "mean": np.mean(ca_stats_means),
                "std": np.mean(ca_stats_stds),
            }
            return (
                np.mean(total_losses),
                np.mean(action_losses),
                0,
                np.mean(entropies),
                ca_stats_dict,
            )

    def save(self, checkpoint_path, args):
        torch.save(
            {"policy": self.policy.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.policy.load_state_dict(checkpoint)
