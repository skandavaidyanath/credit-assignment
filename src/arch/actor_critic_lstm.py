import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_packed_sequence

from utils import flatten, lstm_preprocess_buffer_states


class ActorCriticLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        continuous=False,
        n_layers=2,
        hidden_size=64,
        activation_fn="tanh",
    ):
        """
        Default Actor Critic with a LSTM Critic function instead of FFN
        """
        super(ActorCriticLSTM, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        if activation_fn == "tanh":
            activation_fn = nn.Tanh()
        elif activation_fn == "relu":
            activation_fn = nn.ReLU()
        else:
            raise NotImplementedError()

        assert (
            n_layers > 0
        ), "Must have at least one hidden layer for LSTM critic model!"

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(activation_fn)
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation_fn)

        if continuous:
            layers.append(nn.Linear(hidden_size, action_dim))
        else:
            layers.append(nn.Linear(hidden_size, action_dim))
            layers.append(
                nn.Softmax(dim=-1),
            )
        self.actor = nn.Sequential(*layers)

        self.critic_lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.critic_out = nn.Linear(hidden_size, 1)

    @property
    def std(self):
        if not self.continuous:
            raise ValueError("Calling std() on Discrete policy!")
        else:
            return torch.exp(self.log_std)

    def forward(self, state):
        if self.continuous:
            action_mean = self.actor(state)
            std = torch.diag(self.std)
            dist = MultivariateNormal(action_mean, scale_tril=std)
        else:
            action_probs = self.actor(state)
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

    def evaluate(self, states, actions):
        # `states` is a list of list of numpy arrays straight from
        # `buffer.states`
        actor_states = flatten(states)
        actor_states = (
            torch.from_numpy(actor_states).detach().to(self.critic_out.weight.device)
        )
        critic_states, lens = lstm_preprocess_buffer_states(states)

        # Actor stuff
        dists = self.forward(actor_states)
        # For Single Action Environments.
        if self.continuous and self.action_dim == 1:
            actions = actions.reshape(-1, self.action_dim)
        action_logprobs = dists.log_prob(actions)
        dist_entropy = dists.entropy().mean()

        # Critic stuff
        packed_output, _ = self.critic_lstm(critic_states)
        # Unpack the sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply the fully connected layer to each timestep
        batch_size, _, _ = output.size()

        # Prepare the output tensor
        state_values = []

        # For each sequence in the batch, take the relevant timesteps
        for i in range(batch_size):
            # Extract timesteps that are not padded
            relevant_timesteps = output[i, : lens[i]]
            # Apply the fully connected layer
            fc_output = self.critic_out(relevant_timesteps)
            state_values.append(fc_output)

        # Concatenate all the outputs from the batch
        state_values = torch.cat(state_values, dim=0)

        assert len(action_logprobs) == len(state_values)

        return action_logprobs, state_values, dist_entropy
