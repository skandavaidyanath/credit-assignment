import torch
import torch.nn as nn
from dqn.utils import np_to_torch, LinearSchedule

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers=2, hidden_size=64, exploration="eps-greedy"):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim

        layers = []
        inp_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(inp_dim, hidden_size))
            layers.append(nn.Tanh())
            inp_dim = hidden_size
        layers.append(nn.Linear(inp_dim, action_dim))
        self.q_net = nn.Sequential(*layers)

        self.exploration = exploration

    def forward(self, state):
        q_vals = self.q_net(state)
        return q_vals.reshape(-1, self.action_dim)

    def act(self, state, greedy=False, random=False, eps=None):
        with torch.no_grad():
            q_vals = self.forward(state)
            greedy_acts = torch.argmax(q_vals, dim=-1)
            batch_size = q_vals.shape[0]

        if greedy:
            return greedy_acts
        if random:
            return torch.distributions.Categorical(logits=torch.ones_like(q_vals)).sample()
        if self.exploration == "eps-greedy":
            assert eps is not None
            random_acts = torch.distributions.Categorical(logits=torch.ones_like(q_vals)).sample()
            take_random = (torch.rand(batch_size) < eps).int()
            acts = take_random * random_acts + (1 - take_random) * greedy_acts
            return acts
        else:
            raise NotImplementedError

    def evaluate(self, state, action):
        q_vals = self.forward(state)
        q_vals = torch.gather(q_vals, dim=1, index=action.long())
        return q_vals


class DQN:
    def __init__(self, state_dim, action_dim, lr, device, eps_decay_steps,  eps_init=1.0,
                 eps_fin=0.1, gamma=0.99, n_layers=2, hidden_size=64, max_grad_norm=None, loss_fn="huber"):
        self.device = device

        self.gamma = gamma
        self.q_net = QNetwork(
            state_dim,
            action_dim,
            n_layers=n_layers,
            hidden_size=hidden_size,
        ).to(device)

        self.target_q_network = QNetwork(
            state_dim,
            action_dim,
            n_layers=n_layers,
            hidden_size=hidden_size,
        ).to(device)

        self.eps_schedule = LinearSchedule(eps_decay_steps, initial_p=eps_init, final_p=eps_fin)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        if loss_fn == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        #
        # # number of env steps between gradient updates to policy
        # self.policy_update_freq = args.policy_update_freq
        #
        # # number of env steps between updates to the target network
        # self.target_net_update_freq = args.target_net_update_freq
        #
        # # polyak update rate for target net. If None, perform "hard" policy updates
        # self.polyak_rate = polyak_rate
        #
        # self.max_grad_norm = args.max_grad_norm

    def select_action(self, state, greedy=False, random=False, t=None):
        state = torch.FloatTensor(state).to(self.device)
        eps = None
        if t:
            eps = self.eps_schedule.value(t=t)
        return self.q_net.act(state, greedy=greedy, random=random, eps=eps)

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)
        # form bootstrapped q-targets
        with torch.no_grad():
            next_qs = self.q_net(next_states)
            # take greedy action at next step
            next_qs, _ = torch.max(next_qs, dim=-1)
            next_qs = next_qs.reshape(-1, 1)
            # TD target
            target_qs = rewards + (1 - dones) * self.gamma * next_qs

        # get the q-values for current observations evaluated at the taken actions
        current_qs = self.q_net.evaluate(states, actions)
        loss = self.loss_fn(current_qs, target_qs)


        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradient norm
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def unpack_batch(self, batch):
        states, actions, rewards, next_states, dones = batch
        return np_to_torch(states, self.device),\
               np_to_torch(actions, self.device),\
               np_to_torch(rewards, self.device), \
               np_to_torch(next_states, self.device), \
               np_to_torch(dones, self.device)

    def save(self, checkpoint_path, args):
        torch.save(
            {"q_net": self.q_net.state_dict(), "args": args}, checkpoint_path
        )

    def load(self, checkpoint):
        self.q_net.load_state_dict(checkpoint)

if __name__ == '__main__':
    q_net = QNetwork(5, 10)
    states = torch.randn((4, 5))
    actions = torch.randint(low=0, high=10, size=(4, 1))
    rewards = torch.randn((4, 1))
    next_states = torch.randn((4, 5))
    dones = torch.zeros_like(rewards)

    dqn = DQN(state_dim=5, action_dim=10, lr=0.01, device='cpu')
    dqn.train_step((states, actions, rewards, next_states, dones))
    a = q_net.act(states)
    b = 2
