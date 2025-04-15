import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, continuous=False):
        super(PolicyNetwork, self).__init__()
        self.continuous = continuous

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        if continuous:
            self.mean = nn.Linear(64, output_dim)
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            self.action_head = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.network(x)

        if self.continuous:
            mean = self.mean(x)
            std = self.log_std.exp()
            return mean, std
        else:
            action_probs = torch.softmax(self.action_head(x), dim=-1)
            return action_probs

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state)

        if self.continuous:
            mean, std = self.forward(state)

            if deterministic:
                return mean.detach().numpy()

            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            return action.detach().numpy(), log_prob.detach().item()
        else:
            probs = self.forward(state)

            if deterministic:
                action = torch.argmax(probs).item()
                return action, 0.0

            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.detach().item()


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

