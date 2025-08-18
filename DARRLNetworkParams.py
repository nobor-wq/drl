import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-5.0, max_log_std=2.0):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorNet_adv(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNet_adv, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        mean_action = torch.tanh(mu)
        # action = action * self.action_bound
        return mean_action, log_prob, action

class CriticNet_adv(torch.nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=256):
        super(CriticNet_adv, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SAC_lag_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-10, log_std_max=10):
        super(SAC_lag_Net, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class FniNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(FniNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action



