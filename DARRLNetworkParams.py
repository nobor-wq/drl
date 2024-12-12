import torch
import random
import torch.nn as nn
from config import BaseConfig, Configurable
import torch.nn.functional as F
from torch_util import  Module, mlp

def pythonic_mean(x):
    return sum(x) / len(x)

class QNetwork(Configurable, Module):
    class Config(BaseConfig):
        n_critics = 2
        hidden_layers = 2
        hidden_dim = 256
        learning_rate = 1e-3

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
        self.learning_rate = config.learning_rate
        self.qs = torch.nn.ModuleList([
            mlp(dims, squeeze_output=True) for _ in range(self.n_critics)
        ])
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.learning_rate)

    def all(self, state, action):
        sa = torch.cat([state, action], 1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def random_choice(self, state, action):
        sa = torch.cat([state, action], 1)
        return random.choice(self.qs)(sa)


class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(PolicyNetwork, self).__init__()
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


class ActionCostNetwork(Configurable, Module):
    class Config(BaseConfig):
        n_ActionCost = 2
        hidden_layers = 2
        hidden_dim = 256
        learning_rate = 1e-3

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
        self.learning_rate = config.learning_rate
        self.qs = torch.nn.ModuleList([
            mlp(dims, squeeze_output=True) for _ in range(self.n_ActionCost)
        ])
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.learning_rate)

    def all(self, state, action):
        sa = torch.cat([state, action], 1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def compute_cost_for_single_state(self, state, action):
        """
        计算单个状态和动作的成本
        输入：
        - state: 单一状态（Tensor），假设其为 [state_dim] 或 [1, state_dim]
        - action: 对应的动作（Tensor），假设其为 [action_dim] 或 [1, action_dim]

        返回：
        - cost: 计算出的动作成本
        """
        # 确保 state 和 action 是 2D 张量 (batch_size, dim)
        if state.dim() == 1:  # 单一状态的情况，调整为 [1, state_dim]
            state = state.unsqueeze(0)
        if action.dim() == 1:  # 单一动作的情况，调整为 [1, action_dim]
            action = action.unsqueeze(0)

        # 确保拼接时维度匹配
        # sa = torch.cat([state, action], 1)  # 拼接状态和动作
        return self.min(state, action)  # 你可以选择用 min, mean 或其他方式

    def random_choice(self, state, action):
        sa = torch.cat([state, action], 1)
        return random.choice(self.qs)(sa)

#
# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_dim=256):
#         super(QNetwork, self).__init__()
#
#         # 定义第一层，输入为状态和动作拼接后的维度
#         self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
#         self.relu = nn.ReLU()
#
#         # 定义多个隐藏层
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)])
#
#         # 定义输出层，输出一个标量
#         self.fc_out = nn.Linear(hidden_dim, 1)
#
#     def forward(self, state, action):
#         sa = torch.cat([state, action], dim=1)  # 将状态和动作拼接
#         x = self.relu(self.fc1(sa))  # 第一层并应用 ReLU
#
#         # 通过隐藏层
#         for hidden_layer in self.hidden_layers:
#             x = self.relu(hidden_layer(x))
#
#         # 输出 Q 值
#         return self.fc_out(x)

#
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_dim=256, min_log_std=-10, max_log_std=2):
#
#         super(PolicyNetwork, self).__init__()
#
#         # 定义第一层
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.relu = nn.ReLU()
#
#         # 定义多个隐藏层
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)])
#
#         # 定义输出层，输出动作均值和 log 标准差
#         self.mu_head = nn.Linear(hidden_dim, action_dim)
#         self.log_std_head = nn.Linear(hidden_dim, action_dim)
#
#         self.min_log_std = min_log_std
#         self.max_log_std = max_log_std
#
#     def forward(self, state):
#         x = self.relu(self.fc1(state))  # 第一层并应用 ReLU
#         # 通过隐藏层
#         for hidden_layer in self.hidden_layers:
#             x = self.relu(hidden_layer(x))
#         # 计算动作均值和标准差
#         mu = torch.tan(self.mu_head(x))
#         # 20241125 wq 后面的可选的平方根操作，是否留下待定
#         std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()
#         action = mu + std * torch.randn_like(mu)
#         action = torch.clamp(action, -1, 1)
#         return mu, std, action

#
# class ActionCostNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_dim=256):
#         super(ActionCostNetwork, self).__init__()
#
#         # 定义第一层，输入为状态和动作拼接后的维度
#         self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
#         self.relu = nn.ReLU()
#
#         # 定义多个隐藏层
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)])
#
#         # 定义输出层，输出一个标量
#         self.fc_out = nn.Linear(hidden_dim, 1)
#
#     def forward(self, state, action):
#         sa = torch.cat([state, action], dim=1)  # 将状态和动作拼接
#         x = self.relu(self.fc1(sa))  # 第一层并应用 ReLU
#
#         # 通过隐藏层
#         for hidden_layer in self.hidden_layers:
#             x = self.relu(hidden_layer(x))
#
#         # 输出成本值
#         return self.fc_out(x)

#
# class PolicyDeviationNetwork(nn.Module):
#     def __init__(self):
#         """
#         初始化策略偏差网络
#         """
#         super(PolicyDeviationNetwork, self).__init__()
#
#     def forward(self, policy_action, perturbed_action):
#         """
#         计算策略偏差
#         :param policy_action: 原始策略动作
#         :param perturbed_action: 扰动后的策略动作
#         :return: 偏差值
#         """
#         return torch.mean((policy_action - perturbed_action) ** 2)
