import copy
import numpy as np
import torch
from pyglet.gl import Config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from torch import nn
from config import BaseConfig, Configurable
from torch_util import device, Module, mlp, update_ema, freeze_module
from DARRLNetworkParams import QNetwork, PolicyNetwork, ActionCostNetwork
import torch.nn.functional as F

def pythonic_mean(x):
    return sum(x) / len(x)

class DARRL(Module):
    class Config(BaseConfig):
        discount = 0.99
        deterministic_backup = False
        critic_update_multiplier = 1
        critic_cfg = QNetwork.Config()
        acost_cfg = ActionCostNetwork.Config()
        tau = 0.005
        batch_size = 64
        hidden_dim = 256
        hidden_layers = 2
        update_violation_cost = True
        actor_lr = 1e-4
        acost_lr = 1e-4
        lam_lr = 1e-4
        gamma = 0.99

    def __init__(self, config, state_dim, action_dim, optimizer_factory=torch.optim.Adam):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = QNetwork(self.critic_cfg, state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)
        self.action_cost = ActionCostNetwork(self.acost_cfg, state_dim, action_dim)
        self.action_cost_target = copy.deepcopy(self.action_cost)
        freeze_module(self.action_cost_target)
        self.gamma = 0.99
        self.eps1 = torch.tensor(2.0, dtype=torch.float32, device=device)
        self.eps2 = torch.tensor(2.0, dtype=torch.float32, device=device)
        self.actor_lr = config.actor_lr
        self.lam_lr = config.lam_lr
        self.lam1 = torch.zeros(1, requires_grad=True, device=device)
        self.log_lam1 = torch.zeros(1, requires_grad=True, device=device)
        self.lam2 = torch.zeros(1, requires_grad=True, device=device)
        self.log_lam2 = torch.zeros(1, requires_grad=True, device=device)
        self.criterion = nn.MSELoss()
        self.epsilon_a = 0.1
        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)

        self.lam_optimizer = optimizer_factory([self.log_lam1, self.log_lam2], lr=self.lam_lr)
        # kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        # kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-10, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        kernel = C(1.0, (1e-10, 1e5)) * RBF(1.0, (1e-8, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def critic_loss(self, states_cl, action, next_states, reward, cost, done):
        target = self.compute_q_target(next_states, reward, done)
        return self.critic_loss_given_target(states_cl, action, target)


    def compute_q_target(self, next_states, reward, done):
        with torch.no_grad():
            _, _, next_action = self.actor(next_states)
            next_value = self.critic_target.min(next_states, next_action)
            q = reward + self.gamma * (1. - done.float()) * next_value
            # print("目标q值：", q)
            return q
    def critic_loss_given_target(self, states_clgt, action, target):
        qs = self.critic.all(states_clgt, action)
        losses = [self.criterion(q, target) for q in qs]
        # for i, loss in enumerate(losses):
        #     print(f"损失 {i}: 类型 {type(loss)}, 值: {loss}")
        return pythonic_mean(losses)
        # return pythonic_mean([self.criterion(q, target) for q in qs])

    def update_critic(self, *critic_loss_args):
        critic_loss = self.critic_loss(*critic_loss_args)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    def action_cost_loss(self, states, action, next_states, rewards, cost, done):
        target = self.compute_c_target(next_states, cost, done)
        return self.action_loss_given_target(states, action, target)

    def compute_c_target(self, next_states, cost, done):
        with torch.no_grad():
            _, _, next_action = self.actor(next_states)
            next_cost = self.action_cost_target.mean(next_states, next_action)
            c = cost + self.gamma * (1. - done.float()) * next_cost
            # print("目标cost: ",c)
            return c
    def action_loss_given_target(self, states_algt, action, target):
        qs = self.action_cost.all(states_algt, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

    def update_action_cost(self, *action_loss_args):
        action_cost_loss = self.action_cost_loss(*action_loss_args)
        self.action_cost.optimizer.zero_grad()
        action_cost_loss.backward()
        self.action_cost.optimizer.step()
        update_ema(self.action_cost_target, self.action_cost, self.tau)
        return action_cost_loss.detach()

    def actor_loss(self, states_al, perturbed_states):
        _, _, action = self.actor(states_al)
        _, _, perturbed_action = self.actor(perturbed_states)
        actor_Q = self.critic.min(states_al, action)
        # action_C = self.log_lam1 * (self.eps1 - self.action_cost.mean(perturbed_states,perturbed_action))
        # cost_P = self.log_lam2 * (self.eps2 - torch.mean((action - perturbed_action) ** 2))
        # 2024-12-11 wq
        action_C = self.log_lam1 * (torch.tensor(self.eps1, dtype=torch.float32, device=device) - self.action_cost.mean(
            perturbed_states, perturbed_action))
        cost_P = self.log_lam2 * (torch.tensor(self.eps2, dtype=torch.float32, device=device) - torch.mean(
            (action - perturbed_action) ** 2))
        ls = action_C + cost_P
        actor_loss = torch.mean(actor_Q+ls)
        return [actor_loss]

    def update_actor(self, states_ua, perturbed_states):
        losses = self.actor_loss(states_ua, perturbed_states)
        optimizers = [self.actor_optimizer]

        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def update_lam(self, states_ul, perturbed_states):
        _, _, action = self.actor(states_ul)
        _, _, perturbed_action = self.actor(states_ul)
        # action_C = self.log_lam1 * (self.eps1 - self.action_cost.mean(perturbed_states, perturbed_action))
        # cost_P = self.log_lam2 * (self.eps2 - torch.mean((action - perturbed_action) ** 2))
        # 2024-12-11 wq
        action_C = self.log_lam1 * (torch.tensor(self.eps1, dtype=torch.float32, device=device) - self.action_cost.mean(perturbed_states, perturbed_action))
        cost_P = self.log_lam2 * (torch.tensor(self.eps2, dtype=torch.float32, device=device) - torch.mean(
            (action - perturbed_action) ** 2))
        ls = ((action_C + cost_P).detach().mean())
        ls = ls.requires_grad_()
        self.lam_optimizer.zero_grad()
        ls.backward()
        self.lam_optimizer.step()

    def ucb(self, delta_m):
        # print("delta_m:shape", shape(delta_m), delta_m)
        mu, sigma = self.gp.predict(delta_m, return_std=True)
        # print(f"mu shape: {mu.shape}, sigma shape: {sigma.shape}")
        mu = torch.tensor(mu.flatten(), device=device)
        sigma = torch.tensor(sigma.flatten(), device=device)
        # print(f"mu shape: {mu.shape}, sigma shape: {sigma.shape}")
        return mu + 2 * sigma

    def attack(self, states_a):
        memory = []
        # 设置高斯分布的均值和标准差
        mean_m, std_m = 1.0, 0.1  # 乘法扰动的均值和标准差
        mean_a, std_a = 0.0, 0.1  # 加法扰动的均值和标准差
        # 初始化单个乘法扰动 Δm 和加法扰动 Δa
        delta_m0 = np.random.normal(mean_m, std_m)  # 单个值的乘法扰动
        delta_a0 = np.random.normal(mean_a, std_a)  # 单个值的加法扰动

        # 限制 Δm 在 [0.5, 1.5] 范围内
        delta_m0 = np.clip(delta_m0, 0.5, 1.5)
        # 限制 Δa 在 [-0.5, 0.5] 范围内
        delta_a0 = np.clip(delta_a0, -0.5, 0.5)

        delta_m_K = None
        k = 0
        for state in states_a:
            if k == 0:
                delta_m_k = delta_m0
            else:
                X_sample = np.random.normal(1.0, 0.2, 100).reshape(-1, 1)  # 调整为 2D 形状 [100, 1]
                X_sample = np.clip(X_sample, 0.5, 1.5)
                ucb_values = self.ucb(X_sample)
                delta_m_k = X_sample[np.argmax(ucb_values)].item()  # 取出单个值

            delta_m_k = np.clip(delta_m_k, 0.5, 1.5)
            delta_m_k = torch.tensor(delta_m_k, dtype=torch.float)

            # 使用单个状态进行扰动计算
            perturbed_state = delta_m_k * state + delta_a0

            # 计算 perturbed_actions 和 cost
            _, _, perturbed_action = self.actor(perturbed_state)
            perturbed_cost = self.action_cost.compute_cost_for_single_state(perturbed_state, perturbed_action)
            memory.append((delta_m_k.item(), perturbed_cost.item()))

            # 转换为 NumPy 数组
            delta_m_k_numpy = np.array([[delta_m_k.item()]])  # 转为形状为 [1, 1]
            cost_numpy = np.array([[perturbed_cost.item()]])

            # 训练高斯过程回归模型
            self.gp.fit(delta_m_k_numpy, cost_numpy)

            k += 1
            if k == len(states_a) - 1:
                delta_m_K = delta_m_k

        # 加法扰动的更新：在所有乘法扰动更新完之后，计算加法扰动
        delta_a0 = torch.tensor(delta_a0, dtype=torch.float32, requires_grad=True)
        state_a = states_a[-1]
        _, _, action_a = self.actor(state_a)
        cost = self.action_cost.compute_cost_for_single_state(state_a, action_a)

        perturbed_state = delta_m_K * states_a[-1] + delta_a0
        _, _, perturbed_action = self.actor(perturbed_state)
        perturbed_cost = self.action_cost.compute_cost_for_single_state(perturbed_state, perturbed_action)

        loss = F.mse_loss(perturbed_cost, cost)
        self.action_cost.zero_grad()
        loss.backward()

        delta_a_grad = delta_a0.grad.data

        # 计算新的加法扰动 delta_a1
        delta_a1 = delta_a0 + self.epsilon_a * torch.sign(delta_a_grad)  # FGSM 更新规则
        delta_a1 = delta_a1.item()  # 将 tensor 转为数值
        delta_a1 = np.clip(delta_a1, -0.5, 0.5)  # 限制在 [-0.5, 0.5]

        return delta_m_K.item(), delta_a1

    def update(self, replay_buffer):
        samples = replay_buffer.sample(self.batch_size)
        delta_m, delta_a =  self.attack(samples[0])
        # print("乘法和加法扰动分别是：",delta_m,delta_a)
        # breakpoint()
        perturbed_states = samples[0] * delta_m + delta_a
        # 2024-12-12 wq 对 perturbed_states 裁剪到 [0, 1]
        perturbed_states = torch.clamp(perturbed_states, 0, 1)
        self.update_critic(*samples)
        self.update_action_cost(*samples)
        self.update_actor(samples[0], perturbed_states)
        self.update_lam(samples[0], perturbed_states)

    def save_model(self, model_name, env_name):
        name = "./models/" + env_name + "/policy_v%d" % model_name
        torch.save(self.actor, "{}.pkl".format(name))