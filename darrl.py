import copy
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from torch import nn
from config import BaseConfig, Configurable
from torch_util import device, Module, update_ema, freeze_module
from DARRLNetworkParams import QNetwork, PolicyNetwork, ActionCostNetwork
import torch.nn.functional as F

def pythonic_mean(x):
    return sum(x) / len(x)

class DARRL(Configurable, Module):
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
        actor_lr = 1e-4
        acost_lr = 1e-4
        lam_lr = 1e-4

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

        self.gamma = config.discount
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.eps1 = torch.tensor(2.0, dtype=torch.float32, device=device)
        self.eps2 = torch.tensor(2.0, dtype=torch.float32, device=device)
        self.actor_lr = config.actor_lr
        self.lam_lr = config.lam_lr

        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)
        self.action_loss = None

        self.lam1 = torch.zeros(1, requires_grad=True, device=device)
        self.log_lam1 = torch.zeros(1, requires_grad=True, device=device)
        self.lam2 = torch.zeros(1, requires_grad=True, device=device)
        self.log_lam2 = torch.zeros(1, requires_grad=True, device=device)
        self.lam_optimizer = optimizer_factory([self.log_lam1], lr=self.lam_lr)

        self.criterion = nn.MSELoss()

    def critic_loss(self, states_cl, action, next_states, reward, cost, done):
        target = self.compute_q_target(next_states, reward, done)
        return self.critic_loss_given_target(states_cl, action, target)


    def compute_q_target(self, next_states, reward, done):
        with torch.no_grad():
            _, _, next_action = self.actor(next_states)
            next_value = self.critic_target.min(next_states, next_action)
            q = reward + self.gamma * (1. - done.float()) * next_value
            return q

    def critic_loss_given_target(self, states_clgt, action, target):
        qs = self.critic.all(states_clgt, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

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
            return c

    def action_loss_given_target(self, states_algt, action, target):
        cs = self.action_cost.all(states_algt, action)
        return pythonic_mean([self.criterion(c, target) for c in cs])

    def update_action_cost(self, *action_loss_args):
        action_cost_loss = self.action_cost_loss(*action_loss_args)
        self.action_cost.optimizer.zero_grad()
        action_cost_loss.backward()
        self.action_cost.optimizer.step()
        update_ema(self.action_cost_target, self.action_cost, self.tau)
        return action_cost_loss.detach()

    def actor_loss(self, states_al):
        _, _, action = self.actor(states_al)
        actor_Q = self.critic.min(states_al, action)
        action_C = self.lam1 * (self.eps1 - self.action_loss)
        actor_loss = torch.mean(-(actor_Q + action_C))
        return [actor_loss]

    def update_actor(self, states_ua):
        losses = self.actor_loss(states_ua)
        optimizers = [self.actor_optimizer]

        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    def update_lam(self, states_ul):
        _, _, action = self.actor(states_ul)
        # 2024-12-11 wq
        action_C = self.log_lam1 * (self.eps1 - self.action_loss).mean()
        self.lam_optimizer.zero_grad()
        action_C.backward()
        self.lam_optimizer.step()

        self.lam1 = self.log_lam1.exp().detach()

    def update(self, replay_buffer):

        samples = replay_buffer.sample(self.batch_size)

        self.update_critic(*samples)
        self.update_action_cost(*samples)
        states, _, _, _, _, _ = samples
        self.action_loss = self.action_cost.mean(states, self.actor(states)[2])
        self.update_actor(states)
        self.update_lam(states)

    def save_model(self, model_name, env_name):
        name = "./models/" + env_name + "/policy_v%d" % model_name
        torch.save(self.actor, "{}.pkl".format(name))