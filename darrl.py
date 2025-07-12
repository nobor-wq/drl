import copy
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

from config import BaseConfig, Configurable
from torch_util import  Module, update_ema, freeze_module
from DARRLNetworkParams import QNetwork, PolicyNetwork, PolicyNetContinuous_adv, QValueNetContinuous_adv
import csv

def pythonic_mean(x):
    return sum(x) / len(x)

class DARRL(Configurable, Module):
    class Config(BaseConfig):
        discount = 0.99
        deterministic_backup = False
        critic_update_multiplier = 1
        critic_cfg = QNetwork.Config()
        critic_cfg_adv = QNetwork.Config()
        tau = 0.005
        batch_size = 128
        hidden_dim = 256
        hidden_layers = 2
        actor_lr = 1e-4
        actor_lr_adv = 1e-4
        lam_lr = 1e-4

    def __init__(self, config, state_dim, action_dim, Advaction_dim, device, optimizer_factory=torch.optim.Adam):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.device = device

        self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.critic = QNetwork(self.critic_cfg, state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        freeze_module(self.critic_target)

        self.actor_adv = PolicyNetContinuous_adv(state_dim, Advaction_dim).to(self.device)
        self.critic_adv = QNetwork(self.critic_cfg_adv, state_dim, Advaction_dim).to(self.device)
        self.critic_target_adv = copy.deepcopy(self.critic_adv).to(self.device)
        freeze_module(self.critic_target_adv)

        self.gamma = config.discount
        self.tau = config.tau

        self.actor_lr = config.actor_lr
        self.actor_lr_adv = config.actor_lr_adv
        self.lam_lr = config.lam_lr

        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)
        self.actor_optimizer_adv = optimizer_factory(self.actor_adv.parameters(), lr=self.actor_lr_adv)

        self.eps1 = 0.5
        self.eps2 = 0.5
        # self.eps3 = 1.0

        self.lam1 = 0.1
        self.lam2 = 0.1
        # self.lam3 = 0.1

        self.log_lam1 = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_lam2 = torch.zeros(1, requires_grad=True, device=self.device)
        # self.log_lam3 = torch.zeros(1, requires_grad=True, device=self.device)
        self.lam1_optimizer = optimizer_factory([self.log_lam1], lr=self.lam_lr)
        self.lam2_optimizer = optimizer_factory([self.log_lam2], lr=self.lam_lr)
        # self.lam3_optimizer = optimizer_factory([self.log_lam3], lr=self.lam_lr)

        self.target_entropy = -1
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device)
        self.log_alpha_optimizer = optimizer_factory([self.log_alpha], lr=self.lam_lr)

        self.epsilon = 0.03
        self.attack_flag = True

        self.criterion = nn.MSELoss()





    def update_critic(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            _, _, next_ego_actions = self.actor(next_states)
            next_values = self.critic_target.min(next_states, next_ego_actions).to(self.device)
            target_c = rewards + self.gamma * (1. - dones.float()) * next_values

        qs = self.critic.all(states, actions)
        critic_loss = pythonic_mean([self.criterion(q, target_c) for q in qs])
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    # def get_loss(self, states, actions_mu):
    #     # actions, _, _  = self.actor(states)
    #     with torch.no_grad():  # 阻止梯度传播
    #         action_adv, _, _ = self.actor_adv(states)
    #     dis_states = FGSM_vdarrl(action_adv, self.actor, states, epsilon=self.epsilon, algo="drl")
    #     dis_states = dis_states.detach()  # 分离计算图
    #     dis_actions_mu, _, dis_actions = self.actor(dis_states)
    #     action_loss = self.critic_adv.mean(dis_states, dis_actions)
    #
    #     policy_loss = F.mse_loss(dis_actions_mu, actions_mu, reduction='none')
    #     if 0:
    #         print("actions_mu:", actions_mu.shape)
    #         print("dis_actions_mu:", dis_actions_mu.shape)
    #         print("dis_actions:", dis_actions.shape)
    #         print("action_loss:", action_loss.shape)
    #         print("policy_loss:", policy_loss.shape)
    #
    #     return action_loss, policy_loss.squeeze(-1)

    def update_actor(self, states, adv_state, loss_path):
        actions_mu, _, actions = self.actor(adv_state)
        # dis_actions_mu, _, _ = self.actor_adv(states)

        action_loss = self.critic_adv.mean(adv_state, actions)
        # policy_loss = F.mse_loss(dis_actions_mu, actions_mu)
        actor_Q = self.critic.min(adv_state, actions)

        # actor_loss = torch.mean(-actor_Q)
        actor_loss = torch.mean(-actor_Q + action_loss * self.lam1)

        # actor_loss = torch.mean(-actor_Q + action_loss * self.lam1 + policy_loss * self.lam2)

        # lam1_val = (self.lam1.detach().cpu().item()
        #             if torch.is_tensor(self.lam1) else float(self.lam1))
        # lam2_val = (self.lam2.detach().cpu().item()
        #             if torch.is_tensor(self.lam2) else float(self.lam2))
        with open(loss_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                # action_loss.mean().detach().cpu().item(),
                # lam1_val,
                # 取 actions_mu 在整个 batch+维度上的平均
                actions_mu.mean().detach().cpu().item(),
                # dis_actions_mu.mean().detach().cpu().item(),
                # policy_loss.detach().cpu().item(),  # F.mse_loss 默认就是 mean
                # lam2_val,
                actor_loss.detach().cpu().item()
            ])

        # actor_loss = torch.mean(-actor_Q  + policy_loss * self.lam2)
        # actor_loss = torch.mean(-actor_Q + action_loss * self.lam1)
        if 0:
            print("action loss shape:", action_loss.shape)
            print("policy loss shape:", policy_loss.shape)
            print("actor_Q shape:", actor_Q.shape)
            print("actor loss:", actor_loss.shape)
        losses = [actor_loss]
        optimizers = [self.actor_optimizer]
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if self.attack_flag:
            # 2025-05-16 wq lam更新
        action_C =  -(self.log_lam1 * (action_loss - self.eps1).detach()).mean()
        self.lam1_optimizer.zero_grad()
        action_C.backward()
        self.lam1_optimizer.step()
        self.lam1 = self.log_lam1.exp()

        # action_P = -(self.log_lam2 * (policy_loss - self.eps2).detach()).mean()
        # self.lam2_optimizer.zero_grad()
        # action_P.backward()
        # self.lam2_optimizer.step()
        # self.lam2 = self.log_lam2.exp()



    def update_critic_adv(self, states,  actions,  next_states, rewards, dones):
        with torch.no_grad():
            _, next_adv_actions, log_prob = self.actor_adv(next_states)
            entropy = -log_prob.squeeze(-1)
            next_critic_values = self.critic_target_adv.min(next_states, next_adv_actions)
            next_values = next_critic_values + self.log_alpha.exp() * entropy
            target = rewards + self.gamma * dones.float() * next_values
            if 0:
                print("[DEBUG] next_critic_values:", next_critic_values.shape)
                print("[DEBUG] entropy:", entropy.shape)
                print("[DEBUG] next_values:", next_values.shape)
                print("[DEBUG] update_critic_adv target:", target.shape)

        qs = self.critic_adv.all(states, actions)
        # for i, q in enumerate(qs):
        #     print(f"[DEBUG] update_critic_adv q[{i}]:", q.shape)
        critic_loss_adv = pythonic_mean([self.criterion(q, target) for q in qs])
        self.critic_adv.optimizer.zero_grad()
        critic_loss_adv.backward()
        self.critic_adv.optimizer.step()
        update_ema(self.critic_target_adv, self.critic_adv, self.tau)
        return critic_loss_adv.detach()

    def update_actor_adv(self, states):
        _, a_actions, log_prob  = self.actor_adv(states)
        entropy = -log_prob
        actor_Q_adv = self.critic_adv.min(states, a_actions)
        actor_loss_adv = torch.mean(-actor_Q_adv - self.log_alpha.exp() * entropy)
        if 0:
            print("a_actions_adv shape:", a_actions.shape)
            print("actor_loss_adv shape:", actor_loss_adv.shape)
        losses = [actor_loss_adv]
        optimizers = [self.actor_optimizer_adv]
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 2025-05-17 wq 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # actor_C = (self.log_lam3 * (actor_loss - self.eps3).detach()).mean()
        # self.lam3_optimizer.zero_grad()
        # actor_C.backward()
        # self.lam3_optimizer.step()
        # self.lam3 = self.log_lam3.exp()


    def update(self, replay_buffer, eps, loss_path):


        self.epsilon = eps

        samples = replay_buffer.sample(self.batch_size)
        states, adv_state, actions, adv_actions, next_states, rewards, costs, dones = samples

        self.update_critic_adv(states, adv_actions, next_states, costs, dones)
        self.update_actor_adv(states)
        self.update_critic(adv_state, actions,  next_states, rewards, dones)
        self.update_actor(states, adv_state, loss_path)



    def save_model(self, model_name,  modelSavedPath):
        timestamp = time.strftime("%Y%m%d_%H%M")
        # if self.attack_flag:
        attacker_path = modelSavedPath + "/attacker/"
        if not os.path.exists(attacker_path):
            os.makedirs(attacker_path)
        fname_att = f"policy_v{model_name}_{timestamp}.pkl"
        torch.save(self.actor_adv, os.path.join(attacker_path, fname_att))

        defender_path = modelSavedPath + "/defender/"
        if not os.path.exists(defender_path):
            os.makedirs(defender_path)

        fname_def = f"policy_v{model_name}_{timestamp}.pkl"
        torch.save(self.actor, os.path.join(defender_path, fname_def))

