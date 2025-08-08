import os
import time
import torch.nn.functional as F
import numpy as np
import swanlab
from FGSM import *

from DARRLNetworkParams import ActorNet, CriticNet, ActorNet_adv, CriticNet_adv
import torch.optim as optim
from utils import get_config

parser = get_config()
args = parser.parse_args()

class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, size, device):
        self.states_buf = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.states_adv_buf = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.next_states_buf = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.actions_adv_buf = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.costs_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.device = device
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, state, state_adv, action, action_adv, reward, next_state, done, cost):
        self.states_buf[self.ptr] = state
        self.states_adv_buf[self.ptr] = state_adv
        self.next_states_buf[self.ptr] = next_state
        self.actions_buf[self.ptr] = action
        self.actions_adv_buf[self.ptr] = action_adv
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.costs_buf[self.ptr] = cost
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(states=self.states_buf[idxs],
                    states_adv=self.states_adv_buf[idxs],
                    next_states=self.next_states_buf[idxs],
                    actions=self.actions_buf[idxs],
                    actions_adv=self.actions_adv_buf[idxs],
                    rewards=self.rewards_buf[idxs],
                    dones=self.dones_buf[idxs],
                    costs=self.costs_buf[idxs]
                    )


class DRL:
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 eps1 = 0.01,
                 eps2 = 0.01,
                 hidden_dim = 256,
                 discount = 0.99,
                 batch_size = 128,
                 actor_lr = 1e-4,
                 actor_adv_lr = 1e-4,
                 critic_lr = 1e-3,
                 lam_lr = 5e-5,
                 buffer_size=int(1e6),
                 ):
        super(DRL, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_adv_lr = actor_adv_lr
        self.lam_lr = lam_lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.buffer_size = buffer_size
        # 2025-07-26 wq 防御者
        self.actor = ActorNet(self.state_dim, self.action_dim).to(self.device)
        self.critic1 = CriticNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic2 = CriticNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.critic1_target = CriticNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic2_target = CriticNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        # 2025-07-26 wq 攻击者
        self.actor_adv = ActorNet_adv(self.state_dim, self.action_dim).to(self.device)

        self.critic1_adv = CriticNet_adv(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic2_adv = CriticNet_adv(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.critic1_adv_target = CriticNet_adv(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic2_adv_target = CriticNet_adv(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.critic1_adv_target.load_state_dict(self.critic1_adv.state_dict())
        self.critic2_adv_target.load_state_dict(self.critic2_adv.state_dict())

        self.actor_adv_optimizer = optim.Adam(self.actor_adv.parameters(), lr=self.actor_adv_lr)
        self.critic1_adv_optimizer = optim.Adam(self.critic1_adv.parameters(), lr=self.critic_lr)
        self.critic2_adv_optimizer = optim.Adam(self.critic2_adv.parameters(), lr=self.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lam_lr)
        self.target_entropy = -1

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.buffer_size, self.device)
        if args.replay:
            self.replay_buffer_done = ReplayBuffer(self.state_dim, self.action_dim, self.buffer_size, self.device)

        self.lam1 = 1.0
        self.lam2 = 1.0

        self.log_lam1 = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_lam2 = torch.zeros(1, requires_grad=True, device=self.device)
        self.lam1_optimizer = optim.Adam([self.log_lam1], lr=self.lam_lr)
        self.lam2_optimizer = optim.Adam([self.log_lam2], lr=self.lam_lr)

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def update_defender(self, states, states_adv, next_states, actions, dones, rewards):

        if 0:
            print("states_adv shape: ", states_adv.shape)
            print("next_states shape: ", next_states.shape)
            print("actions shape: ", actions.shape)
            print("rewards shape: ", rewards.shape)
            print("dones shape: ", dones.shape)
            print("costs shape: ", costs.shape)

        with torch.no_grad():
            _, _, next_pi = self.actor(next_states)
        q1 = self.critic1(states_adv, actions).squeeze(-1)
        q2 = self.critic2(states_adv, actions).squeeze(-1)

        min_q_next_pi = torch.min(self.critic1_target(next_states, next_pi),
                                  self.critic2_target(next_states, next_pi)).squeeze(-1).to(self.device)

        v_backup = min_q_next_pi
        q_backup = rewards + self.gamma * (1 - dones) * v_backup
        q_backup = q_backup.to(self.device)

        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())

        if 0:
            print("q1 shape: ", q1.shape)
            print("q2 shape: ", q2.shape)
            print("next q  shape: ", min_q_next_pi.shape)
            print("qf1_loss  shape: ", qf1_loss.shape)
            print("qf2_loss  shape: ", qf2_loss.shape)

        # Update two Q network parameter
        self.critic1_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic2_optimizer.step()

        # 2025-07-26 wq actor

        if args.get:
            with torch.no_grad():
                action_adv, _, _ = self.actor_adv(states)
            states_fgsm = FGSM_vdarrl(action_adv, self.actor,
                                    states, algo=args.algo,
                                    epsilon=args.epsilon, attack_option=args.attack_option)
            mu_adv, std, pi_adv = self.actor(states_fgsm)

        mu, std, pi = self.actor(states_adv)
        # Actor loss
        min_q_pi = torch.min(self.critic1(states_adv, pi),
                             self.critic2(states_adv, pi)).squeeze(-1).to(self.device)


        if args.method == "m1":
            actor_loss = (-min_q_pi).mean()
        elif args.method == "m2":
            # 2025-07-27 wq 只有约束1
            action_no_attack, _, action_no_attack_pi = self.actor(states)
            q1_adv = self.critic1_adv(states, action_no_attack_pi).squeeze(-1)
            q2_adv = self.critic2_adv(states, action_no_attack_pi).squeeze(-1)
            action_loss = ((q1_adv + q2_adv) / 2)
            g1 = action_loss - self.eps1
            if args.lag:
                actor_loss = torch.mean(-min_q_pi + g1 * self.lam1)
            else:
                actor_loss = torch.mean(-min_q_pi + action_loss * self.lam1)
            if 0:
                print("action_loss shape: ", action_loss.shape)
                print("g1 shape: ", g1.shape)
                print("actor_loss shape: ", actor_loss.shape)
        elif args.method == "m3":
            # 2025-07-27 wq 只有约束2
            action_no_attack, _, _ = self.actor(states)
            policy_loss = F.mse_loss(mu, action_no_attack)
            policy_loss = policy_loss.squeeze(-1)
            g2 = policy_loss - self.eps2
            if args.lag:
                actor_loss = torch.mean(-min_q_pi + g2 * self.lam2)
            else:
                actor_loss = torch.mean(-min_q_pi + policy_loss * self.lam2)
            if 0:
                print("policy_loss shape: ", policy_loss.shape)
                print("g2 shape: ", g2.shape)
                print("actor_loss shape: ", actor_loss.shape)
        elif args.method == "m4":
            if args.get:
                q1_adv = self.critic1_adv(states_fgsm, pi_adv).squeeze(-1)
                q2_adv = self.critic2_adv(states_fgsm, pi_adv).squeeze(-1)
                action_loss = ((q1_adv + q2_adv) / 2)
                g1 = action_loss - self.eps1
                policy_loss = F.mse_loss(mu, mu_adv)
                # policy_loss = F.l1_loss(mu, mu_adv, reduction='none')
                swanlab.log({
                    "train/action_before_attack_mean": round(mu.mean().item(), 2),
                    "train/attack_action_mean": round(action_adv.mean().item(), 2),
                    "train/action_after_attack_mean": round(mu_adv.mean().item(), 2),
                })
                policy_loss = policy_loss.squeeze(-1)
                g2 = policy_loss - self.eps2
                if args.lag:
                    actor_loss = torch.mean(-min_q_pi + g1 * self.lam1 + g2 * self.lam2)
                else:
                    actor_loss = torch.mean(-min_q_pi + action_loss * self.lam1 + policy_loss * self.lam2 )

            else:
                if args.grad:
                    action_no_attack, _, action_no_attack_pi = self.actor(states)
                    swanlab.log({
                        "train/action_before_attack_mean": round(action_no_attack.mean().item(), 2),
                        # "train/attack_action_mean": round(action_adv.mean().item(), 2),
                        "train/action_after_attack_mean": round(mu.mean().item(), 2),
                    })
                else:
                    with torch.no_grad():
                        action_no_attack, _, action_no_attack_pi = self.actor(states)
                        if args.swanlab:
                            swanlab.log({
                                "train/action_before_attack_mean": round(action_no_attack.mean().item(), 2),
                                # "train/attack_action_mean": round(action_adv.mean().item(), 2),
                                "train/action_after_attack_mean": round(mu.mean().item(), 2),
                            })

                # 2025-07-27 wq 全约束
                if args.critic:
                    q1_adv = self.critic1_adv(states_adv, pi).squeeze(-1)
                    q2_adv = self.critic2_adv(states_adv, pi).squeeze(-1)
                else:
                    q1_adv = self.critic1_adv(states, action_no_attack_pi).squeeze(-1)
                    q2_adv = self.critic2_adv(states, action_no_attack_pi).squeeze(-1)
                action_loss = ((q1_adv + q2_adv) / 2)
                g1 = action_loss - self.eps1

                policy_loss = F.mse_loss(mu, action_no_attack)
                policy_loss = policy_loss.squeeze(-1)
                g2 = policy_loss - self.eps2

                if args.lag:
                    actor_loss = torch.mean(-min_q_pi + g1 * self.lam1 + g2 * self.lam2)
                else:
                    actor_loss = torch.mean(-min_q_pi + action_loss * self.lam1 + policy_loss * self.lam2 )
                if 0:
                    print("action_loss shape: ", action_loss.shape)
                    print("g1 shape: ", g1.shape)
                    print("policy_loss shape: ", policy_loss.shape)
                    print("g2 shape: ", g2.shape)
                    print("actor_loss shape: ", actor_loss.shape)

        # Update actor network parameter
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 2025-07-27 wq 记录
        if args.swanlab:
            actor_loss_val = -actor_loss.detach().cpu().item()
            swanlab.log({"loss/agent_loss": actor_loss_val})
        if args.swanlab and (args.method == "m2" or args.method == "m4"):
            action_value = action_loss.mean().detach().cpu().item()
            g1_value = g1.mean().detach().cpu().item()
            swanlab.log({"loss/action_loss": action_value})
            swanlab.log({"loss/lam1": self.lam1})
            swanlab.log({"loss/g1": g1_value})

            action_C = -(self.log_lam1 * g1.detach()).mean()
            self.lam1_optimizer.zero_grad()
            action_C.backward()
            self.lam1_optimizer.step()
            self.lam1 = self.log_lam1.exp()

        if args.swanlab and (args.method == "m3" or args.method == "m4"):
            policy_value = policy_loss.mean().detach().cpu().item()
            g2_value = g2.mean().detach().cpu().item()
            swanlab.log({"loss/policy_loss": policy_value})
            swanlab.log({"loss/lam2": self.lam2})
            swanlab.log({"loss/g2": g2_value})

            action_P = -(self.log_lam2 * g2.detach()).mean()
            self.lam2_optimizer.zero_grad()
            action_P.backward()
            self.lam2_optimizer.step()
            self.lam2 = self.log_lam2.exp()

        if 0:
            print("min_q_pi shape: ", min_q_pi.shape)
            print("actor_loss shape: ", actor_loss.shape)

        # Polyak averaging for target parameter
        self.soft_target_update(self.critic1, self.critic1_target)
        self.soft_target_update(self.critic2, self.critic2_target)


    def update_attacker(self, states, next_states, actions, dones, costs):
        with torch.no_grad():
            _, next_log_prob, next_pi = self.actor_adv(next_states)
            entropy = -next_log_prob.squeeze(-1)

        if 0:
            print("next_log_prob shape: ", next_log_prob.shape)
            print("entropy shape: ", entropy.shape)

        q1_adv = self.critic1_adv(states, actions).squeeze(-1)
        q2_adv = self.critic2_adv(states, actions).squeeze(-1)

        # print("BEFORE BACKWARD:")
        # print("  q1_adv:", q1_adv, " grad_fn:", q1_adv.grad_fn)
        # print("  q2_adv:", q2_adv, " grad_fn:", q2_adv.grad_fn)

        min_q_next_pi_adv = torch.min(self.critic1_adv_target(next_states, next_pi),
                                      self.critic2_adv_target(next_states, next_pi)).squeeze(-1).to(self.device)

        next_values_adv = min_q_next_pi_adv + self.log_alpha.exp() * entropy
        v_backup_adv = next_values_adv
        q_backup_adv = costs + self.gamma * dones  * v_backup_adv
        q_backup_adv = q_backup_adv.to(self.device)

        qf1_loss_adv = F.mse_loss(q1_adv, q_backup_adv.detach())
        qf2_loss_adv = F.mse_loss(q2_adv, q_backup_adv.detach())

        if 0:
            print("q1_adv shape: ", q1_adv.shape)
            print("q2_adv shape: ", q2_adv.shape)
            print("next_values_adv  shape: ", next_values_adv.shape)
            print("qf1_loss_adv  shape: ", qf1_loss_adv.shape)
            print("qf2_loss_adv  shape: ", qf2_loss_adv.shape)

        # Update two Q network parameter
        self.critic1_adv_optimizer.zero_grad()
        qf1_loss_adv.backward()
        self.critic1_adv_optimizer.step()

        self.critic2_adv_optimizer.zero_grad()
        qf2_loss_adv.backward()
        self.critic2_adv_optimizer.step()

        # 2025-07-26 wq adv actor
        _, log_prob, new_actions = self.actor_adv(states)
        entropy = -log_prob.squeeze(-1)

        q1_value_adv = self.critic1_adv(states, new_actions).squeeze(-1)
        q2_value_adv = self.critic2_adv(states, new_actions).squeeze(-1)

        actor_loss_adv = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value_adv, q2_value_adv))

        if 0:
            print("entropy shape: ", entropy.shape)
            print("q2_value_adv shape: ", q2_value_adv.shape)
            print("actor_loss_adv shape: ", actor_loss_adv.shape)

        if args.swanlab:
            actor_loss_adv_value = -actor_loss_adv.detach().cpu().item()
            swanlab.log({"loss/attack_loss": actor_loss_adv_value})

        self.actor_adv_optimizer.zero_grad()
        actor_loss_adv.backward()
        self.actor_adv_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_target_update(self.critic1_adv, self.critic1_adv_target)
        self.soft_target_update(self.critic2_adv, self.critic2_adv_target)



    def update(self, attacker_flag):
        batch = self.replay_buffer.sample(self.batch_size)

        states = batch['states']
        states_adv = batch['states_adv']
        next_states = batch['next_states']
        actions = batch['actions']
        actions_adv = batch['actions_adv']
        rewards = batch['rewards']
        dones = batch['dones']
        costs = batch['costs']

        if args.attacker:
            self.update_attacker(states, next_states, actions_adv, dones, costs)
        else:
            if args.frequency:
                if attacker_flag:
                    self.update_attacker(states, next_states, actions_adv, dones, costs)
                else:
                    self.update_defender(states, states_adv, next_states, actions, dones, rewards)
            else:
                self.update_attacker(states, next_states, actions, dones, costs)
                self.update_defender(states, states_adv, next_states, actions, dones, rewards)


    def train(self, mode: bool = True):
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        self.actor_adv.train(mode)
        self.critic1_adv.train(mode)
        self.critic2_adv.train(mode)


        return self

    def save_model(self, model_name,  modelSavedPath):
        timestamp = time.strftime("%Y%m%d_%H%M")

        # 假设 modelSavedPath, model_name, timestamp, self.actor 都已定义
        if args.get:
            # 保存攻击者
            attacker_path = os.path.join(modelSavedPath, "attacker")
            os.makedirs(attacker_path, exist_ok=True)
            name_att = f"attacker_v{model_name}_{timestamp}.pth"
            save_path_att = os.path.join(attacker_path, name_att)
            torch.save(self.actor_adv.state_dict(), save_path_att)

            # 保存防御者
            defender_path = os.path.join(modelSavedPath, "defender")
            os.makedirs(defender_path, exist_ok=True)
            name_def = f"defender_v{model_name}_{timestamp}.pth"
            save_path_def = os.path.join(defender_path, name_def)
            torch.save(self.actor.state_dict(), save_path_def)

        elif args.attacker:
            # 只保存攻击者
            attacker_path = os.path.join(modelSavedPath, "attacker")
            os.makedirs(attacker_path, exist_ok=True)
            name_att = f"attacker_v{model_name}_{timestamp}.pth"
            save_path_att = os.path.join(attacker_path, name_att)
            torch.save(self.actor_adv.state_dict(), save_path_att)

        else:
            # 都没有参数时只保存防御者
            defender_path = os.path.join(modelSavedPath, "defender")
            os.makedirs(defender_path, exist_ok=True)
            name_def = f"defender_v{model_name}_{timestamp}.pth"
            save_path_def = os.path.join(defender_path, name_def)
            torch.save(self.actor.state_dict(), save_path_def)


