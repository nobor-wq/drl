import os
import gym
import torch
import argparse
import numpy as np
import pandas as pd
import random as rn
import torch.nn.functional as F

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import matplotlib.pyplot as plt
import Environment.environment
from drac import ActorNet, CostNet  # 假设网络定义在 your_module.py

device = torch.device("cpu")

# 1. 先创建网络结构
actor = ActorNet(26, 1, min_log_std=-5.0, max_log_std=2.0).to(device)
cf1   = CostNet(26, 1, 256).to(device)
cf2   = CostNet(26, 1, 256).to(device)

# 2. 拼出文件路径
model_name = "2000"
model_path = "models/TrafficEnv3-v1/DARRL/defender"
base = os.path.join(model_path, f"policy{model_name}")

actor_path = f"{base}_actor.pth"
cf1_path   = f"{base}_cf1.pth"
cf2_path   = f"{base}_cf2.pth"

# 3. 加载权重
actor.load_state_dict(torch.load(actor_path, map_location=device))
cf1.load_state_dict(  torch.load(cf1_path,   map_location=device))
cf2.load_state_dict(  torch.load(cf2_path,   map_location=device))


print("权重加载完成，actor、cf1、cf2 已就绪。")


parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--action_dim', type=int, default=1, help='action_dim')
parser.add_argument('--state_dim', type=int, default=26, help='state_dim')
parser.add_argument('--action_limit', type=int, default=7.6, help='action limit')
parser.add_argument('--hidden_sizes', type=int, default=256, help='hidden_sizes')
parser.add_argument('--T_horizon', type=int, default=30, help='rollout')
parser.add_argument('--train_step', type=int, default=100, help='train_step')
parser.add_argument('--mode', type=str, default='train', help='train')
parser.add_argument('--speed_range', type=int, default=15.0, help='speed range')
parser.add_argument('--save_dir_model', type=str, default='model/', help='the path to save models')
parser.add_argument('--save_dir_data', type=str, default='result/', help='the path to save data')
parser.add_argument('--save_dir_train_data', type=str, default='train/', help='the path to save training data')
parser.add_argument('--wandb', type=bool, default=False, help='whether to use wandb logging')

args = parser.parse_args()

# Set the environment
env = gym.make('traffic-v0')

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)

state_batch = None
attack_flag = True

def attacker_constraint(state_batch,  k=0.05, epsilon=0.05):
    target_list_grads = []
    perturb_list = []

    pbounds = {'pert1': (0.95, 1.05)}
    optimizer = BayesianOptimization(f=cost_pert_black_attack, pbounds=pbounds, random_state=0)
    util = UtilityFunction(kind='ucb', kappa=0.2, xi=0.0)
    for i in range(5):
        probe_para = optimizer.suggest(util)
        target = cost_pert_black_attack(**probe_para)
        # optimizer.register(probe_para, target.item())
        try:
            optimizer.register(probe_para, target.item())
        except KeyError:
            # 已经有了，就跳过
            continue
        target_list_grads.append(target)
        perturb_list.append(probe_para)

    optimal_perturb = perturb_list[target_list_grads.index(max(target_list_grads))]
    pert1 = optimal_perturb['pert1']

    if state_batch.ndim == 1:
        state_batch = state_batch.unsqueeze(0)  # 变成 [1, state_dim]

    pert2 = k * torch.randn(state_batch.shape[0], state_batch.shape[1])
    pert2.requires_grad = True

    mu_pert, std_pert, pi_pert = actor(pert1 * state_batch + pert2)
    cost_pert = 0.5 * (cf1(pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device) + cf2(
        pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device))

    # Calculate gradient
    cost_pert.mean().backward(retain_graph=True)

    # Calculate sign of gradient
    signs = torch.sign(pert2.grad)

    # Add
    pert2 = pert2 + epsilon * signs

    # new loss
    # mu_pert, std_pert, pi_pert = actor(pert1 * state_batch + pert2)
    # cost_pert = 0.5 * (cf1(pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device) + cf2(
    #     pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device))
    #
    # mu, std, pi = actor(state_batch)
    # a_cost = F.mse_loss(mu_pert, mu)

    return pert1, pert2


def cost_pert_black_attack(pert1):
    mu_pert, std_pert, pi_pert = actor(pert1 * state_batch)
    cost_pert = 0.5 * (cf1(pert1 * state_batch, pi_pert).squeeze(1).to(device) + cf2(
        pert1 * state_batch, pi_pert).squeeze(1).to(device))

    return cost_pert.mean()



for i in range(1):
    env.start(gui=False)

    score = 0.0
    total_reward = []
    episode = []
    print_interval = 10

    v = []
    v_epi = []
    v_epi_mean = []

    cn = 0.0
    cn_epi = []

    sn = 0.0
    sn_epi = []

    for n_epi in range(args.train_step):
        s = env.reset()

        for t in range(args.T_horizon):
            s = torch.FloatTensor(s).to(device)
            if attack_flag:
                state_batch = s
                m, a = attacker_constraint(state_batch)
                print("正常状态：",s)
                s = m*s + a
                print("m ,a :",m, a)
                print("扰动后的状态：",s)

            a, _, _ = actor(s)
            r, next_s, done, r_, c_, info = env.step(a)
            s = next_s


            score += r
            v.append(s[24]*args.speed_range)
            v_epi.append(s[24]*args.speed_range)
            xa = info[0]
            ya = info[1]

            if done:
                break

        if done is True:
            cn += 1

        if xa < -50 and ya > 4.0 and done is False:
            sn += 1

        if (n_epi+1) % print_interval == 0:
            print("# of episode :{}, avg score_v : {:.1f}".format(n_epi+1, score / print_interval))

            episode.append(n_epi+1)
            total_reward.append(score / print_interval)
            cn_epi.append(cn/print_interval)
            sn_epi.append(sn/print_interval)
            print("######cn & sn rate:", cn/print_interval, sn/print_interval)

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            cn = 0.0
            sn = 0.0

    plt.plot(episode, total_reward)
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.show()

    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi

    if not os.path.exists(args.save_dir_data):
        os.mkdir(args.save_dir_data)
    train_data_path = os.path.join(args.save_dir_data, args.save_dir_train_data)
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    df.to_csv('./' + train_data_path + '/train.csv', index=0)
    env.close()


