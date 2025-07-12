import argparse

import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
import numpy as np

from gymnasium.wrappers import TimeLimit
import gymnasium as gym
# import gym
# from gym.wrappers import TimeLimit

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.type_aliases import MaybeCallback
import Environment.environment
from typing import Optional
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import pandas as pd
import os
from FGSM import *
from stable_baselines3.common.buffers import RolloutBuffer
import wandb
from wandb.integration.sb3 import WandbCallback
import random


parser = argparse.ArgumentParser()
parser.add_argument('--addition_msg', default="", help='additional message of the training process')
parser.add_argument('--algo', default="PPO", help='training algorithm')
parser.add_argument('--adv_algo', default="PPO", help='training adv algorithm')
parser.add_argument('--action_dim', default=1)
parser.add_argument('--age_path', default="./logs/age_eval/")
parser.add_argument('--adv_path', default="./logs/adv_eval/")
parser.add_argument('--attack_method', default="fgsm", help='which attack method to be applied')
parser.add_argument('--attack', type=bool, default=True, help='control n_rollout_steps, for PPO')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--env_name', default="TrafficEnv3-v1", help='name of the environment to run')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
parser.add_argument('--n_steps', type=int, default=256, help='control n_rollout_steps, for PPO')
parser.add_argument('--print_interval', default=10)
parser.add_argument('--run_id',  default="", help='')
parser.add_argument('--seed', type=int, default=7, help='random seed for network')
parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
parser.add_argument('--state_dim', default=26)
parser.add_argument('--save_freq', type=int, default=10000, help='frequency of saving the model')
parser.add_argument('--T_horizon', type=int, default=30, help='number of training steps per episode')
parser.add_argument('--train_step', type=int, default=500, help='number of training episodes')

args = parser.parse_args()

def loadEnv(attack, log_path1):
    #, adv_path, adv_algo, seed, save_freq
    # Create environment
    env = gym.make(args.env_name, attack=attack)
    env = TimeLimit(env, max_episode_steps=args.T_horizon)
    env = Monitor(env)
    env.unwrapped.start(gui=False)

    # log path
    eval_log_path = log_path1  + os.path.join(args.env_name, args.adv_algo)
    os.makedirs(eval_log_path, exist_ok=True)
    # 设置设备
    # if  th.cuda.is_available():
    #     pass
    # else:
    device = th.device("cpu")

    # 设置随机种子
    random.seed(args.seed)  # 设置 Python 随机种子
    np.random.seed(args.seed)  # 设置 NumPy 随机种子
    th.manual_seed(args.seed)  # 设置 CPU 随机种子
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
        th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
    th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
    th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

    model_path = os.path.join(eval_log_path, 'model')
    os.makedirs(model_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=model_path)
    args.addition_msg = "ppo_only"
    run_name = f"{args.attack_method}-{args.algo}-{args.addition_msg}"

    run = wandb.init(project="run_result", name=run_name, config=args, sync_tensorboard=True)
    args.run_id = run.id
    model = PPO("MlpPolicy", env, verbose=1, device=device, learning_rate=args.lr,
            batch_size=args.batch_size, n_epochs=args.n_steps, tensorboard_log=f"runs/{args.run_id}")
    wandb_callback = WandbCallback(gradient_save_freq=5, verbose=2, model_save_path=f"models/{run.id}")
    model.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                callback=[checkpoint_callback, wandb_callback], reset_num_timesteps=False)




    # flag = False
    # # init wandb
    # if flag == True :
    #     run_name = f"{args.attack_method}-{args.algo}-{args.addition_msg}"
    #     run = wandb.init(project="run_result", name=run_name, config=args, sync_tensorboard=True)
    #     model = AdversarialPPO(args.algo, log_path2, args.env_name,  "MlpPolicy", env, n_steps=args.n_steps, verbose=1,
    #                        tensorboard_log=f"runs/{run.id}", rollout_buffer_class=rollout_buffer_class , device=device)
    #     wandb_callback = WandbCallback(gradient_save_freq=5, verbose=2, model_save_path=f"models/{run.id}",)
    #     model.learn(total_timesteps=args.train_step*args.n_steps, progress_bar=True, callback=[checkpoint_callback, wandb_callback])
    # else:
    #     model = AdversarialPPO(args.algo, log_path2, args.env_name,  "MlpPolicy", env, n_steps=args.n_steps, verbose=1,
    #                         rollout_buffer_class=rollout_buffer_class , device=device)
    #     model.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
    #                     callback=checkpoint_callback)

    # 绘制评估奖励曲线
    plt.plot(env.get_episode_rewards())
    plt.title('Rewards per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(eval_log_path, "rewards.png"), dpi=300)
    plt.close()

    plt.plot(env.get_episode_lengths())
    plt.title('Steps per episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig(os.path.join(eval_log_path, "steps.png"), dpi=300)
    plt.close()

    reward_df = pd.DataFrame(env.get_episode_rewards())
    step_df = pd.DataFrame(env.get_episode_lengths())
    reward_df.to_csv(os.path.join(eval_log_path, "rewards.csv"), index=False)
    step_df.to_csv(os.path.join(eval_log_path, "steps.csv"), index=False)
    # Save the log

    # Save the agent
    model.save(os.path.join(eval_log_path, "lunar"))
    del model  # delete trained model to demonstrate loading
    env.close()
    # wandb.finish()


loadEnv(False, args.age_path)






