import gym
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import Environment.environment
from dr import DR
import os
import pandas as pd
from datetime import datetime
import torch
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="TrafficEnv3-v0", help='name of the environment to run')
parser.add_argument('--seed', default=1)
parser.add_argument('--train_step', default=100)
parser.add_argument('--T_horizon', default=30)
parser.add_argument('--print_interval', default=10)
parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
parser.add_argument('--state_dim', default=26)
parser.add_argument('--action_dim', default=1)
args = parser.parse_args()


# Set environment
env = gym.make(args.env_name)

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

model_dir_base = os.getcwd() + '/models/' + args.env_name
train_result_dir = os.getcwd() + '/results/' + args.env_name

if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

if not os.path.exists(train_result_dir):
    os.makedirs(train_result_dir)


def train():
    env.start(gui=False)
    alg_cfg = DR.Config()
    alg = DR(alg_cfg, env, state_dim=args.state_dim, action_dim=args.action_dim)
    # 20241210 wq 跟踪当前episode的总奖励
    score = 0.0
    total_reward = []
    episode = []
    #20241210 wq
    v = []
    v_epi = []
    v_epi_mean = []
    # 20241210 wq 碰撞
    cn = 0.0
    cn_epi = []
    # 20241210 wq 成功次数
    sn = 0.0
    sn_epi = []

    for n_epi in range(args.train_step):

        score, v, v_epi, xa, ya, done = alg.interaction(args.T_horizon, score, v, v_epi, args.speed_range, args.max_a, n_epi)

        print("训练episode:",n_epi,"奖励:",score)
        # 2024-12-12 wq cn碰撞的次数，sn成功的次数
        if done is True:
            cn += 1

        if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0':
            if xa < -50.0 and ya > 4.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv2-v0':
            if xa > 50.0 and ya > -5.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv4-v0':
            if ya < -50.0 and done is False:
                sn += 1
        # 2024-12-10 wq 打印统计信息
        if (n_epi+1) % args.print_interval == 0:
            print("# of episode :{}, avg score_v : {:.1f}".format(n_epi+1, score / args.print_interval))

            episode.append(n_epi+1)
            total_reward.append(score / args.print_interval)
            cn_epi.append(cn/args.print_interval)
            sn_epi.append(sn/args.print_interval)
            print("######cn & sn rate:", cn/args.print_interval, sn/args.print_interval)

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            cn = 0.0
            sn = 0.0

    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    train_res = f"{train_result_dir}/train_data_{current_time}.csv"
    df.to_csv(train_res, index=0)

    #plt.plot(episode, total_reward)
    #plt.xlabel('episode')
    #plt.ylabel('total_reward')
    ## 2024-12-13 wq  创建保存图片的目录（如果不存在）
    #image_dir = os.path.join(train_result_dir, "images")
    #os.makedirs(image_dir, exist_ok=True)

    ## 2024-12-13 wq
    # 保存图片，文件名加上当前日期和时间
    #image_path = os.path.join(image_dir, f"reward_plot_{current_time}.png")
    #plt.savefig(image_path)


    env.close()


if __name__ == "__main__":
    train()
