import gymnasium as  gym
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import Environment.environment
from dr import DR
import os
import pandas as pd
import datetime
import torch
import wandb
from utils import get_config
import swanlab
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = get_config()
args = parser.parse_args()

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


device = torch.device(args.device)

if args.wandb:
    run_name = f"{args.env_name}-{args.algo}-{args.addition_msg}"
    run = swanlab.init(project="run_result", name=run_name, config=args)


# Set environment
env = gym.make(args.env_name)

random.seed(args.seed)  # 设置 Python 随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
torch.manual_seed(args.seed)  # 设置 CPU 随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
torch.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化



current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
model_dir_base = os.getcwd() + '/models/' + args.env_name + f'/log-{current_time}/'
train_result_dir =model_dir_base + '/results/'

if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

if not os.path.exists(train_result_dir):
    os.makedirs(train_result_dir)

loss_path = os.path.join(train_result_dir, "loss_history.csv")
#'action_loss', 'lam1', , 'policy_loss', 'lam2', 'adv_action'
header = [ 'age_action', 'actor_loss']
write_header = not os.path.exists(loss_path)
with open(loss_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)




def train():
    # env.start(gui=True)
    alg_cfg = DR.Config()
    alg = DR(alg_cfg, env, state_dim=args.state_dim, action_dim=args.action_dim, adv_action_dim=args.Advaction_dim, device=device).to(device)
    # 20241210 wq 跟踪当前episode的总奖励
    score = 0.0
    step = 0.0

    step_attack = 0.0
    step_attack_sn = 0.0

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
        score, step, step_attack, step_attack_sn,  v, v_epi, xa, ya, done = alg.interaction(args.T_horizon, score, step, step_attack, step_attack_sn, v, v_epi, args.speed_range, n_epi, model_dir_base, loss_path)
        print("训练episode:",n_epi,"奖励:",score)

        # 2024-12-12 wq cn碰撞的次数，sn成功的次数
        if done is True:
            cn += 1

        if args.env_name == 'TrafficEnv3-v1' or args.env_name == 'TrafficEnv3-v0':
            if xa < -50.0 and ya > 4.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv2-v0':
            if xa > 50.0 and ya > -5.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv4-v0':
            if ya < -50.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv8-v0':
            if ya == 10.0 and done is False:
                sn += 1
        # 2024-12-10 wq 打印统计信息
        if (n_epi+1) % args.print_interval == 0:
            print("# of episode :{}, avg score_v : {:.1f}".format(n_epi+1, score / args.print_interval))

            episode.append(n_epi+1)
            total_reward.append(score / args.print_interval)
            cn_epi.append(cn/args.print_interval)
            sn_epi.append(sn/args.print_interval)
            print("######cn & sn :", cn/args.print_interval, sn/args.print_interval)
            #
            if args.swanlab:
                swanlab.log(
                    {
                        "defender-cn": cn / args.print_interval,
                        "defender-sn": sn / args.print_interval,
                        "defender-mean_scores": score / args.print_interval,
                        "defender-mean_steps": step / args.print_interval,
                        "mean_attack_steps": step_attack / args.print_interval,
                    }
                )

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            step = 0.0
            step_attack = 0.0
            step_attack_sn = 0.0
            cn = 0.0
            sn = 0.0


    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi
    current_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    train_res = f"{train_result_dir}/train_data_{current_time_now}.csv"
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
