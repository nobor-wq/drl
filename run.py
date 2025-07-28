import gymnasium as  gym
import numpy as np
import random
import Environment.environment
import os
import pandas as pd
import datetime
import torch
import drl
from utils import get_config
from DARRLNetworkParams import ActorNet
import swanlab
from FGSM import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = get_config()
args = parser.parse_args()

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


device = torch.device(args.device)

if args.attacker:
    args.addition_msg = "attacker"
if args.lag:
    args.addition_msg = "lag"

if args.swanlab:
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

if args.attacker:
    prefix = os.path.join('models', args.env_name, args.algo, 'defender')
    if args.algo == "drl":
        filename = f'{args.model_name}.pth'
        model_path_drl = os.path.join(prefix, filename)
        if not os.path.isfile(model_path_drl):
            raise FileNotFoundError(f"找不到模型文件：{model_path_drl}")
        agent_model = ActorNet(state_dim=26, action_dim=1).to(device)
        agent_model.load_state_dict(torch.load(model_path_drl, map_location=device))
        agent_model.eval()


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
model_dir_base = os.getcwd() + '/models/' + args.env_name + f'/log-{current_time}/'
train_result_dir =model_dir_base + '/results/'

if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

if not os.path.exists(train_result_dir):
    os.makedirs(train_result_dir)


def train():

    model_t = drl.DRL(args.state_dim, args.action_dim, device)
    model_t.train()
    # 20241210 wq 跟踪当前episode的总奖励
    score = 0.0
    step = 0.0

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
        state, _ = env.reset()

        for t in range(args.T_horizon):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            step += 1
            if n_epi > 10:
               model_t.update()

            if args.attacker:
                with torch.no_grad():
                    _, _, action_adv = model_t.actor_adv(state_tensor)
                if args.algo == "drl":
                    state_adv = FGSM_vdarrl(action_adv, agent_model,
                                            state_tensor, algo=args.algo,
                                            epsilon=args.episode, attack_option=args.attack_option)
                    ego_action_attack, _, _ = agent_model(state_adv)
                    action = ego_action_attack
            else:
                with torch.no_grad():
                    _, _, action_before_attack = model_t.actor(state_tensor)
                    _, _, action_adv = model_t.actor_adv(state_tensor)

                state_adv = FGSM_vdarrl(action_adv, model_t.actor,
                            state_tensor, algo=args.algo,
                            epsilon=args.episode, attack_option=args.attack_option)

                with torch.no_grad():
                    _, _, ego_action_attack = model_t.actor(state_adv)
                    action = ego_action_attack

                swanlab.log({
                    "train/action_before_attack": round(action_before_attack.item(), 2),
                    "train/attack_action": round(action_adv.item(), 2),
                    "train/action_after_attack": round(action.item(), 2)
                })

            next_state, reward, done, _, info = env.step(action)

            model_t.replay_buffer.add(
                state=state_tensor,
                state_adv = state_adv,
                action=action,
                action_adv = action_adv,
                reward=reward,
                next_state=torch.tensor(next_state).to(device),
                cost=info['cost'],
                done=done
            )

            state = next_state
            score += reward

            v.append(state[24] * args.speed_range)
            v_epi.append(state[24] * args.speed_range)
            xa = info['x_position']
            ya = info['y_position']

            if done:
                break

        if args.attacker:
            if (n_epi + 1) % 100 == 0 :
                model_t.save_model(int(score), model_dir_base)
                print("#The attacker models are saved!#", n_epi + 1)
        else:
            if (n_epi + 1) % 100 == 0 and (n_epi + 1) >= 2000:
                model_t.save_model(int(score), model_dir_base)
                print("#The defender models are saved!#", n_epi + 1)

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
                    }
                )

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            step = 0.0

            cn = 0.0
            sn = 0.0
    env.close()


    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi
    current_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    train_res = f"{train_result_dir}/train_data_{current_time_now}.csv"
    df.to_csv(train_res, index=0)



if __name__ == "__main__":
    train()
