import os
import drl
import torch
import random
import swanlab
import datetime
import numpy as np
import pandas as pd
from FGSM import *
import gymnasium as gym
from utils import get_config
import Environment.environment
from stable_baselines3 import PPO, SAC, TD3
from DARRLNetworkParams import ActorNet, SAC_lag_Net, FniNet


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
    run_name = f"{args.env_name}-{args.algo}-{args.seed}-{args.method}-{args.epsilon}-{args.addition_msg}"
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
    elif args.algo == "PPO" :
        model_path_ppo = os.path.join(prefix, "lunar_baseline")
        agent_model = PPO.load(model_path_ppo, device=device)
    elif args.algo == "SAC":
        model_path_ppo = os.path.join(prefix, "lunar_baseline")
        agent_model = SAC.load(model_path_ppo, device=device)
    elif args.algo == "TD3":
        model_path_ppo = os.path.join(prefix, "lunar_baseline")
        agent_model = TD3.load(model_path_ppo, device=device)
    elif args.algo == "SAC_lag":
        agent_model = SAC_lag_Net(26, 1)
        model_path_slag = os.path.join(prefix, "lunar_baseline")
        state_dict_slag = torch.load(model_path_slag + ".pt", map_location=device)
        agent_model.load_state_dict(state_dict_slag)
        agent_model.eval()
    elif args.algo == "FNI":
        agent_model = FniNet(26, 1)
        score = f"policy_v{411}.pth"
        model_path_fni = os.path.join(prefix, score)
        state_dict = torch.load(model_path_fni, map_location=device)
        agent_model.load_state_dict(state_dict)
        agent_model.eval()
    elif args.algo == "DARRL":
        agent_model = FniNet(26, 1)
        score = f"policy2000_actor.pth"
        model_path_fni = os.path.join(prefix, score)
        state_dict = torch.load(model_path_fni, map_location=device)
        agent_model.load_state_dict(state_dict)
        agent_model.eval()




attack_prefix = os.path.join('models', args.env_name, args.algo, str(args.epsilon), str(args.seed), 'attacker')
# current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
model_dir_base = os.path.join(os.getcwd(), attack_prefix)
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

    attacker_flag = False


    for n_epi in range(args.train_step):
        state, _ = env.reset()

        for t in range(args.T_horizon):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            step += 1
            if n_epi > 10:
                if args.frequency:
                    if (n_epi + 1) % 200 < 100:
                        attacker_flag = False
                    else:
                        attacker_flag = True
                model_t.update(attacker_flag)

            if args.attacker:
                with torch.no_grad():
                    _, _, action_adv  = model_t.actor_adv(state_tensor)
                if args.algo == "drl":
                    state_adv = FGSM_vdarrl(action_adv, agent_model,
                                            state_tensor, algo=args.algo,
                                            epsilon=args.epsilon, device=args.device,
                                            attack_option=args.attack_option)
                    with torch.no_grad():
                        ego_action_attack, _, _ = agent_model(state_adv)
                        action = ego_action_attack
                elif args.algo == "PPO" or args.algo == "SAC":
                    state_adv = FGSM_v2(action_adv, agent_model, state_tensor,
                                        epsilon=args.epsilon, device=args.device)
                    with torch.no_grad():
                        ego_action_attack, _ = agent_model.predict(state_adv.cpu(), deterministic=True)
                        action = torch.from_numpy(ego_action_attack).to(state_tensor.device)  # 或者 .cuda()

                elif args.algo == "SAC_lag":
                    state_adv = FGSM_vdarrl(action_adv, agent_model, state_tensor,algo=args.algo,
                                        epsilon=args.epsilon, device=args.device)
                    with torch.no_grad():
                        _, _, ego_action_attack = agent_model.sample(state_adv)
                        action = ego_action_attack  # 或者 .cuda()
                elif args.algo == "FNI":
                    state_adv = FGSM_vdarrl(action_adv, agent_model, state_tensor,algo=args.algo,
                                        epsilon=args.epsilon, device=args.device)
                    with torch.no_grad():
                        ego_action_attack, _, _ = agent_model(state_adv)
                        action = ego_action_attack  # 或者 .cuda()



            else:
                if args.get:
                    with torch.no_grad():
                        _, _, action = model_t.actor(state_tensor)
                else:
                    with torch.no_grad():
                        # _, _, action_before_attack = model_t.actor(state_tensor)
                        _, _, action_adv  = model_t.actor_adv(state_tensor)

                    state_adv = FGSM_vdarrl(action_adv, model_t.actor,
                                state_tensor, algo=args.algo,
                                epsilon=args.epsilon, device=args.device,
                                attack_option=args.attack_option)

                    with torch.no_grad():
                        _, _, ego_action_attack = model_t.actor(state_adv)
                        action = ego_action_attack

            next_state, reward, done, _, info = env.step(action)

            if args.replay:
                if done:
                    model_t.replay_buffer_done.add(
                        state=state_tensor,
                        state_adv=state_adv,
                        action=action,
                        action_adv=action_adv,
                        reward=reward ,
                        next_state=torch.tensor(next_state).to(device),
                        done=done,
                        cost=info['cost']

                    )

            if args.get:
                model_t.replay_buffer.add(
                    state=state_tensor,
                    state_adv=state_tensor,
                    action=action,
                    action_adv=action,
                    reward=reward,
                    next_state=torch.tensor(next_state).to(device),
                    done=done,
                    cost=info['cost']
                )
            else:
                model_t.replay_buffer.add(
                    state=state_tensor.detach(),
                    state_adv=state_adv.detach(),
                    action=action.detach(),
                    action_adv = action_adv.detach(),
                    reward=reward,
                    next_state=torch.tensor(next_state).to(device).detach(),
                    done=done,
                    cost=info['cost']
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
            if (n_epi + 1) % 100 == 0 and (n_epi + 1) >= int(args.train_step * 0.7) :
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
