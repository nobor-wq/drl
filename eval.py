import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from rlConfig import get_config
import Environment.environment
import os
import torch as th
import argparse
from FGSM import *
from DARRLNetworkParams import PolicyNetwork, ActorNet, PolicyNetContinuous_adv, FniNet
from SAC_Agent_Continuous import SAC_Lag, GaussianPolicy
import drac
from gif_make import gif_generate
import random
from PIL import Image
# get parameters from config.py
parser = get_config()
args = parser.parse_args()
# args_eval, unknown = parser_eval.parse_known_args()

# 设置设备

# if args.use_cuda and th.cuda.is_available():
#     device = th.device(f"cuda:{args.cuda_number}")
# else:
device = th.device("cpu")

random.seed(args.seed)  # 设置 Python 随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
torch.manual_seed(args.seed)  # 设置 CPU 随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
torch.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化





# 创建环境
if args.env_name == "TrafficEnv8-v0":
    env = gym.make(args.env_name, use_gui=args.use_gui, render_mode=args.render_mode)
else:
    env = gym.make(args.env_name, attack=False, use_gui=args.use_gui, render_mode=args.render_mode)
env = TimeLimit(env, max_episode_steps=args.T_horizon)
env = Monitor(env)
env.unwrapped.start()

if args.attack:
    if args.best_model:
        adv_model_path = os.path.join(args.adv_path, args.adv_algo, args.env_name, args.algo, args.addition_msg, 'best_model')
    else:
        # advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg, 'lunar')
        adv_model_path = os.path.join(args.adv_path, args.env_name, args.adv_algo, 'attacker', args.adv_algo_name)

    # if args.adv_algo == 'SAC':
    #     model = SAC.load(adv_model_path, device=device)
    # elif args.adv_algo == 'PPO':
    #     model = PPO.load(adv_model_path, device=device)
    # elif args.adv_algo == 'drl':
        # model = PolicyNetContinuous_adv(args.state_dim, args.action_dim)
    model = th.load(adv_model_path, map_location=device)
    model.eval()
    # else:
    #     model = PPO.load(adv_model_path, device=device)


# 加载训练好的自动驾驶模型
if args.best_model:
    model_path = os.path.join(args.age_path, args.env_name, args.algo, 'best_model/best_model')
else:
    model_path = os.path.join(args.age_path, args.env_name, args.algo, 'defender', 'lunar_baseline')
    print('**********************************************************')

if args.algo == 'PPO':
    print('*******************algo is PPO*******************')
    trained_agent = PPO.load(model_path, device=device)
elif args.algo == 'SAC':
    print('*******************algo is SAC*******************')
    trained_agent = SAC.load(model_path, device=device)
elif args.algo == 'SAC_lag':
    print('*******************algo is SAC_lag*******************')
    trained_agent = GaussianPolicy(26, 1)
    state_dict = torch.load(model_path+".pt", map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
elif args.algo == 'TD3':
    print('*******************algo is TD3*******************')
    trained_agent = TD3.load(model_path, device=device)
elif args.algo == 'drl':
    print('*******************algo is drl*******************')
    model_path_drl = os.path.join(args.age_path, args.env_name, args.algo, 'defender', args.algo_name)
    # trained_agent = PolicyNetwork(args.state_dim, args.action_dim)
    trained_agent = th.load(model_path_drl, map_location=device)
    trained_agent.eval()
elif args.algo == 'DARRL':
    trained_agent = ActorNet(26,1)
    model_path_drl = os.path.join(args.age_path, args.env_name, args.algo, 'defender', 'policy2000_actor.pth')
    state_dict = torch.load(model_path_drl, map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
elif args.algo == "FNI":
    trained_agent = FniNet(26, 1)
    score = f"policy_v{411}"
    path = os.path.join('models', args.env_name, args.algo, 'defender', score) + '.pth'
    state_dict = torch.load(path, map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()

# 进行验证
rewards = []
steps = []

maxSpeed = 15.0
ct = 0
sn = 0
sat = 0
speed_list = []
attack_count_list = []
mean_attack_reward_list = []
mean_delta_list = []
for episode in range(args.train_step):
    obs, info = env.reset(options="seed")
    img = env.render()
    speed = 0
    episode_reward = 0
    episode_steps = 0
    if args.attack:
        save_dir = f'./render/{args.env_name}/{args.algo}/WithAttack/{episode}'
    else:
        save_dir = f'./render/{args.env_name}/{args.algo}/WithoutAttack/{episode}'
    # 创建目录（如果不存在的话）
    os.makedirs(save_dir, exist_ok=True)
    for _ in range(args.T_horizon):
        obs_tensor = obs_as_tensor(obs, device)
        # print("扰动前的状态：", obs_tensor)
        if args.attack:
            speed_list.append(obs[-2])
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor)
                actions = actions.detach().cpu().numpy()
            elif args.algo == 'IL':
                actions = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            elif args.algo == 'drl':
                actions, _, _ = trained_agent(obs_tensor.cpu())
            elif args.algo == 'SAC_lag':
                _, _, actions = trained_agent.sample(obs_tensor)
            else:
                actions, _ = trained_agent.predict(obs, deterministic=True)

            if isinstance(actions, np.ndarray):
                # 假设 action 是一个标量数组，比如 array([0])，用 item() 取标量
                actions_tensor = th.tensor(actions, device=obs_tensor.device)
            elif isinstance(actions, torch.Tensor):
                actions_tensor = actions

            # actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            adv_actions, _,  _= model(obs_tensor.cpu())
            print(episode_steps, 'attack', 'Victim action is', actions.item(), 'adv actions is', adv_actions.item())

            if args.attack_method == 'fgsm':
                if args.algo == 'drl' or args.algo == 'DARRL' or args.algo == 'SAC_lag' or args.algo == 'FNI':
                    # print("states: ", obs_tensor)
                    adv_state = FGSM_vdarrl(adv_actions, victim_agent=trained_agent, last_state=obs_tensor, algo=args.algo)
                    # print("adv_states: ", adv_state)
                else:
                    adv_state = FGSM_v2(adv_actions, victim_agent=trained_agent, last_state=obs_tensor)

            if args.algo in ('FNI', 'DARRL'):
                adv_action_fromState, _, _ = trained_agent(adv_state)
                action = adv_action_fromState.detach().cpu().numpy()
            elif args.algo == 'drl':
                adv_action_fromState, _, _ = trained_agent(adv_state)
                action = adv_action_fromState.detach().cpu().numpy()
            elif args.algo == 'SAC_lag':
                _, _, adv_action_fromState = trained_agent.sample(adv_state)
                action = adv_action_fromState.detach().cpu().numpy()
            else:
                adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                action = adv_action_fromState
            print(episode_steps, 'attack', '{} action is'.format(args.attack_method), adv_action_fromState.item())

            if args.diff_between:
                obs_np = obs_tensor.detach().cpu().numpy()
                adv_np = adv_state.cpu().numpy()
                delta = adv_np - obs_np
                mean_delta = np.mean(np.abs(delta))
                mean_delta_list.append(mean_delta)


            obs, reward, done, T, info = env.step(action)
        else:
            # print("state: ", obs_tensor)
            speed_list.append(obs[-2])
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            elif args.algo == 'IL':
                actions = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            elif args.algo == 'drl':
                actions, _, _ = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            elif args.algo == 'SAC_lag':
                _, _, actions = trained_agent.sample(obs_tensor)
            else:
                actions, _ = trained_agent.predict(obs, deterministic=True)
            # print("state is ",obs_tensor)
            print('action is ', actions)
            obs, reward, done, T, info = env.step(actions)
        episode_reward += reward
        episode_steps += 1
        if done:
            ct += 1
            break
        if args.use_gui:
            img = env.render()
            img = Image.fromarray(img)
            img.save(f'{save_dir}/{episode_steps}.jpg')

    if args.to_gif:
        gif_generate(save_dir, args.duration)
    env.close()


    xa = info['x_position']
    ya = info['y_position']
    if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v1' or args.env_name == 'TrafficEnv6-v0':
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
    rewards.append(episode_reward)
    steps.append(episode_steps)

if args.diff_between:
    save_dir_dif = f'./render/{args.env_name}/{args.algo}/WithAttack/'
    os.makedirs(save_dir_dif, exist_ok=True)
    # 开始画图
    plt.figure(figsize=(8, 4))
    plt.plot(mean_delta_list, marker='o', linewidth=2)
    plt.title('Average Perturbation over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Delta')
    plt.grid(True)

    # 保存到 save_dir
    save_path = os.path.join(save_dir_dif, 'mean_delta_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 计算平均奖励和步数
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

# 计算碰撞率
cr = ct / args.train_step * 100
sr = sn / args.train_step * 100

# 计算平均速度
mean_speed = np.mean(speed_list)
std_speed = np.std(speed_list)

# 计算平均攻击次数
# attack_list = [x for x in attack_count_list if x != 0]
# mean_attack_times = np.mean(attack_list)
# std_attack_times = np.std(attack_list)

# 计算单次攻击的收益
# mean_attack_reward = np.mean(mean_attack_reward_list)
# std_attack_reward = np.std(mean_attack_reward_list)

# print('attack lists ', attack_count_list, 'attack times ', len(attack_list))
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}")
print(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}")
# print(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}")
print(f"Collision rate: {cr:.2f}")
print(f"Success rate: {sr:.2f}")
# print(f"Success attack rate: {asr:.2f}")
# print(f"Reward per attack: {mean_attack_reward:.2f} +/- {std_attack_reward:.2f}")

# 定义日志文件路径
log_file = "eval_attack_log.txt"

# 将参数和结果写入日志文件
with open(log_file, 'a') as f:  # 使用 'a' 模式以追加方式写入文件
    # 写入参数
    f.write("Parameters:\n")
    for arg in vars(args):  # 遍历 args 中的所有参数
        f.write(f"{arg}: {getattr(args, arg)}\n")

    # 写入结果
    f.write("\nResults:\n")
    f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    f.write(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}\n")
    f.write(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}\n")
    # f.write(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}\n")
    f.write(f"Collision rate: {cr:.2f}\n")
    f.write(f"Success rate: {sr:.2f}\n")
    # f.write(f"Success attack rate: {asr:.2f}\n")
    # f.write(f"Reward per attack: {mean_attack_reward:.2f} +/- {std_attack_reward:.2f}\n")
    f.write("-" * 50 + "\n")

