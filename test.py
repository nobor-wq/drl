import matplotlib.pyplot as plt  # 导入绘图模块
import gym
import os
import random
import numpy as np
import datetime
import Environment.environment
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
def load_model(model_n, env_n):
    name = "./models/" + env_model + "/darrl" + "/policy_v%d" % model_n
    test_m = torch.load("{}.pkl".format(name))
    return test_m

def load_model_pth( env_n, model_type):
    # 构造模型路径
    name = "./models/" + env_n + "/" + model_type + "/policy_ppo.pth"
    # 加载模型权重
    model = torch.load(name)
    # 将权重加载到模型中
    # model.load_state_dict(state_dict)
    return model



max_a = 7.6
print_interval = 10
seed = 4
num_episodes = 100
max_steps_per_episode = 30
env_name = "TrafficEnv3-v0"

env_model = "TrafficEnv3-v1"
model_name = 389
model_type = "ppo"

actor = load_model(model_name, env_name)
# actor_pth = load_model_pth(env_model,model_type)
test_env = gym.make(env_name)
isAttack = False




def adversarial_attack(states):
    from scipy.optimize import minimize
    from pykrige.ok import OrdinaryKriging
    # 2025-01-18 wq 加载动作损失模型
    cost_id = 389
    cost_name = "./models/" + env_model + "/darrl" + "/cost_v%d" % cost_id
    action_cost = torch.load("{}.pkl".format(cost_name))


    # 初始扰动
    a_init = np.random.normal(loc=0.0, scale=0.01)
    a_init = np.clip(a_init, -0.05, 0.05)
    m_train = np.linspace(0.95, 1.05, 3)
    # m_train = np.round(m_train, 4)

    m_train = torch.tensor(m_train, dtype=torch.float32)
    a_init = torch.tensor(a_init, dtype=torch.float32, requires_grad=True)

    X = np.array([]).reshape(0, 1)
    Y = np.array([]).reshape(0, 1)

    for i in range(m_train.shape[0]):  # 对每个 m_train 点进行计算
        m_point = m_train[i:i + 1]  # 获取当前的 m_train 点

        # 计算 perturbed_states
        perturbed_states = m_point * states + a_init

        with torch.no_grad():
            perturbed_actions, _, _  = actor(perturbed_states)
        # 计算 action-cost
        cost_train = action_cost.compute_cost_for_single_state(perturbed_states, perturbed_actions)
        cost_train_avg = torch.mean(cost_train)

        # 更新 X 和 Y
        X = np.vstack([X, m_point.detach().numpy()])  # 将当前 m_point 添加到 X 中
        Y = np.vstack([Y, cost_train_avg.detach().numpy().reshape(-1, 1)])  # 将当前 cost_train_avg 添加到 Y 中
    print("已知X:",X)
    print("已知Y:", Y)
    # breakpoint()
    ok = OrdinaryKriging(
        np.array(X),
        np.zeros_like(X),  # For 1D data, Y-coordinates are zeros spherical
        np.array(Y),
        variogram_model="gaussian",
    )

    for i in range(4):
        def ucb(x):
            mean, variance = ok.execute("points", x, [0])
            if variance[0] < 0:
                print(f"Warning: Negative variance encountered: {variance[0]}")
                variance[0] = 0  # 或者其他默认值

            return mean[0] + 2.0 * np.sqrt(variance[0])

        m_x = np.random.uniform(0.95, 1.05)  # 避免太接近边界
        m_next = minimize(ucb, x0=m_x, bounds=[(0.95, 1.05)]).x[0]
        m_next = torch.tensor(m_next, dtype=torch.float32)

        if np.any(np.isclose(X, m_next, atol=1e-4)):  # 设置一个适当的容差值
            continue  # 如果已存在，跳过这一轮，进入下一个循环

        p_states = m_next * states + a_init

        with torch.no_grad():
            p_actions,  _, _  = actor(p_states)
        cost_next = action_cost.compute_cost_for_single_state(p_states, p_actions)
        cost_next_avg = torch.mean(cost_next)

        m_next = m_next.detach().cpu().numpy().reshape(-1, 1)
        cost_next_avg = cost_next_avg.detach().cpu().numpy().reshape(-1)
        X = np.vstack([X, m_next])  # 将新的 m_train 添加到 X 中
        Y = np.vstack([Y, cost_next_avg])  # 将新的 cost_train_avg 添加到 Y 中

        ok = OrdinaryKriging(
            np.array(X),
            np.zeros_like(X),  # For 1D data, Y-coordinates are zeros spherical
            np.array(Y),
            variogram_model="gaussian",
        )
    print("X:",X)
    print("Y:",Y)

    max_index = np.argmax(Y)
    max_X = X[max_index]
    max_X = torch.tensor(max_X, dtype=torch.float32)
    # 计算损失函数 C_pi

    p_states = max_X * states + a_init
    with torch.no_grad():
        p_actions,  _, _  = actor(p_states)
    loss = action_cost.compute_cost_for_single_state(max_X * states + a_init, p_actions)
    loss = loss.mean()

    # 反向传播计算梯度
    action_cost.optimizer.zero_grad()
    loss.backward()

    # 提取对 a_init 的梯度，并计算符号函数
    gradient = a_init.grad.sign()

    # FGSM 更新
    a_last = a_init + 0.01 * gradient
    a_last = torch.clamp(a_last, min=-0.05, max=0.05).detach()

    print("m,a :", max_X, a_last)
    return max_X, a_last



# 假设有一个测试函数
def test_model(model, test_env, num_episodes=50, max_steps_per_episode=30):

    total_reward = 0.0  # 当前回合的总奖励
    rewards_per_episode = []  # 用于记录每轮的总奖励
    cn = 0.0
    cn_epi = []
    # 20241210 wq 成功次数
    sn = 0.0
    sn_epi = []

    for episode in range(num_episodes):  # 进行多轮测试
        state = test_env.reset()  # 初始化环境
        state = torch.tensor(state, dtype=torch.float32)  # 确保状态是张量
        #
        if isAttack:
            m, a = adversarial_attack(state)
            state = m * state + a
        # state = np.clip(state, 0, 1)


        done = False
        for step in range(max_steps_per_episode):  # 每轮最多进行30步
            pi, _, _ = model(state)  # 通过模型预测动作
            action = max_a * pi.item()  # 计算实际的动作
            reward, next_state, done, r_, cost, info = test_env.step(action)  # 执行动作并获取反馈
            # print("===============reward: ", reward)
            state = torch.tensor(next_state, dtype=torch.float32)  # 更新状态
            #
            if isAttack:
                m, a = adversarial_attack(state)
                state = m * state + a
            # state = np.clip(state, 0, 1)

            total_reward += reward  # 累加当前回合的奖励
            xa = info[0]
            ya = info[1]
            if done:  # 如果当前回合结束，跳出循环
                break
        if done:
            cn += 1
        if xa < -50.0 and ya > 4.0 and done is False:
            sn += 1

        if (episode + 1) % print_interval == 0:
            rewards_per_episode.append(total_reward/print_interval)  # 记录当前回合的总奖励
            cn_epi.append(cn/print_interval)
            sn_epi.append(sn/print_interval)
            print("######cn & sn rate:", cn/print_interval, sn/print_interval)
            cn = 0.0
            sn = 0.0
            total_reward = 0.0  # 当前回合的总奖励
        test_env.close()
    return rewards_per_episode, cn_epi, sn_epi



# test_env.seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# test_env.start(gui=False)


rewards, cn, sn = test_model(actor, test_env, num_episodes, max_steps_per_episode)

print(rewards)
print(cn)
print(sn)

plt.figure(figsize=(12, 6))

# 确保横坐标从 print_interval 开始，长度与 rewards 对齐
episodes_x = range(print_interval, num_episodes + 1, print_interval)

# 绘制总奖励曲线
plt.subplot(1, 2, 1)
plt.plot(episodes_x, rewards, marker='o')  # 横轴改为 episodes_x
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)

# 绘制碰撞率和成功率
plt.subplot(1, 2, 2)
plt.plot(episodes_x, cn, marker='o', label="Collision Rate", color='red')  # 横轴改为 episodes_x
plt.plot(episodes_x, sn, marker='o', label="Success Rate", color='blue')  # 横轴改为 episodes_x
plt.xlabel('Episodes (Every 10th)')
plt.ylabel('Rate')
plt.title('Collision Rate and Success Rate every 10 Episodes')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
if not os.path.exists('test'):
    os.makedirs('test')

model_folder = os.path.join('test', str(model_name))

# 创建文件夹
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

print(f"Folder created: {model_folder}")

if isAttack:
    plt.savefig(f"{model_folder}/Attacktest_{seed}_{current_date}.png")
else:
    plt.savefig(f"{model_folder}/test_{seed}_{current_date}.png")
plt.show()
