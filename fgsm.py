import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random


# 假设已经定义了ActionCostNetwork类及其min方法
# 假设action_cost_function已经定义好，使用ActionCostNetwork来计算行动成本

# FGSM更新加法扰动的函数
def update_additive_perturbation(delta_a0, epsilon_a, grad_cost):
    return delta_a0 + epsilon_a * np.sign(grad_cost)


# 贝叶斯优化过程
def bayesian_optimization(K, epsilon_m, epsilon_a, delta_m0, delta_a0, action_cost):
    # 初始化高斯过程回归模型
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # 存储数据的内存
    memory = []

    # 贝叶斯优化主循环
    for k in range(1, K + 1):
        # 步骤 2: 通过最大化 UCB 选择新的乘法扰动
        def ucb(delta_m):
            mu, sigma = gp.predict(np.array([[delta_m]]), return_std=True)
            return mu + 2 * sigma  # 选择最大 UCB 值

        # 最大化UCB，当前示例中直接选择初始扰动
        delta_m_k = delta_m0  # 初始乘法扰动

        # 步骤 3: 计算行动-成本函数
        state = torch.tensor(np.random.rand(5), dtype=torch.float32)  # 随机状态示例
        action = torch.tensor(np.random.rand(1), dtype=torch.float32)  # 随机动作示例
        cost = action_cost(state, action)  # 计算当前扰动下的成本

        # 步骤 4: 存储数据到内存
        memory.append((delta_m_k, cost))

        # 步骤 5: 更新 GP 代理模型
        X = np.array([x[0] for x in memory]).reshape(-1, 1)
        y = np.array([x[1] for x in memory])
        gp.fit(X, y)

    # 步骤 7: 使用FGSM更新加法扰动
    # 假设我们已计算出梯度
    grad_cost = np.random.rand()  # 这里模拟计算的梯度
    delta_a1 = update_additive_perturbation(delta_a0, epsilon_a, grad_cost)

    return delta_m_k, delta_a1


# 假设你已经定义了 action_cost 函数
# action_cost 函数使用你定义的 ActionCostNetwork 类来计算行动-成本
def action_cost(state, action):
    # 使用 ActionCostNetwork 来计算成本，假设 state 和 action 已经是张量
    return self.action_cost.min(state, action).item()  # 获取最小的 Q 值作为成本


# 初始化参数
epsilon_m = 0.1  # 乘法扰动的最大幅度
epsilon_a = 0.1  # 加法扰动的最大幅度
K = 50  # 总迭代次数

# 初始化乘法扰动和加法扰动
delta_m0 = np.random.normal(1.0, 0.1)  # 初始化乘法扰动
delta_a0 = np.random.normal(0.0, 0.1)  # 初始化加法扰动

# 调用贝叶斯优化函数
final_delta_m, updated_delta_a = bayesian_optimization(K, epsilon_m, epsilon_a, delta_m0, delta_a0, action_cost)

# 输出最终的扰动
print("Final multiplicative perturbation:", final_delta_m)
print("Updated additive perturbation:", updated_delta_a)
