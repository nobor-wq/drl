# 111
# def attack(self, states):
#     memory = []
#     m_dim = 26
#     # 设置高斯分布的均值和标准差
#     mean_m, std_m = 1.0, 0.1  # 乘法扰动的均值和标准差
#     mean_a, std_a = 0.0, 0.1  # 加法扰动的均值和标准差
#     # 初始化乘法扰动 Δm 和加法扰动 Δa
#     delta_m0 = np.random.normal(mean_m, std_m, m_dim)  # 乘法扰动
#     delta_a0 = np.random.normal(mean_a, std_a)  # 加法扰动
#     # 限制 Δm 在 [0.5, 1.5] 范围内
#     delta_m0 = np.clip(delta_m0, 0.5, 1.5)
#     # 限制 Δa 在 [-0.5, 0.5] 范围内
#     delta_a0 = np.clip(delta_a0, -0.5, 0.5)
#     # print(type(states), shape(states), states)
#     delta_m_K = None
#     k = 0
#     for state in states:
#         print("k:", k)
#         if k == 0:
#             delta_m_k = delta_m0
#         else:
#             # X_sample = np.linspace(0.5, 1.5, 100).reshape(-1, 1)
#             # X_sample = np.random.normal(0, 1, (100, 26))
#             X_sample = np.random.normal(1.0, 0.2, (100, 26))  # 生成标准正态分布
#             X_sample = np.clip(X_sample, 0.5, 1.5)  # 限制范围在 [0.5, 1.5] 之间
#             ucb_values = self.ucb(X_sample)
#             delta_m_k = X_sample[np.argmax(ucb_values)]
#         delta_m_k = np.clip(delta_m_k, 0.5, 1.5)
#         delta_m_k = torch.tensor(delta_m_k, dtype=torch.double)
#         print("delta_m_k:", delta_m_k)
#         print("state:", state)
#         # 使用单个状态进行扰动计算
#         perturbed_state = delta_m_k * state + delta_a0
#         if k == len(states) - 1:
#             delta_m_K = delta_m_k
#         # 计算 perturbed_actions 和 cost
#         perturbed_state = perturbed_state.float()
#         print("perturbed_state:", perturbed_state)
#         _, _, perturbed_action = self.actor(perturbed_state)
#         cost = self.action_cost.compute_cost_for_single_state(perturbed_state, perturbed_action)
#         memory.append((delta_m_k, cost))
#         # 将 delta_m_k 转换为 1 行 26 列的数组，即形状为 [1, 26]
#         delta_m_k = delta_m_k.reshape(1, -1)
#         # 使用 detach() 来确保没有梯度计算
#         delta_m_k_numpy = delta_m_k.detach().numpy()  # 转换为 NumPy 数组
#         cost_numpy = cost.detach().numpy()  # 同样转为 NumPy 数组
#         # 训练高斯过程回归模型
#         self.gp.fit(delta_m_k_numpy, cost_numpy)
#         k += 1


# def attack(self, states):
#     memory = []
#     # 设置高斯分布的均值和标准差
#     mean_m, std_m = 1.0, 0.1  # 乘法扰动的均值和标准差
#     mean_a, std_a = 0.0, 0.1  # 加法扰动的均值和标准差
#     # 初始化乘法扰动 Δm 和加法扰动 Δa
#     delta_m0 = np.random.normal(mean_m, std_m)  # 乘法扰动
#     delta_a0 = np.random.normal(mean_a, std_a)  # 加法扰动
#     # 限制 Δm 在 [0.5, 1.5] 范围内
#     delta_m0 = np.clip(delta_m0, 0.5, 1.5)
#     # 限制 Δa 在 [-0.5, 0.5] 范围内
#     delta_a0 = np.clip(delta_a0, -0.5, 0.5)
#
#     for state in states:
#         delta_m_K = None
#         for k in range(1, 50):
#             print("k:", k)
#             i f(k == 1):
#                 delta_m_k = delta_m0
#             else:
#                 X_sample = np.linspace(0.5, 1.5, 100).reshape(-1, 1)
#                 print("X:shape", shape(X_sample))
#                 ucb_values = self.ucb(X_sample)
#                 delta_m_k = X_sample[np.argmax(ucb_values)]
#             delta_m_k = np.clip(delta_m_k ,0.5 ,1.5)
#             states_array = np.array(state)
#             perturbed_states = delta_m_k * states_array + delta_a0
#             perturbed_states = torch.tensor(perturbed_states, dtype=torch.float32)
#             if k == 49:
#                 delta_m_K = delta_m_k
#             _, _, perturbed_actions = self.actor(perturbed_states)
#             cost = self.action_cost.min(perturbed_states, perturbed_actions)
#             memory.append((delta_m_k, cost))
#
#             X = np.array([x[0] for x in memory]).reshape(-1, 1)
#             y = np.array([x[1].detach().numpy() for x in memory])
#             self.gp.fit(X, y)
#         states_array = np.array(state)
#         perturbed_states_l = delta_m_K * states_array + delta_a0
#         perturbed_states_l = torch.tensor(perturbed_states_l, dtype=torch.float32)
#         perturbed_actions_l = self.actor(perturbed_states_l)
#         cost = self.action_cost.min(perturbed_states_l, perturbed_actions_l)
#         delta_a_grad = np.sign(self.action_cost.grad(perturbed_states_l, perturbed_actions_l))  # 计算代价的梯度
#         delta_a1 = delta_a0 + self.epsilon_a * delta_a_grad  # 更新加法扰动
#         # 限制加法扰动范围
#         delta_a1 = np.clip(delta_a1, -0.5, 0.5)
#     return delta_m_K, delta_a1
#
#
#
#
#
#
#
#
#


#
#
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# import matplotlib.pyplot as plt
#
#
# # 初始化高斯过程模型
# def create_gp_model():
#     kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))  # 定义核函数
#     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
#     return gp
#
#
# # 上置信界（UCB）采集函数
# def ucb_acquisition_function(gp, X, kappa=1.96):
#     # 预测均值和标准差
#     mu, sigma = gp.predict(X, return_std=True)
#     mu = mu.flatten()  # 或者使用 mu.ravel()
#     sigma = sigma.flatten()
#     # mu.reshape(-1,1)
#     # sigma.reshape(-1,1)
#     # print("mu" , mu)
#     # print("sigma", sigma)
#     print(f"mu shape: {mu.shape}")
#     print(f"sigma shape: {sigma.shape}")
#
#     return mu + kappa * sigma  # UCB 公式
#
#
# # 模拟目标函数（对抗样本的损失函数）
# def objective_function(x):
#     return (x - 2) ** 2  # 这是一个简单的二次函数，实际使用时需要根据问题替换成目标函数
#
#
# # 训练高斯过程并选择最大UCB的点
# def optimize_gp(gp, X_train, y_train, bounds, n_iter=10):
#     for i in range(n_iter):
#         # 训练GP模型
#         gp.fit(X_train, y_train)
#
#         # 在范围内进行搜索，选择最大UCB值的点
#         X_sample = np.linspace(bounds[0], bounds[1], 10).reshape(-1, 1)
#         ucb_values = ucb_acquisition_function(gp, X_sample)
#         # print("X_sample",X_sample)
#         # print("ucb_values",ucb_values)
#         print(f"mu + 2 * sigma: {ucb_values}")
#         # 选择最大UCB值对应的点作为新的扰动
#         new_sample = X_sample[np.argmax(ucb_values)]
#
#
#         # 评估目标函数的值
#         new_y = objective_function(new_sample)
#
#         # 更新训练数据
#         X_train = np.vstack([X_train, new_sample])
#         y_train = np.append(y_train, new_y)
#
#         print(f"Iteration {i + 1}: New sample = {new_sample}, Objective value = {new_y}")
#
#     return X_train, y_train
#
#
# # 初始化训练数据
# X_train = np.array([[0.5]])  # 初始训练点
# y_train = objective_function(X_train)
#
# # 定义扰动的范围
# bounds = (0, 5)  # 假设扰动在[0, 5]之间
#
# # 创建GP模型
# gp = create_gp_model()
#
# # 优化过程
# X_opt, y_opt = optimize_gp(gp, X_train, y_train, bounds, n_iter=10)
#
# # 绘制结果
# plt.plot(X_opt, y_opt, 'r.', markersize=10, label="Samples")
# plt.xlabel("x")
# plt.ylabel("Objective Function Value")
# plt.title("Optimization using GP and UCB")
# plt.legend()
# plt.show()
