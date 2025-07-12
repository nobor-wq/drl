import numpy as np

#seed 4 78 123 1234 12345
# 假设这五组成功率如下
success_rates = [0.02, 0.02, 0.02, 0.02, 0.02]

# 计算均值（平均成功率）
mean_rate = np.mean(success_rates)

# 计算样本标准差（这里使用 ddof=1 表示样本标准差）
std_rate = np.std(success_rates, ddof=1)

print("成功率: {:.2f} ± {:.2f}".format(mean_rate, std_rate))
