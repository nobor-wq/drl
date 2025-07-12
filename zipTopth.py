from stable_baselines3 import PPO  # 根据你的算法选择
import torch

# 你的 zip 文件路径
zip_path = "logs/age_eval/TrafficEnv3-v1/PPO/lunar.zip"

# 加载模型
model = PPO.load(zip_path)  # 如果是自定义算法，换成 AdversarialPPO
# 获取模型参数
params = model.get_parameters()

# 保存为 .pth
torch.save(params, "logs/age_eval/TrafficEnv3-v1/PPO/lunar_baseline_1.pth")
print("模型参数已保存为 lunar_baseline.pth")
