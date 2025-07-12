
from DARRLNetworkParams import PolicyNetwork
import Environment.environment
import torch

# env_name = "TrafficEnv3-v0"
env_model = "TrafficEnv3-v1"


model_name = 287
action_dim = 1
state_dim = 26

model = PolicyNetwork(state_dim, action_dim)

name = "./models/" + env_model + "/policy_v%d" % model_name
test_m = torch.load("{}.pkl".format(name))
print(type(test_m))  # 检查是否为 OrderedDict

state_dict = test_m.state_dict()

# model.load_state_dict(test_m)

print(model)  # 打印当前模型结构


torch.save(state_dict, "{}weights.pth".format(name))
