import torch
import torch.nn as nn

def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.01,
                 device="cpu", num_iterations=50):
    alpha = epsilon/num_iterations
    device = torch.device(device)


    low = torch.zeros_like(last_state, device=device)
    high = torch.zeros_like(last_state, device=device)

    # 距离和速度所在的维度索引
    dist_speed_dims = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25]
    # 角度所在的维度索引
    angle_dims = [1, 5, 9, 13, 17, 21]

    high[dist_speed_dims] = 1.0
    low[angle_dims] = -0.5
    high[angle_dims] = 0.5

    # 2) 构造 epsilon 球的局部上下界（保持在 original_state ± epsilon 内）
    orig = last_state.clone().detach().to(device)
    clamp_min = torch.clamp(orig - epsilon, min=low)  # 不低于全局 low
    clamp_max = torch.clamp(orig + epsilon, max=high)  # 不高于全局 high

    last_state = last_state.to(last_state.device)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    adv_action = adv_action.clone().detach().to(device)  # requires_grad=False
    loss = nn.MSELoss()

    for i in range(num_iterations):
        last_state.requires_grad = True

        outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)

        if outputs[0].dim() > 1:
            outputs = outputs[0].squeeze(0)
        else:
            outputs = outputs[0]
        victim_agent.policy.zero_grad()

        cost = -loss(outputs, adv_action).to(device)
        cost.backward()
        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()
    return last_state

def FGSM_vdarrl(adv_action, victim_agent, last_state, algo, epsilon=0.01,
                device="cpu", attack_option="a2", num_iterations=50):
    device = torch.device(device)
    alpha = epsilon/num_iterations

    if attack_option == "a1":
        clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state)).to(device)
        clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state)).to(device)
    elif attack_option == "a2":
        low = torch.zeros_like(last_state, device=device)
        high = torch.zeros_like(last_state, device=device)

        # 距离和速度所在的维度索引
        dist_speed_dims = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25]
        # 角度所在的维度索引
        angle_dims = [1, 5, 9, 13, 17, 21]

        high[dist_speed_dims] = 1.0
        low[angle_dims] = -0.5
        high[angle_dims] = 0.5

        # 2) 构造 epsilon 球的局部上下界（保持在 original_state ± epsilon 内）
        orig = last_state.clone().detach().to(device)
        clamp_min = torch.clamp(orig - epsilon, min=low)  # 不低于全局 low
        clamp_max = torch.clamp(orig + epsilon, max=high)  # 不高于全局 high

    last_state = last_state.clone().detach().to(device).requires_grad_(True)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()
    adv_action = adv_action.clone().detach().to(device)

    victim_agent = victim_agent.to(device)


    for i in range(num_iterations):
        last_state = last_state.detach().requires_grad_(True)
        last_state = last_state.to(device)

        if algo == "SAC_lag":
            _, _, outputs = victim_agent.sample(last_state)
            victim_agent.zero_grad()
        else:
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()

        cost = -loss(outputs, adv_action).to(device)
        cost.backward()
        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()

    return last_state


