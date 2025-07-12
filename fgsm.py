import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from utils import get_config

parser = get_config()
args = parser.parse_args()

def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.03,
                 num_iterations=50):
    alpha = epsilon/num_iterations
    device = args.device

    state0 = last_state.clone().detach()

    clamp_min = torch.where(
        state0 >= 0,
        torch.clamp(state0 - epsilon, min=0.0),
        state0 - epsilon
    ).to(device)

    clamp_max = torch.where(
        state0 >= 0,
        state0 + epsilon,
        torch.clamp(state0 + epsilon, max=0.0)
    ).to(device)

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
def FGSM_vdarrl(adv_action, victim_agent, last_state, algo, methond="test", epsilon=0.03,
                  num_iterations=50):
    device = args.device
    alpha = epsilon / num_iterations

    # clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state)).to(device)
    # clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state)).to(device)

    state0 = last_state.clone().detach()

    clamp_min = torch.where(
        state0 >= 0,
        torch.clamp(state0 - epsilon, min=0.0),
        state0 - epsilon
    ).to(device)

    clamp_max = torch.where(
        state0 >= 0,
        state0 + epsilon,
        torch.clamp(state0 + epsilon, max=0.0)
    ).to(device)

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
            if methond == "get":
                _, _, outputs = victim_agent(last_state)
            else:
                outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()

        cost = -loss(outputs, adv_action).to(device)
        cost.backward()
        # if outputs < 0:
        #     print("last_state.grad.sign(): ", last_state.grad.sign())
        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()

    return last_state


