# 对所有算法绘制0.05扰动下的动作偏移
import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
import os
from stable_baselines3 import SAC, PPO, TD3
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from DARRLNetworkParams import ActorNet, ActorNet_adv, SAC_lag_Net, FniNet
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法

# =============================================================================
# 2. 辅助函数
# =============================================================================
def get_action_from_my_policy(state, algo, agent, device):
    """
    Gets the deterministic action from a trained agent for a given state.

    Args:
        state (np.ndarray): The input state (observation).
        algo (str): The name of the algorithm (e.g., 'PPO', 'SAC', 'FNI').
        agent: The trained agent/policy object.
        device (torch.device): The device to run inference on (e.g., 'cuda:0' or 'cpu').

    Returns:
        np.ndarray: The resulting action.
    """
    # Ensure inference is done without calculating gradients
    with th.no_grad():
        # Convert numpy state to a torch tensor with a batch dimension
        obs_tensor = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)

        action = None
        if algo in ('FNI', 'DARRL', 'IGCARL'):
            # These models return mu, std, _
            mu, _, _ = agent(obs_tensor)
            action = mu
            action = action.cpu().numpy().flatten()
        elif algo == 'SAC_Lag':
            # This model returns mu, log_std
            _, _, action = trained_agent.sample(obs_tensor)
            action = action.cpu().numpy().flatten()
        elif algo in ('PPO', 'SAC', 'TD3'):
            action, _ = trained_agent.predict(state, deterministic=True)
            action = action.flatten()
        if action is None:
            raise NotImplementedError(f"Algorithm '{algo}' is not supported in this function.")

        # Move action to CPU and convert to numpy array
        return action


# =============================================================================
# 3. 主分析流程
# =============================================================================

# --- 分析参数 ---
STATE_DIM = 26
EPSILON = 0.05
NUM_SAMPLES = 1000  # 每个算法的采样次数，可根据需要调整
DEVICE = th.device("cuda:0" if th.cuda.is_available() else "cpu")
ENV_NAME = "TrafficEnv3-v1"
BASE_MODEL_PATH = "./models/"

# 要比较的算法列表
ALGOS_TO_COMPARE = ['PPO', 'SAC', 'TD3', 'SAC_Lag', 'FNI', 'DARRL', 'IGCARL']


# 获取一个初始状态作为分析中心
s_0_list_old = [
[0.7977517,0.1577962,0.93034124,0.75,0.21962911,0.27327174,0.77657455,0.5,0.7572472,0.3474857,0.6482767,0.25,
 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.6970827,0.16172528,0.92838246,0.75,0.08866517,0.30876663,0.7946154,0.5,0.6678262,0.348244,0.6508252,0.25,
 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.60052097,0.16535841,0.8345101,0.75,0.07479899,0.32035795,0.7935997,0.5,0.5789093,0.3494217,0.63977635,0.25,
#  0.05985176,-0.33977908,0.7769097,0.5,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.09816216,0.10622387,0.28014633,0.75,0.062797114,0.29066005,0.47732234,0.75,0.031482898,0.43358952,0.4282253,
 0.25,1.0,0.0,0.0,0.0,0.101141006,-0.29512632,0.7742487,0.5,1.0,0.0,0.0,0.0,1.0,0.98121536],
[0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276],
# [0.040507667,0.11337618,0.7817835,0.5,0.18617517,0.22367246,0.7745424,0.5,0.042812925,0.49141803,0.60328126,0.75,
#  0.086867556,-0.41184512,0.3975589,0.25,0.045703318,-0.24654646,0.34781283,0.25,0.12427995,-0.0599202,0.60824966,0.25,0.70840377,0.7909674],
[0.1042355,0.15856001,0.615245,0.25,0.1364679,0.23103333,0.4116078,0.75,0.047947843,0.36629468,0.7996352,0.5,
1.0,0.0,0.0,0.0,0.12228283,-0.29213965,0.77727324,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
# 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276]
]
# s_0 = [0.040507667,0.11337618,0.7817835,0.5,0.18617517,0.22367246,0.7745424,0.5,0.042812925,0.49141803,0.60328126,0.75, 0.086867556,-0.41184512,0.3975589,0.25,0.045703318,-0.24654646,0.34781283,0.25,0.12427995,-0.0599202,0.60824966,0.25,0.70840377,0.7909674]
# s_0 = [0.1042355,0.15856001,0.615245,0.25,0.1364679,0.23103333,0.4116078,0.75,0.047947843,0.36629468,0.7996352,0.5,
# 1.0,0.0,0.0,0.0,0.12228283,-0.29213965,0.77727324,0.5,1.0,0.0,0.0,0.0,1.0,0.0]
# s_0 = [0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
# 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276]
# s_0_list = [
# [0.08654119,0.016529402,0.34176302,0.75,0.091991864,0.2390672,0.77353585,0.5,0.0236178,0.4379837,0.54115963,0.75,
#  0.06298592,-0.39364055,0.2655102,0.25,0.049562525,-0.22966859,0.7925001,0.5,0.060031988,-0.112641454,0.5304923,0.25,0.8689022,0.8533303],
# [0.09049825,0.010381286,0.34176302,0.75,0.08932654,0.23116758,0.77353585,0.5,0.018577801,0.44857523,0.54115963,0.75,
#  0.061774652,-0.3806577,0.2655102,0.25,0.053285517,-0.21829413,0.7925001,0.5,0.06524592,-0.111627474,0.5304923,0.25,0.93672794,0.84997636],
# [0.09909227,0.0,0.46403053,0.75,0.27803245,0.20093006,0.79522604,0.5,1.0,0.0,0.0,0.0,0.10142445,-0.4215388,
#  0.5704252,0.25,0.04866946,-0.22357178,0.37614575,0.25,0.039262325,0.5,0.66966283,0.75,1.0,0.75],
# [0.594429,0.16439703,0.8345101,0.75,0.06846599,0.32740197,0.7935997,0.5,0.5731807,0.35057157,0.63977635,
#  0.25,1.0,0.0,0.0,0.0,0.06594423,-0.33063725,0.7769097,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.89304006,0.15401646,0.9280561,0.75,0.3449353,0.2647862,0.7827482,0.5,0.8407728,0.3476244,0.6384082,
#  0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0]
# [0.040507667,0.11337618,0.7817835,0.5,0.18617517,0.22367246,0.7745424,0.5,0.042812925,0.49141803,0.60328126,0.75,
#  0.086867556,-0.41184512,0.3975589,0.25,0.045703318,-0.24654646,0.34781283,0.25,0.12427995,-0.0599202,0.60824966,0.25,0.70840377,0.7909674],
# ]

s_0_list = [
[1.0,0.0,0.0,0.0,0.84218675,0.25604877,0.7884088,0.5,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
[0.9943977,0.15164976,0.9323587,0.75,0.4782257,0.26065767,0.79861337,0.5,0.9296712,0.34693593,0.6366861,0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.69107354,0.16086648,0.92838246,0.75,0.08210146,0.3137213,0.7946154,0.5,0.66206485,0.34922925,0.6508252,0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.61510015,0.1252045,0.9286797,0.75,0.10761138,0.29805413,0.78600657,0.5,0.48318028,0.35189053,0.6658786,0.25,1.0,0.0,0.0,0.0,0.079305775,-0.3161035,0.7745433,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
[0.22282372,0.1300715,0.6376857,0.25,0.16283907,0.2397277,0.52179575,0.25,0.035458326,0.42912328,0.78937477,0.5,1.0,0.0,0.0,0.0,0.12820108,-0.29015085,0.7996682,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
[0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276],
[0.09385265,0.0,0.5289496,0.75,0.23851956,0.16701941,0.77459085,0.5,1.0,0.0,0.0,0.0,0.1593269,-0.45129517,0.6583022,0.25,0.05048467,-0.30014032,0.52363515,0.25,0.059703574,0.5,0.731638,0.75,0.33593234,0.75],
[0.08303008,0.0,0.28938824,0.75,0.3728997,0.1692704,0.7774285,0.5,1.0,0.0,0.0,0.0,0.0952156,-0.41590863,0.66253346,0.25,0.45049047,-0.18414032,0.7865022,0.5,0.06247909,-0.13943785,0.51415986,0.25,0.52589774,0.75],
[0.1599409,0.0,0.48187014,0.75,0.8106836,0.1785545,0.77347684,0.5,1.0,0.0,0.0,0.0,0.07129907,-0.38245523,0.6534352,0.25,0.8248173,-0.17986563,0.7846309,0.5,0.10176676,-0.078173734,0.65420383,0.25,0.8847849,0.75],
[0.24818082,0.0,0.67077035,0.75,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.14250404,-0.4453222,0.652128,0.25,1.0,0.0,0.0,0.0,0.058242872,-0.15416919,0.64918244,0.25,1.0,0.75],
]

all_results = []


# --- 主循环：遍历所有算法 ---
for s_idx, s_0_raw in enumerate(tqdm(s_0_list)):
    s_0 = np.array(s_0_raw, dtype=np.float32)
    for algo_name in ALGOS_TO_COMPARE:
        print(f"\n--- Processing algorithm: {algo_name} ---")

        # --- 加载模型 (逻辑来自您的脚本) ---
        trained_agent = None
        try:
            if algo_name == 'PPO':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender', 'lunar_baseline')
                trained_agent = PPO.load(model_path, device=DEVICE)
            elif algo_name == 'SAC':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender', 'lunar_baseline')
                trained_agent = SAC.load(model_path, device=DEVICE)
            elif algo_name == 'TD3':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender', 'lunar_baseline')
                trained_agent = TD3.load(model_path, device=DEVICE)
            elif algo_name == 'SAC_Lag':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', 'SAC_lag', 'defender', 'lunar_baseline.pt')
                trained_agent = SAC_lag_Net(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))
            elif algo_name == 'DARRL':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender', 'policy2000_actor.pth')
                trained_agent = FniNet(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))
            elif algo_name == "FNI":
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender', 'policy_v411.pth')
                trained_agent = FniNet(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))
            elif algo_name == "IGCARL":
                model_path_drl = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', 'drl', '0.05', 'm4', '5', 'defender/defender.pth')
                trained_agent = ActorNet(state_dim=26, action_dim=1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path_drl, map_location=DEVICE))

            if trained_agent is None:
                print(f"Warning: Could not load model for {algo_name}. Skipping.")
                continue

            if hasattr(trained_agent, 'eval'):
                trained_agent.eval()

            # --- 数据生成 ---
            # 计算基准动作
            action_base = get_action_from_my_policy(s_0, algo_name, trained_agent, DEVICE)
            action_base_scalar = action_base[0]

            # 在半径为 epsilon 的球面上采样
            for _ in tqdm(range(NUM_SAMPLES), desc=f"Sampling for {algo_name}"):
                direction = np.random.randn(STATE_DIM)
                direction = direction / np.linalg.norm(direction)

                s_perturbed = s_0 + EPSILON * direction

                # 计算动作偏移
                action_perturbed = get_action_from_my_policy(s_perturbed, algo_name, trained_agent, DEVICE)
                offset = action_perturbed[0] - action_base_scalar
                # offset = action_perturbed[0]
                all_results.append({'algo': algo_name, 'offset': offset})

        except Exception as e:
            print(f"Error processing {algo_name}: {e}")

# --- 4. 汇总数据并绘图 ---
ALGOS_TO_COMPARE_NEW = ['PPO', 'SAC', 'TD3', 'SAC_Lag', 'FNI', 'DARRL', 'IGCARL (Ours)']
if not all_results:
    print("No data was generated. Cannot create plot.")
else:
    df_results = pd.DataFrame(all_results)
    df_results['algo'] = df_results['algo'].replace({'IGCARL': 'IGCARL (Ours)'})
    offset_summary = df_results.groupby('algo')['offset'].agg(
        min_offset='min',
        max_offset='max'
    )

    # 2. 计算最大偏移范围 (max - min)
    offset_summary['max_offset_range'] = offset_summary['max_offset'] - offset_summary['min_offset']

    # 3. 对结果进行排序，使其与图表顺序一致
    offset_summary = offset_summary.loc[ALGOS_TO_COMPARE_NEW]

    # 4. 打印计算结果
    print("--- Algorithm Offset Analysis ---")
    print(offset_summary)
    print("---------------------------------")

    sns.set(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df_results,
        x='algo',
        y='offset',
        order=ALGOS_TO_COMPARE,  # 保持指定顺序
        ax=ax,
        hue='algo',  # Assign 'algo' to hue
        legend=False # Disable the legend
    )

    # 美化图表
    ax.axhline(0, color='r', linestyle='--', label='No Change')
    # ax.set_title(f'Action Offset Distribution of Algorithms under Perturbation (ε={EPSILON})', fontsize=16)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Action Offset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig('action_box_plot.pdf', bbox_inches='tight')

    print("正在生成小提琴图...")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=df_results,
        x='algo',
        y='offset',
        order=ALGOS_TO_COMPARE_NEW,  # 保持与之前一致的顺序
        ax=ax,
        palette='muted',
        hue='algo',  # Assign 'algo' to hue
        legend=False # Disable the legend,

    )
    yticks = np.linspace(-0.5, 0.5, 5)
    # 设置刻度
    # 设置上下限
    plt.ylim(-0.5, 0.5)

    # 设置自动生成的刻度
    plt.yticks(yticks)
    # --- 核心修改：添加数值标注 ---
    # # 获取当前图表的Y轴范围，以便将文本放在顶部
    # y_min, y_max = ax.get_ylim()
    # # 将文本放置在图表顶部 95% 的位置
    # text_y_position = y_max * 0.95
    #
    # stability_summary = df_results.groupby('algo')['offset'].mean().to_dict()
    # for i, algo_name in enumerate(ALGOS_TO_COMPARE):
    #     # 从字典中获取该算法的平均绝对偏移量
    #     mean_abs_offset = stability_summary.get(algo_name, 0)
    #
    #     # 准备要显示的文本
    #     annotation_text = f"{mean_abs_offset:.4f}"
    #
    #     # 使用 ax.text() 在图表上添加文本
    #     ax.text(
    #         x=i,  # 文本的x坐标 (0, 1, 2, ...)
    #         y=text_y_position,  # 文本的y坐标
    #         s=annotation_text,  # 要显示的字符串
    #         ha='center',  # 水平居中对齐
    #         va='top',  # 垂直顶部对齐
    #         fontsize=11,
    #         fontweight='bold',
    #         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)  # 添加一个背景框
    #     )

    # --- 美化图表 ---
    # ax.axhline(0, color='r', linestyle='--', label='No Change')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Action Offset from Clean Observation')
    ax.tick_params(axis='x', rotation=45)
    # ax.legend(loc='lower right')
    # 1. 获取所有的X轴标签对象列表
    xtick_labels = ax.get_xticklabels()

    # 2. 检查列表是否为空，以防万一
    if xtick_labels:
        # 3. 获取最后一个标签对象
        last_label = xtick_labels[-1]

        # 4. 设置其属性
        last_label.set_color('red')
        last_label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('action_violin_plot.pdf', bbox_inches='tight')


    # =============================================================================
    # 图表二: 平均绝对偏移量条形图 (Bar Chart of Mean Absolute Offset)
    # =============================================================================
    # print("\n正在生成平均绝对偏移量条形图...")
    #
    # # 1. 计算每个算法的平均绝对偏移量
    # df_results['abs_offset'] = df_results['offset'].abs()
    # stability_summary = df_results.groupby('algo')['abs_offset'].mean().reset_index()
    #
    # # 2. 按指定顺序排序
    # stability_summary = stability_summary.set_index('algo').loc[ALGOS_TO_COMPARE].reset_index()


    # # 3. 绘制条形图
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # barplot = sns.barplot(
    #     data=stability_summary,
    #     x='algo',
    #     y='abs_offset',
    #     ax=ax,
    #     palette='coolwarm_r',
    #     hue='algo',  # Assign 'algo' to hue
    #     legend=False # Disable the legend
    # )
    #
    # # 美化图表
    # ax.set_xlabel('Algorithm')
    # ax.set_ylabel('Mean Absolute Offset (Lower is More Stable)')
    # ax.tick_params(axis='x', rotation=45)
    #
    # # 在每个条形柱上显示数值
    # for p in barplot.patches:
    #     barplot.annotate(format(p.get_height(), '.4f'),
    #                      (p.get_x() + p.get_width() / 2., p.get_height()),
    #                      ha = 'center', va = 'center',
    #                      xytext = (0, 9),
    #                      textcoords = 'offset points')
    #
    # plt.tight_layout()
    # plt.savefig('action_stability_barchart.png', dpi=300)