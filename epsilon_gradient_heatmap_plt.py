import pickle
import numpy as np  # 导入 numpy 以便正确处理文件中的数组
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'averaged_z_matrices.pkl'
# --- 从文件读取字典 ---
print(f"--- Loading dictionary from {file_path} ---")

try:
    with open(file_path, 'rb') as f:
        averaged_z_matrices = pickle.load(f)

    print("--- Dictionary loaded successfully! ---")

    # --- 验证并使用加载的数据 (示例) ---
    print("\nLoaded dictionary keys:")
    print(list(averaged_z_matrices.keys()))

    # 访问其中一个 NumPy 数组
    # 假设 ('PPO', 0.05) 是一个有效的键
    key_to_check = ('PPO', 0.05)
    if key_to_check in averaged_z_matrices:
        a = averaged_z_matrices[key_to_check]
        print(f"\nData for key {key_to_check}:")
        print(f"  - Type: {type(a)}")
        print(f"  - Shape: {a.shape}")
        print(f"  - A snippet of the matrix:\n{a[:2, :2]}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# --- 分析参数 ---
STATE_DIM = 26
RESOLUTION = 100

# 要比较的算法列表
EPSILONS = [0.1]
ALGOS_TO_COMPARE = ['PPO', 'SAC', 'TD3', 'SAC_Lag', 'FNI', 'DARRL', 'IGCARL']

# 设置共享的行和列标题
fig, axes = plt.subplots(len(EPSILONS), len(ALGOS_TO_COMPARE), figsize=(26,5))
mappable = None

print("\n--- Generating final plot ---")
for i, epsilon in enumerate(EPSILONS):
    for j, algo_name in enumerate(ALGOS_TO_COMPARE):
        ax = axes[j]
        z_to_plot = averaged_z_matrices.get((algo_name, epsilon), np.zeros((RESOLUTION, RESOLUTION)))

        sns.heatmap(
            np.transpose(z_to_plot), ax=ax, cmap='coolwarm', center=0, vmin=-0.3, vmax=0.3, cbar=False
        )
        ax.set_aspect('equal', adjustable='box')

        # ==================== 核心修改：在角落標註 (min, max) ====================
        # 1. 計算最大最小值
        min_val = np.min(z_to_plot)
        max_val = np.max(z_to_plot)

        # 2. 格式化文字
        print(f"{algo_name}, {min_val}, {max_val}")
        # --- 设置每个子图的刻度 ---

        tick_positions = np.linspace(0, RESOLUTION, num=3)

        tick_labels_float = np.linspace(-epsilon, epsilon, num=3)

        ax.set_xticks(tick_positions, labels=[f'{val:.1f}' for val in tick_labels_float], rotation='horizontal')

        ax.set_yticks(tick_positions, labels=[f'{val:.1f}' for val in tick_labels_float], rotation='horizontal')
        if (i, j) == (0, 0):  # 只需获取一次mappable
            mappable = ax.collections[0]

# --- 调整和美化整个图表 ---
for i, epsilon in enumerate(EPSILONS):
    axes[0].set_ylabel(f'Action Offset from Clean Observation', fontsize=16)
for j, algo_name in enumerate(ALGOS_TO_COMPARE):
    if algo_name == 'IGCARL':
        label_text = 'IGCARL (Ours)'
        # 将标签文本设为 'IGCARL (Ours)' 并加粗
        axes[j].set_xlabel(label_text, labelpad=15, fontweight='bold',fontsize=16, color='red')
    else:
        # 其他算法标签保持不变
        axes[j].set_xlabel(algo_name, labelpad=15, fontsize=16)

# 隐藏内部子图多余的标签
for j in range(len(ALGOS_TO_COMPARE)):
    if j > 0:
        axes[j].set_ylabel('')

cbar_ax = fig.add_axes([0.3, 0.9, 0.4, 0.02])
if mappable:
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
cbar_ax.xaxis.set_ticks_position('top')
cbar_ax.xaxis.set_label_position('top')

fig.subplots_adjust(
    left=0.05, right=0.98, top=0.92, bottom=0.06, wspace=0.17, hspace=0.17
)
plt.savefig('heatmap_grid_comparison_averaged.png', dpi=300, bbox_inches='tight')