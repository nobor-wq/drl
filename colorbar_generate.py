import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.size": 20,       # 图例、坐标轴刻度、标题等字体大小
    "axes.titlesize": 20,  # 子图标题
    "axes.labelsize": 20,  # x/y 轴标签
    "xtick.labelsize": 20, # x 轴刻度
    "ytick.labelsize": 20, # y 轴刻度
    "legend.fontsize": 20  # 图例
})

def save_colorbar_as_image(cmap, vmin=0, vmax=1, decimal_places=1,
                           orientation='vertical', filename='colorbar.png'):
    """
    保存独立的colorbar为图片

    参数:
    cmap_name: 颜色映射名称
    vmin, vmax: 数值范围
    orientation: 'vertical' 或 'horizontal'
    filename: 保存的文件名
    """
    # 创建图形和轴
    if orientation == 'vertical':
        fig, ax = plt.subplots(figsize=(1, 20))
    else:
        fig, ax = plt.subplots(figsize=(20, 1))

    # 创建ScalarMappable对象
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 创建colorbar
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation)
    # cbar.set_label('Value')

    # 设置刻度格式
    if decimal_places == 0:
        # 整数格式
        format_str = '{:.0f}'
    else:
        # 小数格式
        format_str = '{:.' + str(decimal_places) + 'f}'
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format_str.format(tick) for tick in ticks])
    # 保存图片
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()

    print(f"Colorbar saved as {filename}")


# 使用示例
coolwarm = plt.cm.get_cmap('coolwarm', 256)
coolwarm_blues = ListedColormap(coolwarm(np.linspace(0, 0.5, 128)))
save_colorbar_as_image(cmap=coolwarm_blues, vmin=0, vmax=1, orientation='horizontal', filename='colorbar1.png')

# 水平方向的colorbar
coolwarm = plt.get_cmap("coolwarm").copy()
save_colorbar_as_image(cmap=coolwarm, vmin=-1, vmax=1, orientation='horizontal', filename='colorbar2.png')