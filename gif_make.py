import os
from PIL import Image

def gif_generate(image_folder, duration):
    # 定义要合成为 GIF 的图像路径
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]  # 仅获取 JPG 文件

    start = 1
    end = len(image_files)

    image_files = [f"{i}.jpg" for i in range(start, end + 1)]

    # 加载图像并存储在列表中
    images = [Image.open(os.path.join(image_folder, img)) for img in image_files]

    # 创建 GIF 动画
    output_gif_path = os.path.join(image_folder, 'simulation.gif')  # GIF 文件的输出路径

    # 使用 `save` 方法创建 GIF，设置 `save_all` 为 True 以保存所有帧
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],  # 从第二张图像开始追加
        duration=duration,  # 帧速率（每帧持续时间）
        loop=0  # 循环次数，0 表示无限循环
    )

    print(f"GIF saved to {output_gif_path}")