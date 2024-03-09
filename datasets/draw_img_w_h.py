import os
import matplotlib.pyplot as plt
from PIL import Image

def plot_image_size_distribution(folder_path, save_path=None):
    # 获取文件夹中所有图片的宽度和高度
    image_sizes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(folder_path, filename)
            with Image.open(filepath) as img:
                width, height = img.size
                if height < 400:
                    continue
                image_sizes.append((width, height))

    # 绘制点状图
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Size Distribution')
    plt.grid(True)

    # 保存图像
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 测试函数
folder_path = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k/train/images"
save_path = ".image_size_distribution.png"
plot_image_size_distribution(folder_path, save_path)