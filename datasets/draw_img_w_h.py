import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

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

    # 创建一个二维直方图
    hist, xedges, yedges = np.histogram2d([size[1] for size in image_sizes], [size[0] for size in image_sizes], bins=(10, 10))

    # 找到直方图中所有点的最大数量
    max_count = np.amax(hist)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.xlabel('Width')
    plt.ylabel('Height')

    # 遍历直方图中的所有点
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if hist[i, j] > 0:
                # 计算点的大小，根据频率调整
                point_size = 200 + 1000 * hist[i, j] / max_count
                # 计算点的颜色，根据大小调整
                color = hist[i, j] / max_count
                # 绘制点
                plt.scatter(yedges[j], xedges[i], color=plt.cm.bwr(color), marker='o', s=point_size, alpha=0.8, edgecolors='black')
                # 在每个点上显示数量
                if int(hist[i, j]) > 100:
                    plt.text(yedges[j], xedges[i], f"{int(hist[i, j])}", ha='center', va='center', color='black', fontsize=8)

#     plt.title('Image Size Distribution')
    plt.grid(True)

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
#        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

# 测试函数
folder_path = "/Users/haojing08/Documents/paper-writing/T-Mamba/Tooth-2D-X-Ray-6k/train/images"
save_path = "image_size_distribution.pdf"
plot_image_size_distribution(folder_path, save_path)
