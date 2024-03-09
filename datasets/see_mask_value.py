import os
import numpy as np
from PIL import Image

def print_unique_pixel_values(folder_path):
    unique_values = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            filepath = os.path.join(folder_path, filename)
            with Image.open(filepath) as img:
                # 转换为numpy数组
                img_array = np.array(img)
                # 找到数组中的唯一值并添加到集合中
                unique_values.update(np.unique(img_array))
    
    # 打印出所有图像中存在的不同的像素值
    print("所有图像中存在的不同的像素值:")
    for value in sorted(unique_values):
        print(value)

# 测试函数
folder_path = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k/train/annotations"
print_unique_pixel_values(folder_path)
