import os
from PIL import Image

def process_grayscale_images(folder_path, threshold=128):
    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # 使用 PIL 打开灰度图像
        with Image.open(filepath).convert('L') as img:
            # 将像素值根据阈值转换为二值图像（0 或 255）
            binary_img = img.point(lambda p: 0 if p < threshold else 255, '1')
            # 保存二值图像，并覆盖原始图像
            binary_img.save(filepath)

if __name__ == "__main__":
    folder_path = "/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_labelled/masks_ours"  # 替换成你的文件夹路径

    # 将文件夹中的所有灰度图像处理为二值图像
    process_grayscale_images(folder_path)