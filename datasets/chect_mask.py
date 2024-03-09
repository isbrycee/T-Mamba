import os
from PIL import Image

def convert_to_binary_image(image_path):
    # 使用 PIL 打开图像
    with Image.open(image_path) as img:
        # 确保图像是三通道图像
        if img.mode.startswith('RGB'):
            # 将图像转换为灰度图
            gray_img = img.convert("L")
            # 将灰度图转换为二值图像（阈值设为128）
            binary_img = gray_img.point(lambda p: 0 if p < 128 else 255, '1')
            # 保存二值图像，并替换原始图像
            binary_img.save(image_path)

if __name__ == "__main__":
    folder_path = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k/train/images"  # 替换成你的文件夹路径

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        convert_to_binary_image(filepath)