import os
from PIL import Image

def get_pixel_set(folder_path):
    pixel_set = set()

    # 遍历文件夹下的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            # 打开灰度图像文件
            img = Image.open(file_path).convert('L')
            # 获取图像的像素数据
            pixels = list(img.getdata())
            # 将像素值添加到集合中
            pixel_set.update(pixels)

    return pixel_set

if __name__ == "__main__":
    folder_path = '/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_labelled/masks_ours'
    pixel_set = get_pixel_set(folder_path)
    print("图像中的像素值集合：", pixel_set)
