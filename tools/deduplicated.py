import os
from PIL import Image
import shutil

def get_image_hash(image):
    return hash(image.tobytes())

def find_duplicate_images(folder_path):
    image_hashes = {}
    duplicate_images = []

    # 遍历文件夹下的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.jpg'):
            # 打开图像文件
            img = Image.open(file_path)
            # 计算图像哈希值
            img_hash = get_image_hash(img)
            # 如果哈希值已存在，说明图像重复
            if img_hash in image_hashes:
                duplicate_images.append(file_path)
            else:
                image_hashes[img_hash] = file_path
    print(f"重复的图像数量：{len(duplicate_images)}")
    return duplicate_images

if __name__ == "__main__":
    folder_path = '/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_unlabelled'
    duplicate_images = find_duplicate_images(folder_path)
    if duplicate_images:
        print("重复的图像：")
        for image in duplicate_images:
            shutil.move(image, '/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_unlabelled_same')
            # move mask
            # mask_path = image.replace('images', 'masks').replace('.jpg', '.png')
            # shutil.move(mask_path, '/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_labelled/masks_new')
    else:
        print("未发现重复的图像。")
