import os
import cv2
import numpy as np
from PIL import Image

def visualize_mask_on_image(image_folder, mask_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历图像文件夹中的所有图像
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 加载图像
            # if filename != 'train_622-No006.jpg':
            #     continue
            # if filename not in ['6-No007.jpg', '9-No007.jpg', '60-1-No008.jpg', '74-1-No008.jpg', '77-No004.jpg', '214-1-No008.jpg', '1010-No007.jpg']:
            #     continue
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # 加载对应的掩码
            mask_filename = filename.split('.')[0] + '.png'  # 假设掩码文件名与图像文件名相同，但后缀为_mask.png
            mask_path = os.path.join(mask_folder, mask_filename)
            if not os.path.exists(mask_path):
                print(f"Mask file not found for image: {image_path}")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 1] = 255
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue

            # # 阈值化掩码
            # _, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # # 获取掩码的边缘
            # contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # 在图像上绘制边缘
            # result_image = image.copy()
            # cv2.drawContours(result_image, contours, -1, (0, 0, 255), 6)

            # # 计算前景区域与图像面积之比
            # area_ratio = sum(cv2.contourArea(cnt) for cnt in contours) / (image.shape[0] * image.shape[1])

            # # 在右下角添加文本
            # cv2.putText(result_image, f'Tooth Mask Ratio: {area_ratio:.2f}', (image.shape[1] - 1050, image.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 2.7, (0, 255, 0), 4, cv2.LINE_AA)
            # 保存可视化结果
            # output_path = os.path.join(output_folder, filename)
            # cv2.imwrite(output_path, result_image)

            ############## for only mask ##############
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            # Set transparent red for white pixels in the mask
            mask_rgba[mask == 255] = [255, 0, 0, 50]  # Transparent red (R,G,B,A)
            # Create a PIL image from the RGBA mask
            mask_pil = Image.fromarray(mask_rgba, 'RGBA')
            # Composite the RGBA mask onto the original PIL image
            image_pil = Image.open(image_path)
            result = Image.alpha_composite(image_pil.convert("RGBA"), mask_pil)
            
            file_name = image_path.split("/")[-1].split(".")[0]
            segmentation_image_path = os.path.join(output_folder, file_name + "_segmentation" + ".png")
            result.save(segmentation_image_path)


            # if area_ratio > 0.23:
            #     output_path = os.path.join(output_folder, filename)
            #     cv2.imwrite(output_path, result_image)

# 测试脚本
image_folder = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k/test/images"
mask_folder = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k/test/annotations"
output_folder = "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k-val-infer-compare/gt"

visualize_mask_on_image(image_folder, mask_folder, output_folder)
