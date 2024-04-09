import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Times New Roman'
def round_to_nearest_0025(number):
    # 将数字除以0.025并四舍五入取整
    rounded_number = round(number / 0.025) * 0.025
    return rounded_number

def calculate_pixel_ratio(folder_path):
    ratios = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(folder_path, filename)
            with Image.open(filepath) as img:
                # 转换为numpy数组
                img_array = np.array(img)
                # 计算图像总面积
                total_pixels = img_array.size
                # 统计像素值为255的数量
                count_255 = np.sum(img_array == 1)
                # 计算像素值为255的像素数占图像总面积的比例
                ratio = round_to_nearest_0025(round(count_255 / total_pixels, 2))
                ratios.append(ratio)
    return ratios

# 输入两个图像文件夹路径
folder_path1 = "/Users/haojing08/Documents/paper-writing/T-Mamba/Tooth-2D-X-Ray-6k/train/annotations"
folder_path2 = "/Users/haojing08/Documents/paper-writing/T-Mamba/Tooth-2D-X-Ray-6k/test/annotations"

# 分别计算两个文件夹中像素比例
ratios1 = calculate_pixel_ratio(folder_path1)
ratios2 = calculate_pixel_ratio(folder_path2)

# 统计每个比例出现的次数并计算比例的比例
# ratio_counts1 = {ratio: ratios1.count(ratio) / len(ratios1) for ratio in set(ratios1)}
# ratio_counts2 = {ratio: ratios2.count(ratio) / len(ratios2) for ratio in set(ratios2)}
ratio_counts1 = {ratio: ratios1.count(ratio) for ratio in set(ratios1)}
ratio_counts2 = {ratio: ratios2.count(ratio) for ratio in set(ratios2)}

# 将字典按照键排序
sorted_ratios1 = sorted(ratio_counts1.items())
sorted_ratios2 = sorted(ratio_counts2.items())

# 提取比例和对应的比例比例
x1 = [ratio[0] for ratio in sorted_ratios1]
y1 = [ratio[1] for ratio in sorted_ratios1]
x2 = [ratio[0] for ratio in sorted_ratios2]
y2 = [ratio[1] for ratio in sorted_ratios2]

# f_cubic_1 = interp1d(x1, y1, kind='quadratic')
# f_cubic_2 = interp1d(x2, y2, kind='quadratic')
# 绘制图表并保存
plt.figure(figsize=(10, 6))
# plt.scatter(x1, y1, label="Training set", marker='o', color='blue')
# plt.scatter(x2, y2, label="Test set", marker='o', color='red')
# 绘制拟合曲线

l1,=plt.plot(x1, y1, color='red', label='Training set',  linestyle='-')
l2,=plt.plot(x2, y2, color='blue', label='Test set', linestyle='--') # , linestyle='--

# plt.plot(x2, func(x2, *popt2), color='red', linestyle='--')
plt.xlabel('Ratio of Tooth area to image area', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.ylabel('# Samples', fontdict={'family' : 'Times New Roman', 'size' : 12})
# plt.title('Mask Ratio Distribution')
plt.legend(handles=[l1, l2], loc='upper right', prop={'family' : 'Times New Roman', 'size' : 12})
plt.grid(True)

# 添加不同范围的底色
# plt.axvspan(0, 0.1, color='blue', alpha=0.1)
# plt.axvspan(0.1, 0.2, color='yellow', alpha=0.1)
# plt.axvspan(0.2, 0.4, color='red', alpha=0.1)

# plt.savefig("TED3_mask_ratio_all_new.png")
plt.savefig('TED3_mask_ratio_all_new.pdf', bbox_inches='tight', format='pdf')
