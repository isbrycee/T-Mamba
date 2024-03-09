import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 读取.nii.gz文件
file_path = 'images/1001250407_20190923.nii.gz'
img = nib.load(file_path)

# 获取图像数据
img_data = img.get_fdata()

# 可以查看图像的形状
print("Image shape:", img_data.shape)

# 可以查看图像的头信息
print("Image header:", img.header)

# 显示图像的一个切片
plt.imshow(img_data[:, :, img_data.shape[2]//2], cmap='gray')
plt.title('Slice at the middle of the volume')
plt.savefig('images/1001250407_20190923.png')

# 显示图像的所有切片成 gif
# Function to create a GIF from 3D volume
def create_gif(data, output_path='output.gif'):
    fig, ax = plt.subplots()
    ax.axis('off')

    # Create and save each slice as an image
    images = []
    for i in range(data.shape[2]):
        img = ax.imshow(data[:, :, i], cmap='gray', animated=True)
        images.append([img])

    # Create GIF
    ani = FuncAnimation(fig, lambda x: x, frames=images, interval=1, blit=True)
    ani.save(output_path, writer='pillow', fps=10)

create_gif(img_data, 'images/1001250407_20190923.gif')