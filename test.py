""" import numpy as np

# 加载 .npz 文件
npz_file = np.load('./2D_Data/test.npz')

# 查看文件中包含的数组名称
print(npz_file.files)

# 如果你想查看某个数组的具体内容，可以用以下方法：
for array_name in npz_file.files:
    array = npz_file[array_name]
    print(f"Array Name: {array_name}")
    print(f"Shape: {array.shape}")
    print(f"Type: {array.dtype}") """
import numpy as np
from PIL import Image  # 使用PIL库加载图像

# 1. 加载三通道（RGB）图像
img_path = 'test.png'  # 图像路径
img_rgb = Image.open(img_path).convert('RGB')  # 加载图像并转换为RGB模式

# 2. 将PIL图像转换为NumPy数组
img_array = np.array(img_rgb)

# 3. 保存为 .npz 文件
npz_save_path = 'test.npz'
np.savez_compressed(npz_save_path, imgs=img_array)

print(f"Image saved to {npz_save_path}")


