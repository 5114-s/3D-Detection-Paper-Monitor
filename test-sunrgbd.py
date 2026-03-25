import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 👇👇👇 强行把 PyTorch 的“家”搬到你指定的目录！
os.environ['TORCH_HOME'] = '/data/ZhaoX/OVM3D-Det-1/torch_hub_cache'
# 👆👆👆

# ==========================================
# 🌟 新增核心：SUN RGB-D 动态内参读取器
# ==========================================
def get_sunrgbd_intrinsic(calib_path):
    """精准提取 SUN RGB-D 当前照片的专属物理相机参数"""
    # 备用内参 (Kinect v2 的近似默认值，防止文件丢失时报错)
    fallback = [529.50, 529.50, 365.00, 265.00] 
    if not os.path.exists(calib_path): 
        print(f"⚠️ 找不到标定文件 {calib_path}，使用默认内参！")
        return fallback
    try:
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('K:'):
                    # 读取 3x3 矩阵展平后的 9 个数值
                    vals = [float(x) for x in line.strip().split()[1:]]
                    # 返回 [fx, fy, cx, cy]
                    return [vals[0], vals[4], vals[2], vals[5]]
    except Exception as e: 
        print(f"⚠️ 解析标定文件失败: {e}，使用默认内参！")
    return fallback

print("正在从本地加载 Metric3D v2 (ViT-Small)...")
model = torch.hub.load('Metric3D', 'metric3d_vit_small', source='local', pretrain=True)
model.cuda().eval()
print("模型加载成功！")

# ==========================================
# 🌟 路径配置 (请确保图片和对应的 txt 标定文件都在)
# ==========================================
image_path = 'test.jpg'     # 替换为你的 SUN RGB-D 图片
calib_path = 'test.txt'     # 替换为同名的 SUN RGB-D 标定文件

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"无法读取图片 {image_path}，请检查路径！")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# ==========================================
# 核心修复 1：图像尺寸对齐 (Padding)
# ==========================================
pad_h = (32 - h % 32) % 32
pad_w = (32 - w % 32) % 32
img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

img_tensor = torch.from_numpy(img_padded).permute(2,0,1).float()

# ==========================================
# 核心修复 2：ImageNet 标准化 (Normalization)
# ==========================================
mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std

img_tensor = img_tensor.unsqueeze(0).cuda()

# ==========================================
# 🌟 注入 SUN RGB-D 真实内参
# ==========================================
intrinsic = get_sunrgbd_intrinsic(calib_path)
print(f"当前图片的真实内参 [fx, fy, cx, cy]: {intrinsic}")

input_data = {
    'input': img_tensor,
    'intrinsic': intrinsic
}

with torch.no_grad():
    print("正在估算真实的物理深度...")
    pred_depth, confidence, output_dict = model.inference(input_data)

# ==========================================
# 核心修复 3：裁剪还原 (Crop)
# ==========================================
depth_map = pred_depth.squeeze().cpu().numpy()
depth_map = depth_map[:h, :w]  

np.save("output_depth_sunrgbd.npy", depth_map)

# ==========================================
# 🌟 可视化验证 (改为室内 10 米截断)
# ==========================================
# 将原来的 80 改为 10，这样室内房间的家具层次感才会清晰显示
depth_map_vis = np.clip(depth_map, 0, 10)

plt.figure(figsize=(12, 4))
plt.imshow(depth_map_vis, cmap='magma') 
plt.colorbar(label='Depth (meters) - Clipped at 10m')
plt.title('Metric3D v2 - SUN RGB-D Indoor Scene (10m Clip)')
plt.axis('off')
plt.savefig('depth_visual_sunrgbd.png', bbox_inches='tight', dpi=300)
print("完美！SUN RGB-D 室内深度图已保存为 depth_visual_sunrgbd.png")