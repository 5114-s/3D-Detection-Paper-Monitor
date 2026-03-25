import os
# 👇👇👇 加入这行霸道的代码，强行把 PyTorch 的“家”搬到你指定的目录！
os.environ['TORCH_HOME'] = '/data/ZhaoX/OVM3D-Det-1/torch_hub_cache'
# 👆👆👆

import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

print("正在从本地加载 Metric3D v2 (ViT-Small)...")
model = torch.hub.load('Metric3D', 'metric3d_vit_small', source='local', pretrain=True)
model.cuda().eval()
print("模型加载成功！")

image_path = 'test-1.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# ==========================================
# 核心修复 1：图像尺寸对齐 (Padding)
# ==========================================
# 强行把长宽补齐到 32 的整数倍，避免 ViT 边缘崩溃
pad_h = (32 - h % 32) % 32
pad_w = (32 - w % 32) % 32
# 我们在右侧和下方进行填充 (保留左上角的光心坐标不变)
img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

# 转为 Tensor
img_tensor = torch.from_numpy(img_padded).permute(2,0,1).float()

# ==========================================
# 核心修复 2：ImageNet 标准化 (Normalization)
# ==========================================
# 满足大模型的“饮食习惯”，防止特征数值爆炸产生 NaN
mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std

img_tensor = img_tensor.unsqueeze(0).cuda()

# 注入 KITTI 左侧相机真实内参 [fx, fy, cx, cy]
intrinsic = [718.856, 718.856, 607.1928, 185.2157]
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
# 把刚才补齐的多余黑边切掉，恢复成原始的 376x1232
depth_map = depth_map[:h, :w]  

np.save("output_depth_flawless.npy", depth_map)

# 可视化验证
depth_map_vis = np.clip(depth_map, 0, 80)
plt.figure(figsize=(12, 4))
plt.imshow(depth_map_vis, cmap='magma') 
plt.colorbar(label='Depth (meters) - Clipped at 80m')
plt.title('Metric3D v2 - Flawless Scale & Artifacts Removed')
plt.axis('off')
plt.savefig('depth_visual_flawless.png', bbox_inches='tight', dpi=300)
print("完美可视化结果已保存为 depth_visual_flawless.png")