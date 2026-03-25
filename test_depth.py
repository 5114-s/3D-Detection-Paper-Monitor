#!/usr/bin/env python3
"""快速测试深度输出"""
import cv2
import numpy as np
import torch
import sys
import os

# 添加路径
sys.path.insert(0, "/data/ZhaoX/OVM3D-Det-1")
from cubercnn.generate_label.process_indoor import HybridDepthExtractor
from types import SimpleNamespace

# 模拟一个内参矩阵 (可以替换成你实际用的)
K = np.array([
    [577.87, 0, 319.5],
    [0, 577.87, 239.5],
    [0, 0, 1]
], dtype=np.float32)

# 使用 SUNRGBD 真实图像
test_img_path = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg"
if not os.path.exists(test_img_path):
    print(f"图像不存在: {test_img_path}")
    sys.exit(1)
print(f"使用测试图像: {test_img_path}")

# 初始化深度提取器
print("初始化 HybridDepthExtractor...")
depth_extractor = HybridDepthExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

# 测试深度提取
print(f"\n提取深度: {test_img_path}")
depth = depth_extractor.get_depth(test_img_path, K)

# 打印深度统计
print(f"\n=== 深度统计 ===")
print(f"深度图形状: {depth.shape}")
print(f"深度范围: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"深度均值: {depth.mean():.4f}")
print(f"深度中位数: {np.median(depth):.4f}")
print(f"深度标准差: {depth.std():.4f}")

# 保存深度图可视化
depth_vis = (depth / depth.max() * 255).astype(np.uint8)
cv2.imwrite("/data/ZhaoX/OVM3D-Det-1/test_depth.png", depth_vis)
print(f"\n深度图已保存到: /data/ZhaoX/OVM3D-Det-1/test_depth.png")

# 如果有 scale/shift 信息也打印出来
print("\n=== 额外信息 ===")
raw_img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
h, w = raw_img.shape[:2]
pad_img = cv2.copyMakeBorder(raw_img, 0, max(0, 896-h), 0, max(0, 896-w), cv2.BORDER_CONSTANT)[:896, :896]
img_t_sam = (torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(depth_extractor.device) - torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1).to(depth_extractor.device)) / torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1).to(depth_extractor.device)
img_t_dino = (torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(depth_extractor.device)/255.0 - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(depth_extractor.device)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(depth_extractor.device)
gt_intrinsic = torch.eye(4).float().unsqueeze(0).to(depth_extractor.device)
gt_intrinsic[0, :3, :3] = torch.tensor(K).float()
out = depth_extractor.image_encoder({"images": img_t_sam, "image_for_dino": img_t_dino, "vit_pad_size": torch.tensor([[h//16, w//16]], device=depth_extractor.device), "gt_intrinsic": gt_intrinsic})

print(f"scale: {out['scale'].cpu().numpy()}")
print(f"shift: {out['shift'].cpu().numpy()}")
print(f"pred_K:\n{out['pred_K'].cpu().numpy()}")
print(f"\n真实 K:\n{K}")
