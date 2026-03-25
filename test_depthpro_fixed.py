#!/usr/bin/env python3
"""直接测试DepthPro（不用MoGe）"""
import numpy as np
import cv2
import sys
import torch
import os

os.chdir('/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src')
sys.path.insert(0, '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src')

from depth_pro.depth_pro import DepthProConfig, DepthPro, create_model_and_transforms

print('加载 DepthPro...')
device = 'cuda:1'
ckpt = '/data/ZhaoX/OVM3D-Det-1/external/checkpoints/depth_pro.pt'

config = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    decoder_features=256,
    checkpoint_uri=None,  # 不自动加载
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)

model, transform = create_model_and_transforms(
    device=device, 
    precision=torch.float16,
    config=config
)
print('模型创建完成')

# 手动加载权重
state_dict = torch.load(ckpt, map_location=device)
# 过滤掉 fov 权重（形状不匹配）
filtered_dict = {}
for k, v in state_dict.items():
    if 'fov.' in k:
        print(f'跳过 fov 权重: {k}')
        continue
    filtered_dict[k] = v
    if any(x in k for x in ['encoder', 'decoder', 'head']):
        print(f'加载: {k}')

model.load_state_dict(filtered_dict, strict=False)
print('权重加载完成')

# SUNRGBD
SUNRGBD_IMAGE = "/data/ZhaoX/OVM3D-Det-1/datasets/SUNRGBD/kv1/NYUdata/NYU0001/image/NYU0001.jpg"
SUNRGBD_DEPTH = "/data/ZhaoX/OVM3D-Det-1/datasets/SUNRGBD/kv1/NYUdata/NYU0001/depth_bfx/NYU0001.png"
img = cv2.imread(SUNRGBD_IMAGE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
gt = cv2.imread(SUNRGBD_DEPTH, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
print(f'\n图像尺寸: {w}x{h}')
print(f'真实深度范围: {gt[gt>0].min():.2f}m - {gt[gt>0].max():.2f}m')

# DepthPro推理
img_tensor = transform(img_rgb)
with torch.no_grad():
    pred = model.infer(img_tensor)  # 不传f_px，让模型自己估计

depth_dp = pred['depth'].cpu().numpy()
f_px = pred.get('focallength_px', None)
if f_px is not None:
    print(f'模型估计的f_px: {f_px:.2f}')

if depth_dp.shape != (h, w):
    depth_dp = cv2.resize(depth_dp, (w, h))

print(f'DepthPro深度范围: {depth_dp.min():.3f}m - {depth_dp.max():.3f}m')

# 中心区域测试
cy, cx = h // 2, w // 2
region = min(h, w) // 4
dp_center = depth_dp[cy-region:cy+region, cx-region:cx+region]
gt_center = gt[cy-region:cy+region, cx-region:cx+region]
dp_med = np.median(dp_center[dp_center>0])
gt_med = np.median(gt_center[gt_center>0])
print(f'\n中心区域DepthPro: {dp_med:.3f}m')
print(f'中心区域GT: {gt_med:.3f}m')
print(f'误差: {abs(dp_med - gt_med) / gt_med * 100:.1f}%')
