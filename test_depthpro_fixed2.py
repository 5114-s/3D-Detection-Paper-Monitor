#!/usr/bin/env python3
"""测试 DepthPro，修复 fov 权重加载问题"""
import numpy as np
import cv2
import sys
import torch
import os

os.chdir('/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src')
sys.path.insert(0, '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src')

from depth_pro.depth_pro import DepthProConfig, create_model_and_transforms

print('加载 DepthPro...')
device = 'cuda:1'
ckpt = '/data/ZhaoX/OVM3D-Det-1/external/checkpoints/depth_pro.pt'

config = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    decoder_features=256,
    checkpoint_uri=None,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)

model, transform = create_model_and_transforms(
    device=device,
    precision=torch.float16,
    config=config
)
print('模型创建完成')

# 加载全部权重（不再跳过 fov）
state_dict = torch.load(ckpt, map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f'\n缺少的键 ({len(missing)}):')
    for k in missing:
        print(f'  {k}')

if unexpected:
    print(f'\n多余的键 ({len(unexpected)}):')
    for k in unexpected:
        print(f'  {k}')

if not missing and not unexpected:
    print('✓ 所有权重加载成功！')

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
    pred = model.infer(img_tensor)

depth_dp = pred['depth'].cpu().numpy()
f_px = pred.get('focallength_px', None)
if f_px is not None:
    print(f'\n模型估计的f_px: {f_px:.2f}')
    fov_deg = 2 * torch.atan(0.5 * w / f_px.float()).item()
    print(f'对应 FOV: {fov_deg:.1f}°')

if depth_dp.shape != (h, w):
    depth_dp = cv2.resize(depth_dp, (w, h))

print(f'DepthPro深度范围: {depth_dp.min():.3f}m - {depth_dp.max():.3f}m')

# 统计误差
valid = gt > 0
abs_err = np.abs(depth_dp[valid] - gt[valid])
print(f'\n===== 误差统计 =====')
print(f'Absolute Error: mean={abs_err.mean():.3f}m, median={np.median(abs_err):.3f}m')
print(f'Absolute Relative Error: {np.mean(np.abs(depth_dp[valid] - gt[valid]) / gt[valid]) * 100:.2f}%')
print(f'Sq. Relative Error: {np.mean(((depth_dp[valid] - gt[valid]) / gt[valid])**2) ** 0.5 * 100:.2f}%')
print(f'Threshold @ 1.25: {np.mean((np.maximum(depth_dp[valid]/gt[valid], gt[valid]/depth_dp[valid]) < 1.25)) * 100:.2f}%')
print(f'Threshold @ 1.5625: {np.mean((np.maximum(depth_dp[valid]/gt[valid], gt[valid]/depth_dp[valid]) < 1.5625)) * 100:.2f}%')

# 中心区域测试
cy, cx = h // 2, w // 2
region = min(h, w) // 4
dp_center = depth_dp[cy-region:cy+region, cx-region:cx+region]
gt_center = gt[cy-region:cy+region, cx-region:cx+region]
dp_med = np.median(dp_center[dp_center>0])
gt_med = np.median(gt_center[gt_center>0])
print(f'\n中心区域DepthPro: {dp_med:.3f}m')
print(f'中心区域GT: {gt_med:.3f}m')
print(f'中心区域误差: {abs(dp_med - gt_med) / gt_med * 100:.1f}%')
