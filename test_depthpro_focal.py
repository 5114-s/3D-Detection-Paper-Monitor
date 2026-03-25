"""
直接用DepthPro深度测试3D框
"""
import cv2
import numpy as np
import torch
import sys
import os

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import DepthProLoader

def test_depthpro_only():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 相机内参
    K = np.array([
        [529.5, 0, 365.0],
        [0, 529.5, 262.0],
        [0, 0, 1]
    ], dtype=np.float32)
    fx = K[0, 0]
    
    # 加载DepthPro
    print("加载DepthPro...")
    depthpro_loader = DepthProLoader()
    depthpro_loader.load_model()
    
    # 测试不同的focal_length
    test_focals = [
        (529.5, "K[0,0]=529.5"),
        (365.0, "K[0,2]*1.5=365"),
        (400.0, "K[0,2]*1.1=400"),
        (600.0, "K[0,0]*1.1=600"),
    ]
    
    for f_px, desc in test_focals:
        print(f"\n=== 测试 focal_length={f_px} ({desc}) ===")
        result = depthpro_loader.infer(img_rgb, focal_length_px=f_px)
        depth = result['depth']
        print(f"深度范围: [{depth.min():.2f}, {depth.max():.2f}], mean={depth.mean():.2f}")
        
        # 床区域
        bed = depth[50:200, 200:400]
        valid = bed[bed > 0.1]
        if len(valid) > 0:
            print(f"床区域: mean={valid.mean():.2f}m, range=[{valid.min():.2f}, {valid.max():.2f}]")
        
        # 保存可视化
        vis = np.clip(depth * 40, 0, 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        filename = f"depthpro_f{f_px:.0f}.jpg"
        cv2.imwrite(f'{PROJECT_ROOT}/{filename}', vis_color)
        print(f"保存: {PROJECT_ROOT}/{filename}")

if __name__ == '__main__':
    test_depthpro_only()
