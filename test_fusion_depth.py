#!/usr/bin/env python
# Copyright (c) Teacher-Student Distillation Pipeline
"""
测试 MoGe + Depth Pro 融合深度模型

使用方法:
    python test_fusion_depth.py --image <image_path> --method ransac_align

融合方法:
    1. ransac_align: RANSAC线性对齐(推荐,LabelAny3D原生方法)
    2. weighted_adaptive: 自适应加权融合
    3. frequency_split: 频域分离融合
    4. weighted: 简单加权融合
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "MoGe"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "ml-depth-pro", "src"))


def test_moge_only(image_path, device="cuda"):
    """单独测试 MoGe 模型"""
    print("\n" + "="*60)
    print("测试 MoGe 模型")
    print("="*60)
    
    from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader
    
    loader = MoGeLoader()
    model = loader.load_model()
    
    if model is None:
        print("MoGe 模型加载失败")
        return None
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推理
    result = loader.infer(image_rgb)
    
    if result is None:
        print("MoGe 推理失败")
        return None
    
    print(f"MoGe 输出:")
    print(f"  - points shape: {result['points'].shape}")
    print(f"  - depth shape: {result['depth'].shape}, range: [{result['depth'].min():.3f}, {result['depth'].max():.3f}]")
    print(f"  - mask shape: {result['mask'].shape}, valid: {result['mask'].sum()}")
    print(f"  - intrinsics:\n{result['intrinsics']}")
    
    return result


def test_depthpro_only(image_path, device="cuda"):
    """单独测试 Depth Pro 模型"""
    print("\n" + "="*60)
    print("测试 Depth Pro 模型")
    print("="*60)
    
    from detany3d_frontend.depth_predictor.moge_depthpro_fusion import DepthProLoader
    
    loader = DepthProLoader()
    model, transform = loader.load_model()
    
    if model is None:
        print("Depth Pro 模型加载失败")
        return None
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推理
    result = loader.infer(image_rgb)
    
    if result is None:
        print("Depth Pro 推理失败")
        return None
    
    print(f"Depth Pro 输出:")
    print(f"  - depth shape: {result['depth'].shape}, range: [{result['depth'].min():.3f}, {result['depth'].max():.3f}] m")
    print(f"  - focallength_px: {result['focallength_px']:.1f} px")
    
    return result


def test_fusion(image_path, method='ransac_align', device="cuda"):
    """测试融合模型"""
    print("\n" + "="*60)
    print(f"测试融合模型 (方法: {method})")
    print("="*60)
    
    from detany3d_frontend.depth_predictor.moge_depthpro_fusion import (
        MoGeLoader, DepthProLoader, create_moge_depthpro_fusion, align_depth_ransac
    )
    
    # 加载模型
    moge_loader = MoGeLoader()
    moge_model = moge_loader.load_model()
    
    depthpro_loader = DepthProLoader()
    depthpro_model, _ = depthpro_loader.load_model()
    
    if moge_model is None or depthpro_model is None:
        print("模型加载失败")
        return None
    
    # 创建融合模型
    fusion_model = create_moge_depthpro_fusion(
        moge_model=moge_loader,
        depthpro_model=depthpro_loader,
        fusion_method=method,
    )
    fusion_model = fusion_model.to(device)
    fusion_model.eval()
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MoGe 推理
    moge_result = moge_loader.infer(image_rgb)
    if moge_result is None:
        print("MoGe 推理失败")
        return None
    
    # Depth Pro 推理
    pro_result = depthpro_loader.infer(image_rgb)
    if pro_result is None:
        print("Depth Pro 推理失败")
        return None
    
    # 手动 RANSAC 对齐测试
    print("\n手动 RANSAC 对齐测试:")
    aligned_depth, diag = align_depth_ransac(
        moge_result['depth'],
        pro_result['depth'],
        mask=moge_result['mask'],
    )
    print(f"  - 对齐: {diag['status']} scale={diag['scale']:.4f} 内点率={diag['inlier_ratio']:.1%}")
    print(f"  - 对齐后深度 range: [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}] m")
    
    # 融合模型测试
    print("\n融合模型输出:")
    print(f"  - 方法: {method}")
    
    return {
        'moge': moge_result,
        'depthpro': pro_result,
        'aligned': aligned_depth,
    }


def visualize_depth(depth, title="Depth"):
    """可视化深度图"""
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imshow(title, depth_vis)
    return depth_vis


def main():
    parser = argparse.ArgumentParser(description="测试 MoGe + Depth Pro 融合深度模型")
    parser.add_argument("--image", type=str, 
                        default="/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg",
                        help="测试图像路径")
    parser.add_argument("--method", type=str, default="ransac_align",
                        choices=['ransac_align', 'weighted_adaptive', 'frequency_split', 'weighted'],
                        help="融合方法")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--test_moge", action="store_true", help="仅测试 MoGe")
    parser.add_argument("--test_depthpro", action="store_true", help="仅测试 Depth Pro")
    parser.add_argument("--save", type=str, default=None, help="保存可视化结果的路径")
    args = parser.parse_args()
    
    # 检查图像是否存在
    if not os.path.exists(args.image):
        print(f"图像不存在: {args.image}")
        # 尝试找其他测试图像
        test_images = [
            "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000001.jpg",
            "/data/ZhaoX/OVM3D-Det-1/data/SUNRGBD/000034/image/000034.jpg",
        ]
        for img in test_images:
            if os.path.exists(img):
                args.image = img
                print(f"使用备用图像: {args.image}")
                break
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试 MoGe
    if args.test_moge:
        test_moge_only(args.image, device)
    
    # 测试 Depth Pro
    if args.test_depthpro:
        test_depthpro_only(args.image, device)
    
    # 测试融合
    if not args.test_moge and not args.test_depthpro:
        result = test_fusion(args.image, args.method, device)
        
        if result is not None and args.save:
            # 保存可视化结果
            moge_vis = visualize_depth(result['moge']['depth'], "MoGe Depth")
            pro_vis = visualize_depth(result['depthpro']['depth'], "Depth Pro")
            aligned_vis = visualize_depth(result['aligned'], "Aligned Depth")
            
            base_name = os.path.splitext(args.save)[0]
            cv2.imwrite(f"{base_name}_moge.jpg", moge_vis)
            cv2.imwrite(f"{base_name}_depthpro.jpg", pro_vis)
            cv2.imwrite(f"{base_name}_aligned.jpg", aligned_vis)
            print(f"可视化结果已保存到: {base_name}_*.jpg")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
