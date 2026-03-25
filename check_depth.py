"""
检查MoGe和DepthPro的输出范围，确定融合策略
"""
import cv2
import numpy as np
import torch

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'

def check_depth_ranges():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {h}x{w}")
    
    # 相机内参
    K = np.array([
        [529.5, 0, 365.0],
        [0, 529.5, 262.0],
        [0, 0, 1]
    ], dtype=np.float32)
    fx = K[0, 0]
    
    # 加载深度模型
    print("\n=== 加载模型 ===")
    import sys
    sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
    sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")
    
    from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader
    
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe 推理
    print("\n=== MoGe 输出分析 ===")
    moge_result = moge_loader.infer(img_rgb)
    depth_moge = moge_result['depth']
    
    print(f"MoGe depth shape: {depth_moge.shape}")
    print(f"MoGe depth min: {depth_moge.min()}")
    print(f"MoGe depth max: {depth_moge.max()}")
    print(f"MoGe depth mean: {depth_moge.mean()}")
    print(f"MoGe depth median: {np.median(depth_moge)}")
    print(f"MoGe depth std: {depth_moge.std()}")
    
    # 检查MoGe深度是否在[0,1]范围内（尺度不变）
    if depth_moge.max() <= 1.1:
        print("\n>>> MoGe输出看起来是尺度不变深度（归一化到[0,1]或类似范围）")
        print(">>> 需要缩放才能与DepthPro比较")
    elif depth_moge.max() > 5:
        print("\n>>> MoGe输出看起来是度量深度（米）")
    
    # 检查MoGe是否有mask
    if 'mask' in moge_result:
        mask = moge_result['mask']
        print(f"MoGe mask shape: {mask.shape}")
        print(f"MoGe mask valid pixels: {(mask > 0.5).sum()}")
    
    # DepthPro 推理
    print("\n=== DepthPro 输出分析 ===")
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=fx)
    depth_depthpro = depthpro_result['depth']
    
    print(f"DepthPro depth shape: {depth_depthpro.shape}")
    print(f"DepthPro depth min: {depth_depthpro.min()}")
    print(f"DepthPro depth max: {depth_depthpro.max()}")
    print(f"DepthPro depth mean: {depth_depthpro.mean()}")
    print(f"DepthPro depth median: {np.median(depth_depthpro)}")
    
    # 分析比例
    print("\n=== 深度比例分析 ===")
    
    # 如果MoGe是归一化的，需要找到正确的缩放因子
    # 假设: DepthPro = scale * MoGe + offset
    # 对于室内场景，如果MoGe是[0,1]归一化的，scale应该使得两者在有效范围内对齐
    
    # 简单计算：使用中位数作为参考
    moge_median = np.median(depth_moge[depth_moge > 0.1])
    depthpro_median = np.median(depth_depthpro[depth_depthpro > 0.1])
    
    simple_scale = depthpro_median / moge_median
    print(f"简单缩放因子（中位数比）: {simple_scale:.4f}")
    
    # 检查原始代码的RANSAC结果
    print(f"\n原始RANSAC scale=1.7009")
    print(f"这意味着: DepthPro ≈ 1.7009 * MoGe")
    print(f"如果MoGe=[0,1], 1.7倍的MoGe最大值={depth_moge.max()*1.7009:.2f}")
    print(f"但DepthPro最大值={depth_depthpro.max():.2f}")
    
    # 验证：MoGe归一化后乘以1.7是否能覆盖DepthPro范围
    moge_scaled = depth_moge * 1.7009
    print(f"\n缩放后MoGe范围: [{moge_scaled.min():.2f}, {moge_scaled.max():.2f}]")
    print(f"DepthPro范围: [{depth_depthpro.min():.2f}, {depth_depthpro.max():.2f}]")
    
    # 分析问题
    print("\n=== 问题分析 ===")
    
    # 如果MoGe是[0,1]范围
    if depth_moge.max() <= 1.1:
        # 那么MoGe * 1.7 ≈ [0, ~8]
        # 这覆盖了DepthPro的范围 [1.3, 11.1]
        print("MoGe可能是归一化的，RANSAC找到的scale=1.7是合理的")
    
    # 检查融合后的深度是否合理
    print("\n=== 融合深度分析 ===")
    
    # 尝试不同的融合策略
    strategies = {
        "RANSAC (scale=1.7009)": depth_moge * 1.7009,
        "简单中位数匹配": depth_moge * simple_scale,
        "纯DepthPro": depth_depthpro,
    }
    
    for name, fused in strategies.items():
        valid = fused[fused > 0.1]
        if len(valid) > 0:
            print(f"{name}:")
            print(f"  范围: [{valid.min():.2f}, {valid.max():.2f}], mean={valid.mean():.2f}")
            
            # 检查床区域 (200-400, 50-200)
            bed = fused[50:200, 200:400]
            valid_bed = bed[bed > 0.1]
            if len(valid_bed) > 0:
                print(f"  床区域: mean={valid_bed.mean():.2f}m, range=[{valid_bed.min():.2f}, {valid_bed.max():.2f}]")
    
    # 保存不同融合策略的结果
    print("\n=== 保存可视化 ===")
    
    for name, fused in strategies.items():
        if fused.shape != (h, w):
            fused_vis = cv2.resize(fused, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            fused_vis = fused
        
        vis = np.clip(fused_vis * 40, 0, 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        filename = name.replace(" ", "_").replace("=", "_").replace(".", "_")
        cv2.imwrite(f'{PROJECT_ROOT}/debug_{filename}.jpg', vis_color)
        print(f"保存: {PROJECT_ROOT}/debug_{filename}.jpg")
    
    print("\n完成!")

if __name__ == '__main__':
    check_depth_ranges()
