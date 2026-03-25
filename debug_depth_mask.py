"""
调试脚本：检查深度和掩码是否正确
"""
import cv2
import numpy as np
import torch
import os

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'

def debug_depth_and_mask():
    # 1. 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {h}x{w}")
    
    # 2. 相机内参
    K = np.array([
        [529.5, 0, 365.0],
        [0, 529.5, 262.0],
        [0, 0, 1]
    ], dtype=np.float32)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # 3. 检查原始深度图
    print("=== 检查原始深度图 ===")
    
    # SUNRGBD symlink 指向 mmdetection3d
    sunrgbd_link = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd'
    if os.path.islink(sunrgbd_link):
        real_path = os.readlink(sunrgbd_link)
        print(f"Symlink: {sunrgbd_link} -> {real_path}")
        
        # 在真实路径下查找
        depth_path = os.path.join(real_path, 'sunrgbd_trainval', 'depth', '000004.png')
        if os.path.exists(depth_path):
            print(f"原始深度图: {depth_path}")
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                print(f"  形状: {depth_raw.shape}, dtype: {depth_raw.dtype}")
                print(f"  值范围: [{depth_raw.min()}, {depth_raw.max()}]")
                # SUNRGBD 深度图通常是 uint16，值是毫米
                depth_raw_m = depth_raw.astype(np.float32) / 1000.0
                print(f"  转换为米: min={depth_raw_m.min():.3f}, max={depth_raw_m.max():.3f}")
                
                # 调整大小用于比较
                if depth_raw_m.shape != (h, w):
                    depth_raw_resized = cv2.resize(depth_raw_m, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    depth_raw_resized = depth_raw_m
                
                # 比较床区域
                print("\n=== 床区域比较 ===")
                bed_x1, bed_y1, bed_x2, bed_y2 = 200, 50, 400, 200
                bed_region_raw = depth_raw_resized[bed_y1:bed_y2, bed_x1:bed_x2]
                valid_bed_raw = bed_region_raw[bed_region_raw > 0.1]
                
                center_u, center_v = (bed_x1 + bed_x2) // 2, (bed_y1 + bed_y2) // 2
                z_raw = depth_raw_resized[center_v, center_u]
                x_raw = (center_u - cx) * z_raw / fx
                y_raw = (center_v - cy) * z_raw / fy
                
                print(f"区域 ({bed_x1}-{bed_x2}, {bed_y1}-{bed_y2}):")
                print(f"  原始深度: min={valid_bed_raw.min():.3f}, max={valid_bed_raw.max():.3f}, mean={valid_bed_raw.mean():.3f}")
                print(f"  中心 ({center_u}, {center_v}): 3D=({x_raw:.3f}, {y_raw:.3f}, {z_raw:.3f})")
        else:
            print(f"未找到: {depth_path}")
            depth_raw_resized = None
    
    # 4. 加载MoGe+DepthPro
    print("\n=== MoGe + DepthPro ===")
    import sys
    sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
    sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
    sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")
    
    from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac
    
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    moge_result = moge_loader.infer(img_rgb)
    depth_moge = moge_result['depth']
    print(f"MoGe: min={depth_moge.min():.3f}, max={depth_moge.max():.3f}, shape={depth_moge.shape}")
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=fx)
    depth_depthpro = depthpro_result['depth']
    print(f"DepthPro: min={depth_depthpro.min():.3f}, max={depth_depthpro.max():.3f}, shape={depth_depthpro.shape}")
    
    aligned_depth, _ = align_depth_ransac(depth_moge, depth_depthpro)
    print(f"融合: min={aligned_depth.min():.3f}, max={aligned_depth.max():.3f}, shape={aligned_depth.shape}")
    
    if aligned_depth.shape != (h, w):
        aligned_depth_resized = cv2.resize(aligned_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        aligned_depth_resized = aligned_depth
    
    # 5. 比较床区域
    print("\n=== 融合深度床区域 ===")
    bed_x1, bed_y1, bed_x2, bed_y2 = 200, 50, 400, 200
    bed_region_fused = aligned_depth_resized[bed_y1:bed_y2, bed_x1:bed_x2]
    valid_bed_fused = bed_region_fused[bed_region_fused > 0.1]
    
    center_u, center_v = (bed_x1 + bed_x2) // 2, (bed_y1 + bed_y2) // 2
    z_fused = aligned_depth_resized[center_v, center_u]
    x_fused = (center_u - cx) * z_fused / fx
    y_fused = (center_v - cy) * z_fused / fy
    
    print(f"区域 ({bed_x1}-{bed_x2}, {bed_y1}-{bed_y2}):")
    print(f"  融合深度: min={valid_bed_fused.min():.3f}, max={valid_bed_fused.max():.3f}, mean={valid_bed_fused.mean():.3f}")
    print(f"  中心 ({center_u}, {center_v}): 3D=({x_fused:.3f}, {y_fused:.3f}, {z_fused:.3f})")
    
    # 6. 整体比较
    if 'depth_raw_resized' in dir() and depth_raw_resized is not None:
        print("\n=== 整体深度比较 ===")
        valid_mask = (depth_raw_resized > 0.1) & (aligned_depth_resized > 0.1)
        if valid_mask.sum() > 100:
            diff = np.abs(aligned_depth_resized[valid_mask] - depth_raw_resized[valid_mask])
            print(f"有效像素: {valid_mask.sum()}")
            print(f"平均绝对误差: {diff.mean():.3f}m")
            print(f"最大绝对误差: {diff.max():.3f}m")
            print(f"误差 < 0.5m: {(diff < 0.5).mean():.1%}")
            print(f"误差 < 1.0m: {(diff < 1.0).mean():.1%}")
    
    # 7. 保存
    print("\n=== 保存 ===")
    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_resized * 40, alpha=255/8), cv2.COLORMAP_JET)
    cv2.imwrite(f'{PROJECT_ROOT}/debug_aligned_depth.jpg', depth_vis)
    print(f"保存: {PROJECT_ROOT}/debug_aligned_depth.jpg")
    print("完成!")

if __name__ == '__main__':
    debug_depth_and_mask()
