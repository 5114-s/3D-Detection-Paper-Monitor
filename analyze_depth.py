"""
分析当前深度图质量
"""
import cv2
import numpy as np
import os

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'

def analyze_depth():
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
    
    # 3. 加载MoGe+DepthPro深度
    print("\n=== 加载深度模型 ===")
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
    
    # 4. 分别获取两个深度
    print("\n=== 各模型深度 ===")
    moge_result = moge_loader.infer(img_rgb)
    depth_moge = moge_result['depth']
    print(f"MoGe: min={depth_moge.min():.3f}, max={depth_moge.max():.3f}, mean={depth_moge.mean():.3f}")
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=fx)
    depth_depthpro = depthpro_result['depth']
    print(f"DepthPro: min={depth_depthpro.min():.3f}, max={depth_depthpro.max():.3f}, mean={depth_depthpro.mean():.3f}")
    
    # 5. 融合
    aligned_depth, _ = align_depth_ransac(depth_moge, depth_depthpro)
    print(f"融合: min={aligned_depth.min():.3f}, max={aligned_depth.max():.3f}, mean={aligned_depth.mean():.3f}")
    
    # 调整到原图大小
    if aligned_depth.shape != (h, w):
        depth_final = cv2.resize(aligned_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_final = aligned_depth
    
    # 6. 分析不同区域
    print("\n=== 区域分析 ===")
    
    regions = {
        "左上角": (0, 0, 200, 200),
        "右上角": (500, 0, 730, 200),
        "左下角": (0, 300, 200, 530),
        "右下角": (500, 300, 730, 530),
        "中心": (200, 150, 500, 400),
        "床区域": (200, 50, 400, 200),
    }
    
    for name, (x1, y1, x2, y2) in regions.items():
        region = depth_final[y1:y2, x1:x2]
        valid = region[region > 0.1]
        if len(valid) > 0:
            print(f"{name} ({x1}-{x2}, {y1}-{y2}): mean={valid.mean():.3f}m, range=[{valid.min():.3f}, {valid.max():.3f}]")
            
            # 反投影中心点
            center_u, center_v = (x1+x2)//2, (y1+y2)//2
            z = depth_final[center_v, center_u]
            x = (center_u - cx) * z / fx
            y = (center_v - cy) * z / fy
            print(f"  中心点3D: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # 7. 保存深度图可视化
    print("\n=== 保存可视化 ===")
    
    # 图像上叠加深度伪彩色
    overlay = img_rgb.copy()
    
    # 创建深度伪彩色
    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_final * 40, alpha=255/8), cv2.COLORMAP_JET)
    
    # 叠加
    alpha = 0.6
    result = cv2.addWeighted(depth_vis, alpha, img, 1-alpha, 0)
    
    # 画一些标记点
    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 标记中心点
        center_u, center_v = (x1+x2)//2, (y1+y2)//2
        cv2.circle(result, (center_u, center_v), 5, (255, 255, 0), -1)
    
    cv2.imwrite(f'{PROJECT_ROOT}/debug_depth_overlay.jpg', result)
    print(f"保存: {PROJECT_ROOT}/debug_depth_overlay.jpg")
    
    # 分别保存MoGe和DepthPro
    # 确保深度值在合理范围内
    moge_vis_raw = np.clip(depth_moge * 40, 0, 255).astype(np.uint8)
    moge_vis = cv2.applyColorMap(moge_vis_raw, cv2.COLORMAP_JET)
    cv2.imwrite(f'{PROJECT_ROOT}/debug_moge_depth.jpg', moge_vis)
    
    depthpro_vis_raw = np.clip(depth_depthpro * 40, 0, 255).astype(np.uint8)
    depthpro_vis = cv2.applyColorMap(depthpro_vis_raw, cv2.COLORMAP_JET)
    cv2.imwrite(f'{PROJECT_ROOT}/debug_depthpro_depth.jpg', depthpro_vis)
    
    # 保存融合深度
    aligned_vis = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth * 40, alpha=255/8), cv2.COLORMAP_JET)
    cv2.imwrite(f'{PROJECT_ROOT}/debug_fused_depth.jpg', aligned_vis)
    
    print(f"保存: {PROJECT_ROOT}/debug_moge_depth.jpg")
    print(f"保存: {PROJECT_ROOT}/debug_depthpro_depth.jpg")
    print(f"保存: {PROJECT_ROOT}/debug_fused_depth.jpg")
    
    # 8. 检查深度一致性（相邻像素差异）
    print("\n=== 深度平滑度检查 ===")
    dy, dx = np.gradient(depth_final)
    gradient_mag = np.sqrt(dx**2 + dy**2)
    valid_grad = gradient_mag[depth_final > 0.1]
    print(f"梯度均值: {valid_grad.mean():.4f}")
    print(f"梯度最大值: {valid_grad.max():.4f}")
    print(f"梯度 > 0.5 的比例: {(valid_grad > 0.5).mean():.2%}")
    print(f"梯度 > 1.0 的比例: {(valid_grad > 1.0).mean():.2%}")
    
    print("\n完成!")

if __name__ == '__main__':
    analyze_depth()
