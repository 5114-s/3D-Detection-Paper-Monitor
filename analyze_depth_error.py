"""
分析深度误差来源
"""
import cv2
import numpy as np
import sys
import os

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac

def analyze_depth_error():
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    print("加载模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe 原始结果
    moge_result = moge_loader.infer(img_rgb)
    moge_K = moge_result['intrinsics']
    depth_moge = moge_result['depth']
    print(f"\n=== MoGe 原始结果 ===")
    print(f"MoGe K: fx={moge_K[0,0]:.1f}, fy={moge_K[1,1]:.1f}")
    print(f"MoGe 深度范围: [{depth_moge.min():.3f}, {depth_moge.max():.3f}]")
    
    # 床区域 MoGe 深度
    bed_box = [117, 98, 551, 451]
    x1, y1, x2, y2 = bed_box
    moge_bed = depth_moge[y1:y2, x1:x2] if depth_moge.shape == (h, w) else \
               cv2.resize(depth_moge, (w, h))[y1:y2, x1:x2]
    valid_moge = moge_bed[moge_bed > 0.1]
    print(f"MoGe 床区域深度: median={np.median(valid_moge):.3f}")
    
    # DepthPro 原始结果
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
    depth_depthpro = depthpro_result['depth']
    
    # 调整到原图大小
    if depth_depthpro.shape != (h, w):
        depth_depthpro = cv2.resize(depth_depthpro, (w, h))
    
    print(f"\n=== DepthPro 原始结果 ===")
    print(f"DepthPro 深度范围: [{depth_depthpro.min():.3f}, {depth_depthpro.max():.3f}]")
    
    # 床区域 DepthPro 深度
    depthpro_bed = depth_depthpro[y1:y2, x1:x2]
    valid_dp = depthpro_bed[depthpro_bed > 0.1]
    print(f"DepthPro 床区域深度: median={np.median(valid_dp):.3f}")
    
    # RANSAC 对齐
    aligned_depth, _ = align_depth_ransac(depth_moge, depth_depthpro)
    if aligned_depth.shape != (h, w):
        aligned_depth = cv2.resize(aligned_depth, (w, h))
    
    print(f"\n=== 对齐后深度 ===")
    print(f"对齐后深度范围: [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}]")
    
    aligned_bed = aligned_depth[y1:y2, x1:x2]
    valid_aligned = aligned_bed[aligned_bed > 0.1]
    print(f"对齐后床区域深度: median={np.median(valid_aligned):.3f}")
    
    # 分析RANSAC对齐
    print(f"\n=== RANSAC 对齐分析 ===")
    # 检查对齐的scale
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    
    # 使用床区域的有效点
    mask_moge = moge_bed > 0.1
    mask_dp = depthpro_bed > 0.1
    mask = mask_moge & mask_dp
    
    if mask.sum() > 100:
        moge_flat = moge_bed[mask].reshape(-1, 1)
        dp_flat = depthpro_bed[mask].reshape(-1, 1)
        
        # RANSAC 拟合
        ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=0.2)
        ransac.fit(moge_flat, dp_flat)
        scale = ransac.estimator_.coef_[0][0]
        inlier_ratio = ransac.inlier_mask_.sum() / mask.sum() if hasattr(ransac, 'inlier_mask_') else 0
        print(f"RANSAC scale (MoGe -> DepthPro): {scale:.4f}")
        print(f"RANSAC inlier_ratio: {inlier_ratio:.2%}")
        
        # 用全部数据拟合
        lr = LinearRegression(fit_intercept=False)
        lr.fit(moge_flat, dp_flat)
        print(f"Linear Regression scale: {lr.coef_[0][0]:.4f}")
    
    # 如果SUNRGBD有真实深度，比较一下
    print(f"\n=== 深度比较 ===")
    print(f"当前估计床深度: {np.median(valid_aligned):.2f}m")
    print(f"如果要达到真实深度(~3.8m):")
    print(f"  需要scale: {3.8 / np.median(valid_moge):.2f}")
    
    # 检查原始深度文件
    depth_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/depth/000004.jpg'
    if os.path.exists(depth_path):
        import PIL.Image
        depth_raw = np.array(PIL.Image.open(depth_path))
        print(f"\n=== SUNRGBD 原始深度 ===")
        print(f"原始深度范围: [{depth_raw.min()}, {depth_raw.max()}]")
        print(f"原始深度类型: {depth_raw.dtype}")
        # SUNRGBD 深度通常需要 /8000 转换到米
        depth_m = depth_raw.astype(np.float32) / 8000.0
        depth_m_bed = depth_m[y1:y2, x1:x2]
        valid_raw = depth_m_bed[depth_m_bed > 0.01]
        print(f"SUNRGBD床区域深度(转换后): median={np.median(valid_raw):.2f}m")
        print(f"SUNRGBD床区域深度: min={valid_raw.min():.2f}, max={valid_raw.max():.2f}")

if __name__ == '__main__':
    analyze_depth_error()
