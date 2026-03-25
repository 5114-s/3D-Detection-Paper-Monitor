"""
分析 RANSAC 对齐问题
为什么床区域对齐后深度仍然偏小
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

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader

def analyze_ransac_issue():
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 加载真实深度
    gt_depth = np.load(f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/metric3d_depth/000004_depth.npy')
    print(f"=== 真实深度 (SUNRGBD metric3d) ===")
    print(f"范围: [{gt_depth.min():.3f}, {gt_depth.max():.3f}]")
    
    print("\n加载模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe + DepthPro
    moge_result = moge_loader.infer(img_rgb)
    moge_K = moge_result['intrinsics']
    depth_moge = moge_result['depth']
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
    depth_depthpro = depthpro_result['depth']
    
    # 调整大小
    if depth_moge.shape != (h, w):
        depth_moge = cv2.resize(depth_moge, (w, h))
    if depth_depthpro.shape != (h, w):
        depth_depthpro = cv2.resize(depth_depthpro, (w, h))
    
    print(f"\n=== 各模型床区域深度对比 ===")
    bed_box = [117, 98, 551, 451]
    x1, y1, x2, y2 = bed_box
    
    gt_bed = gt_depth[y1:y2, x1:x2]
    moge_bed = depth_moge[y1:y2, x1:x2]
    dp_bed = depth_depthpro[y1:y2, x1:x2]
    
    gt_valid = gt_bed[gt_bed > 0.1]
    moge_valid = moge_bed[moge_bed > 0.1]
    dp_valid = dp_bed[dp_bed > 0.1]
    
    print(f"真实深度: median={np.median(gt_valid):.3f}m")
    print(f"MoGe深度: median={np.median(moge_valid):.3f}")
    print(f"DepthPro深度: median={np.median(dp_valid):.3f}m")
    
    # 分析每个模型的准确性
    print(f"\n=== 模型误差分析 ===")
    
    # MoGe vs 真实深度
    mask = (moge_valid > 0.1) & (gt_valid[:len(moge_valid)] > 0.1) if len(gt_valid) >= len(moge_valid) else \
           (moge_valid[:len(gt_valid)] > 0.1) & (gt_valid > 0.1)
    if mask.sum() > 0:
        moge_ratio = np.median(gt_valid[:len(moge_valid)]) / np.median(moge_valid[mask]) if np.median(moge_valid[mask]) > 0 else 0
        print(f"MoGe scale (vs GT): {moge_ratio:.4f}")
    
    # DepthPro vs 真实深度
    if len(gt_valid) >= len(dp_valid):
        mask2 = (dp_valid > 0.1) & (gt_valid[:len(dp_valid)] > 0.1)
        if mask2.sum() > 0:
            dp_ratio = np.median(gt_valid[:len(dp_valid)][mask2]) / np.median(dp_valid[mask2]) if np.median(dp_valid[mask2]) > 0 else 0
    else:
        mask2 = (gt_valid[:len(dp_valid)] > 0.1) & (dp_valid > 0.1)
        if mask2.sum() > 0:
            dp_ratio = np.median(gt_valid[mask2]) / np.median(dp_valid[mask2]) if np.median(dp_valid[mask2]) > 0 else 0
    print(f"DepthPro scale (vs GT): {dp_ratio:.4f}")
    
    # 如果用 DepthPro 直接作为结果
    print(f"\n=== 如果只用 DepthPro ===")
    if gt_depth.shape != (h, w):
        gt_resized = cv2.resize(gt_depth, (w, h))
    else:
        gt_resized = gt_depth
    
    mask3 = (dp_valid > 0.1) & (gt_resized[y1:y2, x1:x2] > 0.1)
    dp_vs_gt = gt_resized[y1:y2, x1:x2][mask3] / dp_bed[mask3]
    print(f"DepthPro/GT 比值: median={np.median(dp_vs_gt):.4f}")
    
    # 分析墙面干扰
    print(f"\n=== 墙面干扰分析 ===")
    # 床区域大部分应该是床，但可能包含墙
    # 让我们看看远距离的墙面对齐效果
    
    # 全局对齐
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    valid_all = (depth_moge < 400) & (depth_depthpro < 100) & (depth_depthpro > 0.1) & (depth_moge > 0.1)
    
    X = depth_moge[valid_all].reshape(-1, 1)
    y = depth_depthpro[valid_all].reshape(-1, 1)
    
    # 全局 RANSAC
    ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=0.2)
    ransac.fit(X, y)
    global_scale = ransac.estimator_.coef_[0][0]
    print(f"全局 RANSAC scale: {global_scale:.4f}")
    
    # 只用床区域拟合
    mask_bed = (moge_bed > 0.1) & (dp_bed > 0.1)
    X_bed = moge_bed[mask_bed].reshape(-1, 1)
    y_bed = dp_bed[mask_bed].reshape(-1, 1)
    
    ransac_bed = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=0.2)
    ransac_bed.fit(X_bed, y_bed)
    bed_scale = ransac_bed.estimator_.coef_[0][0]
    print(f"床区域 RANSAC scale: {bed_scale:.4f}")
    
    # 用床区域scale对齐后的深度
    aligned_bed_depth = depth_moge * bed_scale
    aligned_bed = aligned_bed_depth[y1:y2, x1:x2]
    aligned_valid = aligned_bed[aligned_bed > 0.1]
    print(f"\n用床区域scale对齐后床深度: median={np.median(aligned_valid):.3f}m")
    print(f"真实床深度: median={np.median(gt_valid):.3f}m")
    
    # 结论
    print(f"\n=== 结论 ===")
    print(f"1. 全局 RANSAC scale: {global_scale:.4f} (对齐后 median=2.26m)")
    print(f"2. 床区域 RANSAC scale: {bed_scale:.4f}")
    print(f"3. 如果要达到真实深度(~3.8m):")
    print(f"   需要 scale = 3.8 / {np.median(moge_valid):.3f} = {3.8/np.median(moge_valid):.4f}")
    print(f"4. DepthPro 本身相对准确 (误差约 {(np.median(dp_valid)/np.median(gt_valid)-1)*100:.1f}%)")

if __name__ == '__main__':
    analyze_ransac_issue()
