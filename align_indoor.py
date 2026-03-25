"""
改进的室内深度对齐策略
使用室内距离范围约束 RANSAC
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe')
sys.path.insert(0, '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge')
sys.path.insert(0, '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src')

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader
from sklearn.linear_model import RANSACRegressor, LinearRegression

def align_depth_indoor(relative_depth, metric_depth, moge_mask=None, 
                       indoor_range=(0.5, 10.0)):
    """
    室内优化的深度对齐策略
    
    1. 只使用室内距离范围内的点做 RANSAC
    2. 排除过远（如 >10m）的墙面干扰
    3. 使用中值作为备选
    """
    rel_flat = relative_depth.flatten().astype(np.float64)
    met_flat = metric_depth.flatten().astype(np.float64)
    
    # 基础有效掩码
    valid = (
        np.isfinite(rel_flat) & 
        np.isfinite(met_flat) & 
        (met_flat > indoor_range[0]) & 
        (met_flat < indoor_range[1])
    )
    
    # MoGe mask
    if moge_mask is not None:
        valid &= moge_mask.flatten()
    
    # 只用有效点
    rel_valid = rel_flat[valid].reshape(-1, 1)
    met_valid = met_flat[valid].reshape(-1, 1)
    
    print(f"  室内范围有效点: {len(rel_valid)} / {len(rel_flat)}")
    
    if len(rel_valid) < 100:
        scale = met_valid.mean() / (rel_valid.mean() + 1e-6)
        return relative_depth * scale
    
    # RANSAC
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        min_samples=0.2,
        residual_threshold=0.5,
        random_state=42,
    )
    
    try:
        ransac.fit(rel_valid, met_valid)
        scale = ransac.estimator_.coef_[0, 0]
        inlier_ratio = ransac.inlier_mask_.sum() / len(rel_valid)
        print(f"  RANSAC scale: {scale:.4f}, inlier ratio: {inlier_ratio:.1%}")
    except Exception as e:
        print(f"  RANSAC失败: {e}")
        scale = np.median(met_valid) / (np.median(rel_valid) + 1e-6)
        print(f"  使用中值scale: {scale:.4f}")
    
    # 如果内点比例太低，使用中值对齐
    if inlier_ratio < 0.5:
        scale_median = np.median(met_valid) / (np.median(rel_valid) + 1e-6)
        print(f"  内点比例太低 ({inlier_ratio:.1%})，使用中值scale: {scale_median:.4f}")
        scale = scale_median
    
    return relative_depth * scale


def main():
    import cv2
    
    # 加载图像和深度
    img = cv2.imread('datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 加载真实深度
    gt = np.load('datasets/sunrgbd/sunrgbd_trainval/metric3d_depth/000004_depth.npy')
    bed_box = [117, 98, 551, 451]
    x1, y1, x2, y2 = bed_box
    gt_bed = gt[y1:y2, x1:x2]
    gt_valid = gt_bed[gt_bed > 0.1]
    gt_median = np.median(gt_valid)
    
    print("加载模型...")
    moge = MoGeLoader()
    depthpro = DepthProLoader()
    moge.load_model()
    depthpro.load_model()
    
    # MoGe
    moge_result = moge.infer(img_rgb)
    depth_moge = moge_result['depth']
    moge_mask = moge_result['mask']
    moge_K = moge_result['intrinsics']
    
    # DepthPro
    dp_result = depthpro.infer(img_rgb, focal_length_px=moge_K[0, 0])
    depth_dp = dp_result['depth']
    
    # 调整大小
    if depth_moge.shape != (h, w):
        depth_moge = cv2.resize(depth_moge, (w, h))
        moge_mask = cv2.resize(moge_mask.astype(np.float32), (w, h)) > 0.5
    if depth_dp.shape != (h, w):
        depth_dp = cv2.resize(depth_dp, (w, h))
    
    print(f"\n=== 床区域对比 ===")
    print(f"真实深度: {gt_median:.2f}m")
    
    # 原版 RANSAC
    aligned_old = align_depth_indoor(depth_moge, depth_dp, moge_mask=None, indoor_range=(0.5, 100))
    print(f"原版RANSAC: {np.median(aligned_old[y1:y2, x1:x2]):.2f}m")
    
    # 改进版
    aligned_new = align_depth_indoor(depth_moge, depth_dp, moge_mask, indoor_range=(0.5, 10.0))
    print(f"改进版RANSAC: {np.median(aligned_new[y1:y2, x1:x2]):.2f}m")
    
    # 只用室内范围
    aligned_indoor = align_depth_indoor(depth_moge, depth_dp, moge_mask, indoor_range=(0.5, 5.0))
    print(f"室内5m范围: {np.median(aligned_indoor[y1:y2, x1:x2]):.2f}m")


if __name__ == '__main__':
    main()
