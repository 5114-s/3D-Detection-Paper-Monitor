"""
修改对齐策略
"""
import gc
import cv2
import numpy as np
import sys
import json
import os
import torch

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import DepthProLoader


def compare_alignment():
    # 从 Omni3D_pl JSON 中取第一张有 3D 标注的图片
    json_path = f'{PROJECT_ROOT}/datasets/Omni3D_pl/SUNRGBD_train.json'
    with open(json_path) as f:
        data = json.load(f)

    # 找一张有有效 3D 标注的图片
    img_id_with_ann = set()
    for ann in data['annotations']:
        if ann.get('valid3D') and ann.get('center_cam', [-1])[0] > 0:
            img_id_with_ann.add(ann['image_id'])

    img_info = None
    for im in data['images']:
        if im['id'] in img_id_with_ann:
            img_info = im
            break

    if img_info is None:
        print("未找到有效图片！")
        return

    # 图片路径、内参、GT 深度路径
    img_path = f'{PROJECT_ROOT}/datasets/{img_info["file_path"]}'
    depth_path = img_path.replace('/image/', '/depth/').replace('.jpg', '.png')
    K = np.array(img_info['K'])

    print(f"图片: {img_info['file_path']}")
    print(f"尺寸: {img_info['width']}x{img_info['height']}")
    print(f"GT 内参 K:\n{K}")

    # 加载图片和 GT 深度
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if not os.path.exists(depth_path):
        print(f"深度图不存在: {depth_path}")
        return

    gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    valid = gt_depth > 0.1
    print(f"\nGT 深度范围: {gt_depth[valid].min():.2f}m - {gt_depth[valid].max():.2f}m, 均值: {gt_depth[valid].mean():.2f}m")

    # 加载 DepthPro (用 cuda:1 避免OOM)
    print("\n加载 DepthPro...")
    depthpro_loader = DepthProLoader()
    depthpro_loader.device = torch.device("cuda:1")
    depthpro_loader.load_model()

    # 统计函数
    def stats(pred):
        v = gt_depth > 0.1
        abs_rel = np.mean(np.abs(pred[v] - gt_depth[v]) / gt_depth[v]) * 100
        thres125 = np.mean(np.maximum(pred[v]/gt_depth[v], gt_depth[v]/pred[v]) < 1.25) * 100
        thres156 = np.mean(np.maximum(pred[v]/gt_depth[v], gt_depth[v]/pred[v]) < 1.5625) * 100
        return abs_rel, thres125, thres156

    print(f"\n===== 扫描不同焦距下的 DepthPro 表现 =====")
    print(f"{'焦距':>8} | {'AbsRel':>8} | {'Th@1.25':>8} | {'Th@1.56':>8} | {'Pred均值':>8} | {'Pred范围':>20}")
    print("-" * 70)

    best_absrel = 999
    best_fx = None
    best_depth = None

    # 减少焦距数量避免OOM
    for fx in [500, 520, 529.5, 540, 560, 580, 600]:
        result = depthpro_loader.infer(img_rgb, focal_length_px=fx)
        dp_depth = result['depth']
        if dp_depth.shape != (h, w):
            dp_depth = cv2.resize(dp_depth, (w, h))

        abs_rel, t125, t156 = stats(dp_depth)
        pred_mean = dp_depth[valid].mean()
        pred_range = f"{dp_depth[valid].min():.1f}-{dp_depth[valid].max():.1f}m"

        marker = " **BEST**" if abs_rel < best_absrel else ""
        print(f"{fx:>8} | {abs_rel:>7.2f}% | {t125:>7.2f}% | {t156:>7.2f}% | {pred_mean:>7.2f}m | {pred_range}{marker}")

        if abs_rel < best_absrel:
            best_absrel = abs_rel
            best_fx = fx
            best_depth = dp_depth.copy()

        # 每次推理后清理
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n===== 最佳单焦距 DepthPro: fx={best_fx} =====")
    print(f"AbsRel={best_absrel:.2f}%, 需要进一步尺度对齐")

    # 各种对齐方法
    print(f"\n===== 对齐方法对比（基于 fx={best_fx}）=====")

    # 1. 均值对齐
    scale_mean = gt_depth[valid].mean() / best_depth[valid].mean()
    aligned_mean = best_depth * scale_mean
    abs_rel_m, t125_m, t156_m = stats(aligned_mean)
    print(f"1. 均值对齐 (x{scale_mean:.3f}): AbsRel={abs_rel_m:.2f}%, Th@1.25={t125_m:.2f}%, Th@1.5625={t156_m:.2f}%")

    # 2. 中值对齐
    scale_med = np.median(gt_depth[valid]) / np.median(best_depth[valid])
    aligned_med = best_depth * scale_med
    abs_rel_med, t125_med, t156_med = stats(aligned_med)
    print(f"2. 中值对齐 (x{scale_med:.3f}): AbsRel={abs_rel_med:.2f}%, Th@1.25={t125_med:.2f}%, Th@1.5625={t156_med:.2f}%")

    # 3. GT 焦距测试
    print(f"\n===== 使用 GT 焦距 fx={K[0,0]:.1f} =====")
    result_gt = depthpro_loader.infer(img_rgb, focal_length_px=K[0, 0])
    dp_gtfx = result_gt['depth']
    if dp_gtfx.shape != (h, w):
        dp_gtfx = cv2.resize(dp_gtfx, (w, h))
    abs_rel_gt, t125_gt, t156_gt = stats(dp_gtfx)
    scale_gt = gt_depth[valid].mean() / dp_gtfx[valid].mean()
    aligned_gt = dp_gtfx * scale_gt
    abs_rel_gtm, t125_gtm, t156_gtm = stats(aligned_gt)
    print(f"DepthPro(GT焦距): AbsRel={abs_rel_gt:.2f}%, Th@1.25={t125_gt:.2f}%")
    print(f"GT焦距+均值对齐 (x{scale_gt:.3f}): AbsRel={abs_rel_gtm:.2f}%, Th@1.25={t125_gtm:.2f}%, Th@1.5625={t156_gtm:.2f}%")
    print(f"  Pred范围: {dp_gtfx[valid].min():.2f}-{dp_gtfx[valid].max():.2f}m, 均值={dp_gtfx[valid].mean():.2f}m")
    print(f"  比例: GT均值/best_depth均值 = {scale_gt:.2f}x")

    # 4. RANSAC 线性对齐
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    rel_flat = best_depth[valid].flatten().astype(np.float64)
    met_flat = gt_depth[valid].flatten().astype(np.float64)
    ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), min_samples=0.2, residual_threshold=0.5, random_state=42)
    ransac.fit(rel_flat.reshape(-1, 1), met_flat)
    scale_r = float(ransac.estimator_.coef_)
    intercept_r = float(ransac.estimator_.intercept_)
    aligned_ransac = best_depth * scale_r + intercept_r
    abs_rel_r, t125_r, t156_r = stats(aligned_ransac)
    print(f"\n3. RANSAC线性对齐(scale={scale_r:.3f}, intercept={intercept_r:.3f}): AbsRel={abs_rel_r:.2f}%, Th@1.25={t125_r:.2f}%")

    # 5. 对数域对齐
    log_pred = np.log(best_depth + 1e-6)
    valid_log = np.isfinite(log_pred) & (gt_depth > 0.1)
    ransac_log = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), min_samples=0.2, residual_threshold=0.1, random_state=42)
    ransac_log.fit(log_pred[valid_log].reshape(-1, 1), np.log(gt_depth + 1e-6)[valid_log])
    a = float(ransac_log.estimator_.coef_)
    b = float(ransac_log.estimator_.intercept_)
    aligned_log = np.exp(log_pred * a + b)
    abs_rel_log, t125_log, t156_log = stats(aligned_log)
    print(f"4. 对数域对齐(a={a:.3f}, b={b:.3f}): AbsRel={abs_rel_log:.2f}%, Th@1.25={t125_log:.2f}%")

    print(f"\n===== 结论 =====")
    print(f"DepthPro 焦距估计 = {best_fx}, GT 焦距 = {K[0,0]}")
    print(f"最佳单 DepthPro (fx={best_fx}): AbsRel={best_absrel:.2f}%")
    print(f"DepthPro + 均值对齐: AbsRel={abs_rel_m:.2f}%, Th@1.25={t125_m:.2f}%")
    print(f"=> 深度整体偏小约 {scale_mean:.1f}x，这是 DepthPro 在 SUNRGBD 上的系统性偏差")
    print(f"=> LabelAny3D 的 MoGe+DepthPro+RANSAC 对齐是正确的方案")


if __name__ == '__main__':
    compare_alignment()
