"""
诊断脚本：只跑 DetAny3D encoder，不跑任何几何推理，原样打印深度统计。
"""
import os
import sys
import cv2
import numpy as np
import torch

PROJECT_ROOT = "/data/ZhaoX/OVM3D-Det-1"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "teacher_student"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cubercnn"))

from teacher_student.teacher_detany3d import TeacherDetAny3D

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ========== 1. 加载图像 ==========
    img_path = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg"
    rgb = cv2.imread(img_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"图像尺寸: {w}x{h} (W x H)")

    # ========== 2. 内参：SUN RGB-D 真实 K ==========
    def get_sunrgbd_K(image_path):
        image_path = os.path.abspath(image_path)
        if "sunrgbd" not in image_path.lower():
            return None
        base = os.path.splitext(os.path.basename(image_path))[0]
        dir_img = os.path.dirname(image_path)
        root = os.path.dirname(dir_img)
        calib_path = os.path.join(root, "calib", base + ".txt")
        if not os.path.isfile(calib_path):
            return None
        with open(calib_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    vals = [float(x) for x in parts[:9]]
                    if abs(vals[8] - 1.0) < 1e-6 and abs(vals[1]) < 1e-6 and abs(vals[3]) < 1e-6:
                        fx, fy, cx, cy = vals[0], vals[4], vals[6], vals[7]
                        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)
        return None

    K = get_sunrgbd_K(img_path)
    print(f"原始 K:\n{K}")
    print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    # ========== 3. 加载教师（只用于取深度，不做几何推理） ==========
    print("\n初始化 TeacherDetAny3D（仅取深度）...")
    teacher = TeacherDetAny3D(device=device, use_sam2_mask=False)
    print("Teacher 就绪。")

    # ========== 4. 模拟 teacher_detany3d 的深度获取逻辑 ==========
    long_side = 896
    scale = long_side / float(max(h, w))
    h_resize = int(round(h * scale))
    w_resize = int(round(w * scale))
    print(f"\nDetAny3D encoder resize: {w_resize}x{h_resize} (scale={scale:.4f})")

    # K_geom: 与深度分辨率一致的 K
    K_geom = np.array(K, dtype=np.float64).copy()
    K_geom[0, 0] *= (w_resize / float(w))
    K_geom[1, 1] *= (h_resize / float(h))
    K_geom[0, 2] *= (w_resize / float(w))
    K_geom[1, 2] *= (h_resize / float(h))
    print(f"K_geom (用于几何反投影):\n{K_geom}")
    print(f"  fx={K_geom[0,0]:.2f}, fy={K_geom[1,1]:.2f}, cx={K_geom[0,2]:.2f}, cy={K_geom[1,2]:.2f}")

    # 深度在 resized 图像上算
    rgb_resized = np.ascontiguousarray(
        cv2.resize(rgb, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
    )
    pad_h = max(0, long_side - h_resize)
    pad_w = max(0, long_side - w_resize)
    padded = np.pad(rgb_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")[:long_side, :long_side]
    img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img_t_sam = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / \
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    gt_intrinsic = torch.eye(4).float().unsqueeze(0).to(device)
    gt_intrinsic[0, :3, :3] = torch.tensor(K_geom).float()
    input_dict = {
        "images": img_t_sam,
        "image_for_dino": img_t_sam,
        "vit_pad_size": torch.tensor([[h_resize // 16, w_resize // 16]], dtype=torch.long, device=device),
        "gt_intrinsic": gt_intrinsic,
    }

    print("\n运行 image_encoder (仅取深度，不做几何推理)...")
    with torch.no_grad():
        output_dict = teacher.image_encoder(input_dict)
    depth_raw = output_dict["depth_maps"][0, 0, :h_resize, :w_resize].cpu().numpy()
    print(f"深度图形状 (resized): {depth_raw.shape}")

    # ========== 5. 深度统计（resized 分辨率） ==========
    print("\n=== DetAny3D 深度统计 (resized, scale=1.0) ===")
    print(f"  min   = {depth_raw.min():.4f}")
    print(f"  max   = {depth_raw.max():.4f}")
    print(f"  mean  = {depth_raw.mean():.4f}")
    print(f"  median= {np.median(depth_raw):.4f}")
    print(f"  std   = {depth_raw.std():.4f}")
    p1, p5, p25, p75, p95, p99 = np.percentile(depth_raw, [1, 5, 25, 75, 95, 99])
    print(f"  p1={p1:.4f}, p5={p5:.4f}, p25={p25:.4f}, p75={p75:.4f}, p95={p95:.4f}, p99={p99:.4f}")

    # ========== 6. 反投影一个点验证几何 ==========
    # 取图像中心点
    u_c, v_c = w_resize // 2, h_resize // 2
    z_c = depth_raw[v_c, u_c]
    fx, fy, cx, cy = K_geom[0,0], K_geom[1,1], K_geom[0,2], K_geom[1,2]
    x_cam = (u_c - cx) * z_c / fx
    y_cam = (v_c - cy) * z_c / fy
    print(f"\n图像中心 ({u_c},{v_c}) 的 3D 反投影:")
    print(f"  深度 z={z_c:.4f}")
    print(f"  X={x_cam:.4f}, Y={y_cam:.4f}, Z={z_c:.4f} (单位?)")
    print(f"  → 如果 Z 在 1~8m 之间，则深度单位是米")
    print(f"  → 如果 Z 在 100~800 之间，则深度单位是厘米")

    # 取一个已知区域：图像底部中间（通常是地面，深度应较小）
    u_gnd, v_gnd = w_resize // 2, int(h_resize * 0.9)
    z_gnd = depth_raw[v_gnd, u_gnd]
    y_gnd = (v_gnd - cy) * z_gnd / fy
    print(f"\n图像底部 ({u_gnd},{v_gnd}) 反投影:")
    print(f"  深度 z={z_gnd:.4f}, Y_cam={y_gnd:.4f}")

    # ========== 7. 如果有 SUNRGBD 真实深度图，拿来对比 ==========
    depth_gt_path = img_path.replace("/image/", "/depth/")
    depth_gt_path = depth_gt_path.replace(".jpg", ".png")
    if not os.path.exists(depth_gt_path):
        depth_gt_path = depth_gt_path.replace("/depth/", "/depth_bfx/")
    if not os.path.exists(depth_gt_path):
        # 尝试其他可能的路径
        import glob
        candidates = glob.glob(os.path.join(os.path.dirname(img_path).replace("/image/", "/"), "*.png"))
        if candidates:
            depth_gt_path = candidates[0]
    if os.path.exists(depth_gt_path):
        print(f"\n\n=== 对比 SUNRGBD 真实深度: {depth_gt_path} ===")
        depth_sun = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED)
        if depth_sun is not None:
            print(f"  shape: {depth_sun.shape}, dtype: {depth_sun.dtype}")
            print(f"  min={depth_sun.min()}, max={depth_sun.max()}, mean={depth_sun.mean():.2f}")
            if depth_sun.dtype == np.uint16:
                depth_sun_m = depth_sun.astype(np.float32) / 1000.0
                print(f"  转为米 (÷1000): min={depth_sun_m.min():.4f}, max={depth_sun_m.max():.4f}, mean={depth_sun_m.mean():.4f}")
            # resize 到与 depth_raw 相同分辨率
            depth_sun_resized = cv2.resize(depth_sun_m if depth_sun.dtype == np.uint16 else depth_sun,
                                           (w_resize, h_resize), interpolation=cv2.INTER_NEAREST)
            print(f"\nDetAny3D vs SUNRGBD (两者均在 resized 分辨率):")
            print(f"  DetAny3D median={np.median(depth_raw):.4f}, SUNRGBD median={np.median(depth_sun_resized):.4f}")
            ratio = np.median(depth_raw) / max(np.median(depth_sun_resized), 0.001)
            print(f"  DetAny3D / SUNRGBD ratio = {ratio:.4f}")
            print(f"  → 若 ratio ≈ 1，深度已是米")
            print(f"  → 若 ratio ≈ 2~3，深度整体偏小")
            print(f"  → 若 ratio ≈ 0.3~0.5，深度整体偏大")
        else:
            print(f"  无法读取: {depth_gt_path}")
    else:
        print(f"\n  未找到 SUNRGBD 深度图: {depth_gt_path}")

    # ========== 8. 把 DetAny3D 深度 resize 回原图分辨率，并给出全局统计 ==========
    depth_orig = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)
    print(f"\n=== DetAny3D 深度 (resize 回原图 {w}x{h}) ===")
    print(f"  min={depth_orig.min():.4f}, max={depth_orig.max():.4f}")
    print(f"  mean={depth_orig.mean():.4f}, median={np.median(depth_orig):.4f}")

    # ========== 9. 关键：深度应该是什么量级？ ==========
    print("\n" + "="*60)
    print("判断：DetAny3D 深度是什么单位？")
    print("="*60)
    z_med = np.median(depth_raw)
    z_mean = depth_raw.mean()
    z_max = depth_raw.max()

    if z_med > 50:
        print(f"⚠️  深度中位数={z_med:.1f}，远大于室内合理范围 (1~8m)")
        print(f"   建议设置 depth_scale ≈ {z_med/2.0:.1f}，或直接 ÷{z_med:.1f} 归一化")
        print(f"   即：把深度当成某种归一化深度，需缩放到米")
    elif z_med > 10:
        print(f"⚠️  深度中位数={z_med:.1f}，偏大")
        print(f"   室内场景合理深度中位数应在 1~5m")
        print(f"   建议设置 depth_scale ≈ {z_med:.2f} 或验证是否需要归一化")
    elif z_med < 0.1:
        print(f"⚠️  深度中位数={z_med:.4f}，非常小")
        print(f"   可能是归一化深度 (0~1) 或 RelDepth 格式")
        print(f"   需乘以场景最大深度（如 8m）来估算米")
    else:
        print(f"✅ 深度中位数={z_med:.4f}，在室内合理范围内 (1~5m)")
        print(f"   深度单位很可能是米，不需要额外缩放")
    print()

if __name__ == "__main__":
    main()
