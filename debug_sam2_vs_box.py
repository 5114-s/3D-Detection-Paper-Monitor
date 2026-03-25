"""
调试 SAM2 掩码 vs 2D框 的差异
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
from cubercnn.generate_label.util import project_image_to_cam
from teacher_student.teacher_geometry import create_uv_depth

def debug_sam2_vs_box():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    print("加载模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe + DepthPro
    moge_result = moge_loader.infer(img_rgb)
    moge_K = moge_result['intrinsics']
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
    aligned_depth, _ = align_depth_ransac(moge_result['depth'], depthpro_result['depth'])
    
    if aligned_depth.shape != (h, w):
        aligned_depth = cv2.resize(aligned_depth, (w, h))
    
    # 加载 SAM2
    print("加载 SAM2...")
    sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/sam2')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    GROUNDED_SAM_DIR = f'{PROJECT_ROOT}/Grounded-SAM-2'
    sam2_cfg = os.path.join(GROUNDED_SAM_DIR, "sam2", "configs", "sam2", "sam2_hiera_s.yaml")
    sam2_ckpt = os.path.join(GROUNDED_SAM_DIR, "checkpoints", "sam2_hiera_small.pt")
    
    # build_sam2 需要在 Grounded-SAM-2 目录下才能正确解析 config
    cwd = os.getcwd()
    os.chdir(GROUNDED_SAM_DIR)
    try:
        sam2_model = build_sam2("sam2_hiera_s.yaml", sam2_ckpt, device='cuda')
    finally:
        os.chdir(cwd)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.set_image(img_rgb)
    
    # 床的2D框
    bed_box = [117, 98, 551, 451]  # Grounding DINO 检测结果
    x1, y1, x2, y2 = bed_box
    
    # SAM2 掩码 (用box prompt)
    box = np.array([[x1, y1, x2, y2]], dtype=np.float64)
    mask_sam2, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=box, multimask_output=False
    )
    mask_sam2 = mask_sam2[0].astype(np.float32)
    
    print(f"\n{'='*60}")
    print("SAM2 掩码 vs 2D框 对比")
    print(f"{'='*60}")
    
    # 1. 2D框区域深度统计
    depth_box = aligned_depth[y1:y2, x1:x2]
    valid_box = depth_box[depth_box > 0.1]
    print(f"\n2D框区域 ({len(valid_box)} 像素):")
    print(f"  深度: min={valid_box.min():.2f}, max={valid_box.max():.2f}")
    print(f"  深度: mean={valid_box.mean():.2f}, median={np.median(valid_box):.2f}")
    print(f"  深度: std={valid_box.std():.2f}")
    
    # 2. SAM2掩码区域深度统计
    mask_in_box = mask_sam2[y1:y2, x1:x2]
    depth_sam2 = aligned_depth * mask_sam2
    depth_sam2_in_box = depth_sam2[y1:y2, x1:x2]
    valid_sam2 = depth_sam2_in_box[mask_in_box > 0.5]
    print(f"\nSAM2掩码区域 ({len(valid_sam2)} 像素):")
    print(f"  深度: min={valid_sam2.min():.2f}, max={valid_sam2.max():.2f}")
    print(f"  深度: mean={valid_sam2.mean():.2f}, median={np.median(valid_sam2):.2f}")
    print(f"  深度: std={valid_sam2.std():.2f}")
    
    # 3. 点云对比
    print(f"\n{'='*60}")
    print("点云对比")
    print(f"{'='*60}")
    
    # 2D框点云
    box_mask = np.zeros((h, w), dtype=np.float32)
    box_mask[y1:y2, x1:x2] = 1.0
    depth_for_pc = np.clip(aligned_depth, 0.3, 8.0) * box_mask
    uv_depth_box = create_uv_depth(depth_for_pc, box_mask)
    pc_box = project_image_to_cam(uv_depth_box, moge_K)
    print(f"\n2D框点云: {len(pc_box)} 点")
    print(f"  X: [{pc_box[:,0].min():.2f}, {pc_box[:,0].max():.2f}]")
    print(f"  Y: [{pc_box[:,1].min():.2f}, {pc_box[:,1].max():.2f}]")
    print(f"  Z: [{pc_box[:,2].min():.2f}, {pc_box[:,2].max():.2f}]")
    
    # SAM2掩码点云
    uv_depth_sam2 = create_uv_depth(np.clip(aligned_depth, 0.3, 8.0).astype(np.float64), mask_sam2.astype(np.float64))
    pc_sam2 = project_image_to_cam(uv_depth_sam2, moge_K)
    print(f"\nSAM2掩码点云: {len(pc_sam2)} 点")
    print(f"  X: [{pc_sam2[:,0].min():.2f}, {pc_sam2[:,0].max():.2f}]")
    print(f"  Y: [{pc_sam2[:,1].min():.2f}, {pc_sam2[:,1].max():.2f}]")
    print(f"  Z: [{pc_sam2[:,2].min():.2f}, {pc_sam2[:,2].max():.2f}]")
    
    # 4. 分析深度带
    print(f"\n{'='*60}")
    print("深度带分析")
    print(f"{'='*60}")
    
    for name, pc, valid_depths in [
        ("2D框", pc_box, valid_box),
        ("SAM2", pc_sam2, valid_sam2)
    ]:
        if len(pc) > 0:
            z_med = np.median(valid_depths)
            z_lo = max(0.3, z_med - 1.5)
            z_hi = min(8.0, z_med + 1.5)
            
            # 过滤
            valid_filtered = (pc[:, 2] > z_lo) & (pc[:, 2] < z_hi)
            pc_filtered = pc[valid_filtered]
            
            print(f"\n{name} (中值深度={z_med:.2f}m):")
            print(f"  过滤后点数: {len(pc_filtered)}/{len(pc)}")
            
            if len(pc_filtered) > 0:
                dx = pc_filtered[:,0].max() - pc_filtered[:,0].min()
                dy = pc_filtered[:,1].max() - pc_filtered[:,1].min()
                dz = pc_filtered[:,2].max() - pc_filtered[:,2].min()
                cx = pc_filtered[:,0].mean()
                cy = pc_filtered[:,1].mean()
                cz = pc_filtered[:,2].mean()
                
                print(f"  尺寸: X={dx:.2f}m, Y={dy:.2f}m, Z={dz:.2f}m")
                print(f"  中心: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
                
                # 投影验证
                u_check = moge_K[0,0] * cx / cz + moge_K[0,2]
                v_check = moge_K[1,1] * cy / cz + moge_K[1,2]
                center_u = (x1 + x2) // 2
                center_v = (y1 + y2) // 2
                print(f"  投影: ({u_check:.0f}, {v_check:.0f}) vs 实际: ({center_u}, {center_v})")
    
    # 5. 保存可视化
    print(f"\n{'='*60}")
    print("保存可视化...")
    print(f"{'='*60}")
    
    # 创建掩码对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    import matplotlib.pyplot as plt
    
    # 2D框
    ax = axes[0, 0]
    box_vis = img_rgb.copy()
    cv2.rectangle(box_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    ax.imshow(box_vis)
    ax.set_title('2D Box')
    
    # SAM2掩码
    ax = axes[0, 1]
    sam2_vis = (mask_sam2 * 255).astype(np.uint8)
    ax.imshow(sam2_vis, cmap='gray')
    ax.set_title('SAM2 Mask')
    
    # 掩码差异
    ax = axes[0, 2]
    diff = np.abs(mask_sam2 - box_mask[y1:y2, x1:x2])
    ax.imshow(diff, cmap='hot')
    ax.set_title('Difference')
    
    # 深度图
    ax = axes[1, 0]
    depth_vis = (aligned_depth / 8.0 * 255).astype(np.uint8)
    ax.imshow(depth_vis, cmap='turbo')
    ax.set_title('Depth Map')
    
    # 深度直方图对比
    ax = axes[1, 1]
    ax.hist(valid_box, bins=50, alpha=0.5, label='2D Box', range=(0, 8))
    ax.hist(valid_sam2, bins=50, alpha=0.5, label='SAM2', range=(0, 8))
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Count')
    ax.set_title('Depth Histogram')
    ax.legend()
    
    # SAM2点云俯视图
    ax = axes[1, 2]
    ax.scatter(pc_sam2[:, 0], pc_sam2[:, 2], c=pc_sam2[:, 2], s=0.5, cmap='turbo', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('SAM2 Point Cloud (Top View)')
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 6])
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_ROOT}/debug_sam2_vs_box.jpg', dpi=150)
    print(f"可视化已保存: {PROJECT_ROOT}/debug_sam2_vs_box.jpg")

if __name__ == '__main__':
    debug_sam2_vs_box()
