"""
分析 SAM2 掩码和深度带过滤问题
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac
from cubercnn.generate_label.util import extract_ground, project_image_to_cam
from teacher_student.teacher_geometry import create_uv_depth

def debug_depth_band():
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
    
    # 正确的床框
    bed_box = [117, 98, 551, 451]
    x1, y1, x2, y2 = bed_box
    
    print(f"\n{'='*60}")
    print("深度分析")
    print(f"{'='*60}")
    
    # 1. 床框内全部深度
    bed_depth = aligned_depth[y1:y2, x1:x2]
    valid = bed_depth[bed_depth > 0.1]
    print(f"床框内全部深度 ({len(valid)} 像素):")
    print(f"  min={valid.min():.2f}, max={valid.max():.2f}")
    print(f"  mean={valid.mean():.2f}, median={np.median(valid):.2f}")
    print(f"  std={valid.std():.2f}")
    
    # 2. 分析深度分布
    print(f"\n深度分布:")
    for thresh in [2.0, 2.5, 3.0, 3.5, 4.0]:
        count = (valid < thresh).sum()
        pct = count / len(valid) * 100
        print(f"  < {thresh}m: {count} ({pct:.1f}%)")
    
    # 3. 深度带分析
    z_med = np.median(valid)
    z_q25, z_q75 = np.percentile(valid, 25), np.percentile(valid, 75)
    iqr = z_q75 - z_q25
    print(f"\n深度统计:")
    print(f"  中值: {z_med:.2f}m")
    print(f"  25%分位: {z_q25:.2f}m")
    print(f"  75%分位: {z_q75:.2f}m")
    print(f"  IQR: {iqr:.2f}m")
    
    # 尝试不同的深度带
    for band_scale in [0.5, 1.0, 1.5, 2.0]:
        z_lo = max(0.3, z_med - band_scale)
        z_hi = min(8.0, z_med + band_scale)
        depth_filtered = (aligned_depth > z_lo) & (aligned_depth < z_hi)
        filtered_count = depth_filtered[y1:y2, x1:x2].sum()
        pct = filtered_count / bed_depth.size * 100
        print(f"\n深度带 [{z_lo:.1f}, {z_hi:.1f}]m:")
        print(f"  床框内过滤后像素: {filtered_count}/{bed_depth.size} ({pct:.1f}%)")
        
        # 反投影
        if filtered_count > 0:
            mask_temp = np.zeros((h, w), dtype=np.float32)
            mask_temp[depth_filtered & (np.arange(h)[:, None] >= y1) & (np.arange(h)[:, None] < y2) &
                     (np.arange(w)[None, :] >= x1) & (np.arange(w)[None, :] < x2)] = 1.0
            depth_temp = np.clip(aligned_depth, 0.3, 8.0) * mask_temp
            uv_depth_temp = create_uv_depth(depth_temp, mask_temp)
            pc_temp = project_image_to_cam(uv_depth_temp, moge_K)
            
            if len(pc_temp) > 0:
                dx = pc_temp[:,0].max() - pc_temp[:,0].min()
                dy = pc_temp[:,1].max() - pc_temp[:,1].min()
                dz = pc_temp[:,2].max() - pc_temp[:,2].min()
                print(f"  点云尺寸: X={dx:.2f}m, Y={dy:.2f}m, Z={dz:.2f}m")
    
    # 4. 检查 teacher_geometry 的自适应腐蚀
    print(f"\n{'='*60}")
    print("Teacher Geometry 深度带分析")
    print(f"{'='*60}")
    
    # teacher_geometry 使用的参数
    depth_band_max = 2.0
    z_lo = max(0.3, z_med - depth_band_max)
    z_hi = min(8.0, z_med + depth_band_max)
    
    depth_band = (aligned_depth > z_lo) & (aligned_depth < z_hi)
    depth_band_count = depth_band[y1:y2, x1:x2].sum()
    print(f"使用 depth_band_max={depth_band_max}:")
    print(f"  深度带: [{z_lo:.2f}, {z_hi:.2f}]m")
    print(f"  床框内像素: {depth_band_count}/{bed_depth.size} ({depth_band_count/bed_depth.size*100:.1f}%)")
    
    # 反投影过滤后的点云
    bed_mask_temp = np.zeros((h, w), dtype=np.float32)
    bed_mask_temp[y1:y2, x1:x2] = 1.0
    
    depth_for_pc = np.clip(aligned_depth, 0.3, 8.0) * (depth_band.astype(np.float32) * bed_mask_temp)
    uv_depth_pc = create_uv_depth(depth_for_pc, depth_for_pc > 0)
    pc = project_image_to_cam(uv_depth_pc, moge_K)
    
    print(f"\n反投影后点云: {len(pc)} 点")
    if len(pc) > 0:
        print(f"  X范围: [{pc[:,0].min():.2f}, {pc[:,0].max():.2f}]")
        print(f"  Y范围: [{pc[:,1].min():.2f}, {pc[:,1].max():.2f}]")
        print(f"  Z范围: [{pc[:,2].min():.2f}, {pc[:,2].max():.2f}]")
        
        dx = pc[:,0].max() - pc[:,0].min()
        dy = pc[:,1].max() - pc[:,1].min()
        dz = pc[:,2].max() - pc[:,2].min()
        cx = pc[:,0].mean()
        cy = pc[:,1].mean()
        cz = pc[:,2].mean()
        
        print(f"  估计尺寸: X={dx:.2f}m, Y={dy:.2f}m, Z={dz:.2f}m")
        print(f"  估计中心: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        print(f"  真实床: 中心(~2.5, ~0, ~3.8), 尺寸(2.0 x 0.5 x 1.5)")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 原图 + 床框
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue', linewidth=2))
    ax.set_title('Image + Bed Box (117, 98, 551, 451)')
    
    # 2. 深度图
    ax = axes[0, 1]
    depth_vis = (aligned_depth / 8.0 * 255).astype(np.uint8)
    ax.imshow(depth_vis, cmap='turbo')
    ax.set_title('Aligned Depth (clipped to 8m)')
    
    # 3. 深度直方图
    ax = axes[0, 2]
    ax.hist(valid, bins=50, range=(0, 8))
    ax.axvline(x=z_med, color='r', label=f'median={z_med:.2f}')
    ax.axvline(x=3.8, color='g', label='true depth~3.8')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Count')
    ax.set_title('Depth Histogram in Bed Box')
    ax.legend()
    
    # 4. teacher_geometry 的深度带
    ax = axes[1, 0]
    depth_band_vis = depth_band.astype(np.uint8) * 255
    ax.imshow(depth_band_vis, cmap='gray')
    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    ax.set_title(f'Depth Band [{z_lo:.1f}, {z_hi:.1f}]m (teacher_geometry)')
    
    # 5. 深度带内点云俯视图
    ax = axes[1, 1]
    if len(pc) > 0:
        ax.scatter(pc[:, 0], pc[:, 2], c=pc[:, 2], s=0.5, cmap='turbo', alpha=0.5)
        ax.axvline(x=cx, color='r', label='center x')
        ax.set_xlim([-3, 3])
        ax.set_ylim([0, 6])
        ax.invert_yaxis()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('Point Cloud Top View (filtered)')
        ax.legend()
    else:
        ax.set_title('No points after filtering')
    
    # 6. 深度剖面
    ax = axes[1, 2]
    row = (y1 + y2) // 2
    depth_row = aligned_depth[row, :]
    ax.plot(depth_row, label='depth')
    ax.axvline(x=x1, color='b', label='x1')
    ax.axvline(x=x2, color='b', label='x2')
    ax.axhline(y=z_med, color='r', label=f'med={z_med:.1f}')
    ax.axhline(y=3.8, color='g', label='true=3.8')
    ax.axhspan(z_lo, z_hi, alpha=0.3, color='yellow', label='band')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Depth Profile at Row {row}')
    ax.legend(fontsize=8)
    ax.set_ylim([0, 8])
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_ROOT}/debug_depth_band.jpg', dpi=150)
    print(f"\n调试图已保存: {PROJECT_ROOT}/debug_depth_band.jpg")

if __name__ == '__main__':
    debug_depth_band()
