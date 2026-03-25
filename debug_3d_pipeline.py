"""
详细调试3D框拟合问题
逐个检查：深度 -> 点云 -> 掩码 -> 3D拟合
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
from teacher_student.teacher_geometry import create_uv_depth, adaptive_erode_mask_single

def debug_3d_pipeline():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    print("加载深度模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe + DepthPro
    moge_result = moge_loader.infer(img_rgb)
    moge_K = moge_result['intrinsics']
    print(f"MoGe K: fx={moge_K[0,0]:.1f}, fy={moge_K[1,1]:.1f}, cx={moge_K[0,2]:.1f}, cy={moge_K[1,2]:.1f}")
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
    aligned_depth, _ = align_depth_ransac(moge_result['depth'], depthpro_result['depth'])
    
    if aligned_depth.shape != (h, w):
        aligned_depth = cv2.resize(aligned_depth, (w, h))
    
    print(f"对齐后深度: min={aligned_depth.min():.2f}, max={aligned_depth.max():.2f}")
    
    # 估计地面
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h//2:, :] = 1.0
    depth_clip = np.clip(aligned_depth, 0.3, 8.0).astype(np.float64)
    uv_depth = create_uv_depth(depth_clip, mask)
    pc = project_image_to_cam(uv_depth, moge_K)
    ground_equ = extract_ground(pc)
    print(f"地面方程: {ground_equ}")
    
    # 创建可视化图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 原图 + 2D检测框
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    ax.set_title('原图 + 2D检测框')
    
    # 床的2D框 (从Grounding DINO结果)
    bed_box = [200, 50, 400, 200]
    ax.add_patch(plt.Rectangle((bed_box[0], bed_box[1]), bed_box[2]-bed_box[0], bed_box[3]-bed_box[1],
                                fill=False, color='blue', linewidth=2))
    ax.text(bed_box[0], bed_box[1]-5, 'bed', color='blue', fontsize=12)
    
    # 2. 深度图
    ax = axes[0, 1]
    depth_vis = (aligned_depth / aligned_depth.max() * 255).astype(np.uint8)
    ax.imshow(depth_vis, cmap='turbo')
    ax.set_title('对齐后深度图')
    
    # 3. 深度剖面
    ax = axes[0, 2]
    # 沿床的水平线剖面
    row = 125
    depth_row = aligned_depth[row, :]
    ax.plot(depth_row, label=f'Row {row}')
    ax.set_xlabel('X (像素)')
    ax.set_ylabel('深度 (m)')
    ax.set_title('床中心行深度剖面')
    ax.legend()
    ax.grid(True)
    
    # 4. 点云侧视图 (X-Z平面)
    ax = axes[1, 0]
    valid_mask = (pc[:, 2] > 0) & (pc[:, 2] < 10)
    ax.scatter(pc[valid_mask, 0], pc[valid_mask, 2], c=pc[valid_mask, 2], s=0.1, cmap='turbo')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('点云 X-Z 视图 (俯视)')
    ax.set_xlim([-3, 4])
    ax.set_ylim([0, 10])
    ax.invert_yaxis()
    
    # 5. 点云前视图 (X-Y平面)
    ax = axes[1, 1]
    ax.scatter(pc[valid_mask, 0], pc[valid_mask, 1], c=pc[valid_mask, 2], s=0.1, cmap='turbo')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('点云 X-Y 视图 (侧视)')
    ax.set_xlim([-3, 4])
    ax.set_ylim([-2, 2])
    
    # 6. 床掩码区域点云
    ax = axes[1, 2]
    bed_mask = np.zeros((h, w), dtype=np.float32)
    bed_mask[bed_box[1]:bed_box[3], bed_box[0]:bed_box[2]] = 1.0
    
    # 深度过滤
    depth_for_pc = np.clip(aligned_depth, 0.3, 8.0)
    depth_bed = depth_for_pc * bed_mask
    uv_depth_bed = create_uv_depth(depth_bed, bed_mask)
    pc_bed = project_image_to_cam(uv_depth_bed, moge_K)
    
    print(f"\n=== 床区域点云 ===")
    print(f"点数: {pc_bed.shape[0]}")
    print(f"X范围: [{pc_bed[:,0].min():.2f}, {pc_bed[:,0].max():.2f}]")
    print(f"Y范围: [{pc_bed[:,1].min():.2f}, {pc_bed[:,1].max():.2f}]")
    print(f"Z范围: [{pc_bed[:,2].min():.2f}, {pc_bed[:,2].max():.2f}]")
    
    # 计算床的估计尺寸
    dx = pc_bed[:,0].max() - pc_bed[:,0].min()
    dy = pc_bed[:,1].max() - pc_bed[:,1].min()
    dz = pc_bed[:,2].max() - pc_bed[:,2].min()
    print(f"\n从点云直接估计的尺寸:")
    print(f"  X(宽): {dx:.2f}m")
    print(f"  Y(高): {dy:.2f}m")
    print(f"  Z(深): {dz:.2f}m")
    
    # 绘制床点云
    ax.scatter(pc_bed[:, 0], pc_bed[:, 2], c='red', s=1, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('床区域点云 (俯视) - 红色')
    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 6])
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_ROOT}/debug_3d_pipeline.jpg', dpi=150)
    print(f"\n调试图已保存: {PROJECT_ROOT}/debug_3d_pipeline.jpg")
    
    # ========== 详细分析床区域 ==========
    print(f"\n{'='*60}")
    print("床区域详细分析")
    print(f"{'='*60}")
    
    # 床掩码
    bed_mask = np.zeros((h, w), dtype=np.uint8)
    bed_mask[bed_box[1]:bed_box[3], bed_box[0]:bed_box[2]] = 255
    
    # 深度统计
    depth_in_bed = aligned_depth[bed_box[1]:bed_box[3], bed_box[0]:bed_box[2]]
    valid_depth = depth_in_bed[depth_in_bed > 0.1]
    print(f"\n床区域深度统计:")
    print(f"  有效像素: {len(valid_depth)}/{depth_in_bed.size}")
    print(f"  深度: min={valid_depth.min():.2f}, max={valid_depth.max():.2f}")
    print(f"  深度: mean={valid_depth.mean():.2f}, median={np.median(valid_depth):.2f}")
    
    # 深度过滤 (只保留中值附近±1m的点)
    z_med = np.median(valid_depth)
    z_lo, z_hi = z_med - 1.0, z_med + 1.0
    depth_filtered = np.where((depth_in_bed > z_lo) & (depth_in_bed < z_hi), depth_in_bed, 0)
    
    uv_depth_filtered = create_uv_depth(depth_filtered, depth_filtered > 0)
    pc_filtered = project_image_to_cam(uv_depth_filtered, moge_K)
    
    print(f"\n过滤后点云 (深度{z_lo:.1f}-{z_hi:.1f}m):")
    print(f"  点数: {pc_filtered.shape[0]}")
    if pc_filtered.shape[0] > 0:
        dx_f = pc_filtered[:,0].max() - pc_filtered[:,0].min()
        dy_f = pc_filtered[:,1].max() - pc_filtered[:,1].min()
        dz_f = pc_filtered[:,2].max() - pc_filtered[:,2].min()
        print(f"  X范围: [{pc_filtered[:,0].min():.2f}, {pc_filtered[:,0].max():.2f}] = {dx_f:.2f}m")
        print(f"  Y范围: [{pc_filtered[:,1].min():.2f}, {pc_filtered[:,1].max():.2f}] = {dy_f:.2f}m")
        print(f"  Z范围: [{pc_filtered[:,2].min():.2f}, {pc_filtered[:,2].max():.2f}] = {dz_f:.2f}m")
        
        # 估计床中心
        cx = pc_filtered[:,0].mean()
        cy = pc_filtered[:,1].mean()
        cz = pc_filtered[:,2].mean()
        print(f"  估计中心: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        
        # 验证投影
        u_check = moge_K[0,0] * cx / cz + moge_K[0,2]
        v_check = moge_K[1,1] * cy / cz + moge_K[1,2]
        print(f"  投影回像素: ({u_check:.0f}, {v_check:.0f}), 实际: ({(bed_box[0]+bed_box[2])//2}, {(bed_box[1]+bed_box[3])//2})")
    
    # 检查为什么3D框没有框到物体
    print(f"\n{'='*60}")
    print("3D框问题诊断")
    print(f"{'='*60}")
    
    # 从teacher_geometry获取的床结果
    print(f"\n当前估计的床:")
    print(f"  中心: (-0.33, -0.92, 2.97)")
    print(f"  尺寸: (2.07, 1.91, 0.50) [L, W, H]")
    print(f"\n真实床:")
    print(f"  中心应该在: (-0.35, -0.25, ~3.8)")
    print(f"  尺寸约: (2.0, 0.5, 1.5) [L, W, H]")
    print(f"\n问题分析:")
    print(f"  1. 深度偏小: 估计2.97m vs 应该是~3.8m")
    print(f"  2. Y坐标偏小: -0.92 vs -0.25")
    print(f"  3. 这两个问题可能是同一个原因导致的!")

if __name__ == '__main__':
    debug_3d_pipeline()
