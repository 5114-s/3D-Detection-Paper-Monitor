"""
用正确的床框调试3D框拟合问题
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

def debug_correct_bed():
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
    
    # 正确的床框 (从Grounding DINO检测结果)
    bed_box = [117, 98, 551, 451]  # 正确的xyxy格式
    x1, y1, x2, y2 = bed_box
    bed_center_u = (x1 + x2) // 2
    bed_center_v = (y1 + y2) // 2
    
    print(f"\n正确的床框: ({x1}, {y1}, {x2}, {y2}), 中心: ({bed_center_u}, {bed_center_v})")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 原图 + 正确床框
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue', linewidth=2))
    ax.plot(bed_center_u, bed_center_v, 'r+', markersize=20)
    ax.set_title('Correct Bed Box (117, 98, 551, 451)')
    
    # 2. 深度图
    ax = axes[0, 1]
    depth_vis = (aligned_depth / aligned_depth.max() * 255).astype(np.uint8)
    ax.imshow(depth_vis, cmap='turbo')
    ax.set_title('Aligned Depth Map')
    
    # 3. 床区域深度
    ax = axes[0, 2]
    bed_depth_region = aligned_depth[y1:y2, x1:x2]
    ax.imshow(bed_depth_region, cmap='turbo')
    ax.set_title('Depth in Bed Region')
    
    # 4. 床掩码点云 (俯视)
    ax = axes[1, 0]
    bed_mask = np.zeros((h, w), dtype=np.float32)
    bed_mask[y1:y2, x1:x2] = 1.0
    
    depth_for_pc = np.clip(aligned_depth, 0.3, 8.0)
    depth_bed = depth_for_pc * bed_mask
    uv_depth_bed = create_uv_depth(depth_bed, bed_mask)
    pc_bed = project_image_to_cam(uv_depth_bed, moge_K)
    
    print(f"\n=== 正确的床区域点云 ===")
    print(f"点数: {pc_bed.shape[0]}")
    print(f"X范围: [{pc_bed[:,0].min():.2f}, {pc_bed[:,0].max():.2f}]")
    print(f"Y范围: [{pc_bed[:,1].min():.2f}, {pc_bed[:,1].max():.2f}]")
    print(f"Z范围: [{pc_bed[:,2].min():.2f}, {pc_bed[:,2].max():.2f}]")
    
    # 绘制点云
    valid = pc_bed[:, 2] > 0
    ax.scatter(pc_bed[valid, 0], pc_bed[valid, 2], c=pc_bed[valid, 2], s=0.5, cmap='turbo', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Bed Point Cloud - Top View')
    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 6])
    ax.invert_yaxis()
    
    # 5. 床点云 (侧视图)
    ax = axes[1, 1]
    ax.scatter(pc_bed[valid, 0], pc_bed[valid, 1], c=pc_bed[valid, 2], s=0.5, cmap='turbo', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Bed Point Cloud - Side View')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 1])
    
    # 6. 深度剖面
    ax = axes[1, 2]
    row = bed_center_v
    depth_row = aligned_depth[row, :]
    ax.plot(depth_row)
    ax.axvline(x=x1, color='r', label='bed x1')
    ax.axvline(x=x2, color='r', label='bed x2')
    ax.axvline(x=bed_center_u, color='g', label='bed center')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Depth Profile at Row {row}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_ROOT}/debug_correct_bed.jpg', dpi=150)
    print(f"\n调试图已保存: {PROJECT_ROOT}/debug_correct_bed.jpg")
    
    # 详细分析
    print(f"\n{'='*60}")
    print("正确床区域深度分析")
    print(f"{'='*60}")
    
    depth_in_bed = aligned_depth[y1:y2, x1:x2]
    valid_depth = depth_in_bed[depth_in_bed > 0.1]
    print(f"床区域深度统计:")
    print(f"  有效像素: {len(valid_depth)}/{depth_in_bed.size}")
    print(f"  深度: min={valid_depth.min():.2f}, max={valid_depth.max():.2f}")
    print(f"  深度: mean={valid_depth.mean():.2f}, median={np.median(valid_depth):.2f}")
    
    # 计算床中心点的3D位置
    z = np.median(valid_depth)
    x = (bed_center_u - moge_K[0,2]) * z / moge_K[0,0]
    y = (bed_center_v - moge_K[1,2]) * z / moge_K[1,1]
    print(f"\n床中心点的3D位置 (基于中位深度 {z:.2f}m):")
    print(f"  X = ({bed_center_u} - {moge_K[0,2]}) * {z} / {moge_K[0,0]} = {x:.2f}")
    print(f"  Y = ({bed_center_v} - {moge_K[1,2]}) * {z} / {moge_K[1,1]} = {y:.2f}")
    print(f"  Z = {z:.2f}")
    
    # 估计尺寸
    # 在深度 z 处，1像素对应的实际尺寸
    meter_per_px_x = z / moge_K[0,0]
    meter_per_px_y = z / moge_K[1,1]
    
    bed_width_est = (x2 - x1) * meter_per_px_x
    bed_height_est = (y2 - y1) * meter_per_px_y
    
    print(f"\n基于2D框+深度的尺寸估计:")
    print(f"  每像素在X: {meter_per_px_x:.4f}m")
    print(f"  每像素在Y: {meter_per_px_y:.4f}m")
    print(f"  床宽度(图像X方向): {(x2-x1)}px -> {bed_width_est:.2f}m")
    print(f"  床高度(图像Y方向): {(y2-y1)}px -> {bed_height_est:.2f}m")
    print(f"  真实床尺寸约: 2.0m x 1.5m x 0.5m")
    
    # 检查投影是否正确
    print(f"\n投影验证:")
    u_check = moge_K[0,0] * x / z + moge_K[0,2]
    v_check = moge_K[1,1] * y / z + moge_K[1,2]
    print(f"  床中心反投影: ({u_check:.0f}, {v_check:.0f})")
    print(f"  实际床中心: ({bed_center_u}, {bed_center_v})")
    print(f"  误差: ({u_check - bed_center_u:.1f}, {v_check - bed_center_v:.1f})")

if __name__ == '__main__':
    debug_correct_bed()
