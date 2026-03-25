"""
对比分析：原模型 vs 你的实现
重点检查坐标系和投影差异
"""
import cv2
import numpy as np
import sys

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac
from cubercnn.generate_label.util import extract_ground, project_image_to_cam
from teacher_student.teacher_geometry import create_uv_depth

def compare_coordinates():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # MoGe K
    moge_K = np.array([
        [551.2, 0, 365.0],
        [0, 551.2, 265.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 加载深度模型
    print("加载深度模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe + DepthPro
    moge_result = moge_loader.infer(img_rgb)
    moge_K_actual = moge_result['intrinsics']
    print(f"MoGe 实际 K: fx={moge_K_actual[0,0]:.1f}, fy={moge_K_actual[1,1]:.1f}")
    
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K_actual[0, 0])
    aligned_depth, _ = align_depth_ransac(moge_result['depth'], depthpro_result['depth'])
    
    if aligned_depth.shape != (h, w):
        aligned_depth = cv2.resize(aligned_depth, (w, h))
    
    # 估计地面
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h//2:, :] = 1.0
    depth_clip = np.clip(aligned_depth, 0.3, 8.0).astype(np.float64)
    uv_depth = create_uv_depth(depth_clip, mask)
    pc = project_image_to_cam(uv_depth, moge_K_actual)
    ground_equ = extract_ground(pc)
    
    print(f"\n=== 坐标系分析 ===")
    print(f"地面方程: {ground_equ}")
    
    # 地面法向量
    gn = ground_equ[:3]
    gn_norm = gn / np.linalg.norm(gn)
    print(f"地面法向量(归一化): [{gn_norm[0]:.3f}, {gn_norm[1]:.3f}, {gn_norm[2]:.3f}]")
    
    # SUNRGBD 坐标系
    print(f"\nSUNRGBD 坐标系约定:")
    print(f"  X: 水平向右")
    print(f"  Y: 垂直向下 (像素坐标系)")
    print(f"  Z: 深度方向 (从相机到场景)")
    print(f"\n相机坐标系:")
    print(f"  地面方程 Ax+By+Cz+D=0")
    print(f"  对于水平地面: Y = -D (地面是 Y=-D 的平面)")
    
    # 计算地面高度
    if abs(gn_norm[1]) > 0.9:
        ground_y = -ground_equ[3] / gn_norm[1] if gn_norm[1] != 0 else 0
        print(f"  地面高度 Y = {ground_y:.3f}")
    
    # 检查床的位置
    print(f"\n=== 床的位置分析 ===")
    # 床在图像上的位置（从Grounding DINO检测结果）
    # 假设床框: (x1, y1, x2, y2) = (100, 50, 500, 200)
    bed_box = (100, 50, 500, 200)
    x1, y1, x2, y2 = bed_box
    bed_center_u = (x1 + x2) // 2
    bed_center_v = (y1 + y2) // 2
    
    # 从深度图获取床中心深度
    bed_depth = aligned_depth[y1:y2, x1:x2]
    valid_depth = bed_depth[bed_depth > 0.1]
    if len(valid_depth) > 0:
        bed_depth_median = np.median(valid_depth)
        print(f"床区域深度: {bed_depth_median:.2f}m")
    
    # 反投影床中心
    z = bed_depth_median
    x = (bed_center_u - moge_K_actual[0,2]) * z / moge_K_actual[0,0]
    y = (bed_center_v - moge_K_actual[1,2]) * z / moge_K_actual[1,1]
    print(f"床中心3D (相机坐标): ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # 地面以上的高度
    if abs(gn_norm[1]) > 0.9:
        height_above_ground = -y - ground_y
        print(f"床中心在地面以上: {height_above_ground:.2f}m")
        print(f"床高约0.5m，中心应在地面以上0.25m")
        print(f"高度误差: {height_above_ground - 0.25:.2f}m")
    
    # 检查投影
    print(f"\n=== 投影验证 ===")
    u_proj = moge_K_actual[0,0] * x / z + moge_K_actual[0,2]
    v_proj = moge_K_actual[1,1] * y / z + moge_K_actual[1,2]
    print(f"床中心投影回像素: ({u_proj:.1f}, {v_proj:.1f})")
    print(f"实际床中心像素: ({bed_center_u}, {bed_center_v})")
    print(f"投影误差: ({u_proj - bed_center_u:.1f}, {v_proj - bed_center_v:.1f})")
    
    # 框底部位置
    print(f"\n=== 3D框底部位置 ===")
    # 如果床高0.5m，框底部Y应该是 y - 0.25
    box_bottom_y = y - 0.25
    print(f"估计的框底部Y: {box_bottom_y:.2f}")
    print(f"实际床底部应该在哪里?")
    
    # 如果地面在Y=ground_y，框底部应该与地面对齐
    if abs(gn_norm[1]) > 0.9:
        print(f"地面Y: {ground_y:.2f}")
        print(f"框底部与地面的距离: {box_bottom_y - ground_y:.2f}m")
        if box_bottom_y - ground_y < 0.1:
            print("✓ 框底部与地面基本对齐!")
        else:
            print("✗ 框底部与地面有偏差!")

if __name__ == '__main__':
    compare_coordinates()
