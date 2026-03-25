"""
调试3D框投影问题
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

def debug_3d_projection():
    # 加载图像
    image_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 相机内参
    K = np.array([
        [529.5, 0, 365.0],
        [0, 529.5, 262.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 加载深度模型
    print("加载深度模型...")
    moge_loader = MoGeLoader()
    depthpro_loader = DepthProLoader()
    moge_loader.load_model()
    depthpro_loader.load_model()
    
    # MoGe 推理
    moge_result = moge_loader.infer(img_rgb)
    moge_K = moge_result['intrinsics']
    depth_moge = moge_result['depth']
    print(f"MoGe K: {moge_K[0,0]:.1f}")
    
    # DepthPro 推理 - 使用 MoGe 的焦距
    depthpro_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
    depth_depthpro = depthpro_result['depth']
    
    # RANSAC 对齐
    aligned_depth, _ = align_depth_ransac(depth_moge, depth_depthpro)
    
    if aligned_depth.shape != (h, w):
        aligned_depth = cv2.resize(aligned_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    print(f"\n对齐后深度: min={aligned_depth.min():.2f}, max={aligned_depth.max():.2f}, mean={aligned_depth.mean():.2f}")
    
    # 估计地面
    print("\n=== 地面估计 ===")
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h//2:, :] = 1.0
    depth_clip = np.clip(aligned_depth, 0.3, 8.0).astype(np.float64)
    uv_depth = create_uv_depth(depth_clip, mask)
    pc = project_image_to_cam(uv_depth, np.array(K))
    ground_equ = extract_ground(pc)
    print(f"地面方程: {ground_equ}")
    
    # 检查地面法向量
    gn = ground_equ[:3] / np.linalg.norm(ground_equ[:3])
    print(f"地面法向量(归一化): {gn}")
    
    # Y轴方向
    print(f"Y轴(向下): [0, -1, 0]")
    print(f"点积: {np.dot([0, -1, 0], gn):.3f}")
    
    # 床区域测试
    print("\n=== 床区域测试 ===")
    # 床在图像上的位置: x1=200, y1=50, x2=400, y2=200
    bed_x1, bed_y1, bed_x2, bed_y2 = 200, 50, 400, 200
    
    # 取掩码区域深度
    bed_depth = aligned_depth[bed_y1:bed_y2, bed_x1:bed_x2]
    valid_bed = bed_depth[bed_depth > 0.1]
    if len(valid_bed) > 0:
        bed_depth_mean = valid_bed.mean()
        bed_depth_median = np.median(valid_bed)
        print(f"床区域深度: mean={bed_depth_mean:.2f}, median={bed_depth_median:.2f}")
    
    # 计算床中心的3D坐标
    bed_center_u = (bed_x1 + bed_x2) // 2
    bed_center_v = (bed_y1 + bed_y2) // 2
    z = aligned_depth[bed_center_v, bed_center_u]
    x = (bed_center_u - K[0,2]) * z / K[0,0]
    y = (bed_center_v - K[1,2]) * z / K[1,1]
    print(f"床中心像素({bed_center_u}, {bed_center_v})对应的3D点: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # 估计床的尺寸 (根据2D框)
    # 床在图像上的大小约 200x150 像素
    bed_width_px = bed_x2 - bed_x1  # 200 px
    bed_height_px = bed_y2 - bed_y1  # 150 px
    
    # 在深度 z 处，1像素对应的实际尺寸
    meter_per_px_x = z / K[0,0]  # 每像素在X方向的距离
    meter_per_px_y = z / K[1,1]  # 每像素在Y方向的距离
    
    bed_width_est = bed_width_px * meter_per_px_x
    bed_height_est = bed_height_px * meter_per_px_y
    
    print(f"\n床尺寸估计 (基于2D框+深度):")
    print(f"  每像素在X: {meter_per_px_x:.3f}m")
    print(f"  每像素在Y: {meter_per_px_y:.3f}m")
    print(f"  宽度估计: {bed_width_est:.2f}m")
    print(f"  高度估计: {bed_height_est:.2f}m")
    print(f"  真实床尺寸约: 2.0m x 1.5m x 0.5m")
    
    # 检查SUNRGBD的GT
    print("\n=== SUNRGBD GT (如果有) ===")
    import pickle
    info_path = f'{PROJECT_ROOT}/datasets/sunrgbd/sunrgbd_infos_train.pkl'
    try:
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        for item in infos['data_list']:
            img_info = item.get('images', {})
            cam0 = img_info.get('CAM0', {})
            if '000004' in cam0.get('img_path', ''):
                instances = item.get('instances', [])
                for inst in instances:
                    bbox_3d = inst.get('bbox_3d', [])
                    label = inst.get('bbox_label', -1)
                    if label == 3 and len(bbox_3d) >= 7:  # 床的标签通常是3
                        cx, cy, cz = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                        l, w, h = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                        yaw = bbox_3d[6]
                        print(f"GT床: center=({cx:.2f}, {cy:.2f}, {cz:.2f}), dims=({l:.2f}, {w:.2f}, {h:.2f})")
                break
    except Exception as e:
        print(f"无法读取GT: {e}")

if __name__ == '__main__':
    debug_3d_projection()
