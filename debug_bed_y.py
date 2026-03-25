"""
详细检查床的Y坐标问题
"""
import cv2
import numpy as np

# 加载图像
image_path = '/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
img = cv2.imread(image_path)
h, w = img.shape[:2]

# MoGe K (从之前的调试得知)
moge_K = np.array([
    [551.2, 0, 365.0],
    [0, 551.2, 265.0],
    [0, 0, 1]
], dtype=np.float32)

# 床的2D框
bed_x1, bed_y1, bed_x2, bed_y2 = 200, 50, 400, 200
bed_center_u, bed_center_v = 300, 125

print("=== 床的位置分析 ===")
print(f"图像尺寸: {w}x{h}")
print(f"床2D框: ({bed_x1}, {bed_y1}) -> ({bed_x2}, {bed_y2})")
print(f"床中心像素: ({bed_center_u}, {bed_center_v})")

# 使用估计的深度 2.97m 反推
depth = 2.97
x = (bed_center_u - moge_K[0,2]) * depth / moge_K[0,0]
y = (bed_center_v - moge_K[1,2]) * depth / moge_K[1,1]
z = depth

print(f"\n使用深度 z={z}m 反推床中心的3D位置:")
print(f"  x = ({bed_center_u} - {moge_K[0,2]}) * {z} / {moge_K[0,0]} = {x:.3f}")
print(f"  y = ({bed_center_v} - {moge_K[1,2]}) * {z} / {moge_K[1,1]} = {y:.3f}")
print(f"  z = {z}")
print(f"  床中心3D: ({x:.3f}, {y:.3f}, {z:.3f})")

print(f"\n=== SUNRGBD 坐标系分析 ===")
print("SUNRGBD 坐标系:")
print("  X: 水平向右")
print("  Y: 垂直向下 (Y=0 是地平线, Y>0 是地面以下!)")
print("  Z: 深度方向（从相机到场景）")
print()
print("由于相机俯视，地面上的点 Y < 0")
print("床在地面上，床底部 Y ≈ 0")
print("床中心应该在 Y ≈ -0.25m (床高0.5m的一半)")
print()
print(f"但我们计算得到 Y = {y:.3f}m")
print(f"这意味着: 深度 {z}m 可能偏小，或者 2D 框位置有问题")

# 检查：如果床中心Y应该是 -0.25m，深度应该是多少？
print(f"\n=== 验证 ===")
print(f"如果床中心 Y 应该是 -0.25m:")
corrected_depth = -moge_K[1,1] * 0.25 / y * (-y)  # 简化计算
corrected_depth = moge_K[1,1] * 0.25 / abs(y) * abs(y) / y * (-y)
# 实际上: y = (v - cy) * z / fy => z = y * fy / (v - cy)
if bed_center_v - moge_K[1,2] != 0:
    corrected_z = y * moge_K[1,1] / (bed_center_v - moge_K[1,2])
    print(f"  需要的深度 z = {corrected_z:.2f}m")
    print(f"  当前深度 z = {z:.2f}m")
    print(f"  深度误差: {corrected_z - z:.2f}m")

# 检查SUNRGBD原始GT
print(f"\n=== 检查SUNRGBD GT (如果有) ===")
try:
    import pickle
    info_path = '/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_infos_train.pkl'
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    for item in infos['data_list']:
        img_info = item.get('images', {})
        cam0 = img_info.get('CAM0', {})
        if '000004' in cam0.get('img_path', ''):
            instances = item.get('instances', [])
            print(f"图像: {cam0.get('img_path')}")
            print(f"尺寸: {cam0.get('height')}x{cam0.get('width')}")
            print(f"\n实例数量: {len(instances)}")
            for i, inst in enumerate(instances):
                bbox_3d = inst.get('bbox_3d', [])
                label = inst.get('bbox_label', -1)
                label_name = {3: 'bed', 0: 'table', 1: 'chair'}.get(label, f'label_{label}')
                if len(bbox_3d) >= 7:
                    cx, cy, cz = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                    l, w_dim, h_dim = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                    print(f"  {i}: {label_name} center=({cx:.2f}, {cy:.2f}, {cz:.2f}), dims=({l:.2f}, {w_dim:.2f}, {h_dim:.2f})")
            break
except Exception as e:
    print(f"无法读取GT: {e}")
