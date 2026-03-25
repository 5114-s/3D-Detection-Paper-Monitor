import numpy as np
import open3d as o3d
import cv2
import os

# ==========================================
# 🌟 核心函数 1：OVM3D-Det 自适应掩码腐蚀
# ==========================================
def adaptive_erode_mask(mask, max_k_y, min_k_y, max_k_x, min_k_x):
    """
    根据物体 2D Bounding Box 的大小，动态计算腐蚀核。
    防止远处的极小物体被完全抹除，同时保证近处的特大物体能被有效切除拖尾。
    """
    # ⚠️ 关键修复：将布尔值转换为 0/255，防止 OpenCV 腐蚀运算出错
    mask_uint8 = (mask.astype(np.uint8)) * 255 
    
    y_indices, x_indices = np.where(mask_uint8 > 0)
    if len(y_indices) == 0:
        return mask # 空掩码直接返回
        
    obj_h = y_indices.max() - y_indices.min() + 1
    obj_w = x_indices.max() - x_indices.min() + 1
    
    # 根据物体大小的 5% 动态计算核尺寸
    scale_factor = 0.05 
    k_y = int(np.clip(obj_h * scale_factor, min_k_y, max_k_y))
    k_x = int(np.clip(obj_w * scale_factor, min_k_x, max_k_x))
    
    print(f"    [自适应外科手术] 2D尺寸: {obj_w}x{obj_h} -> 动态腐蚀核: X={k_x}, Y={k_y}")
    
    kernel = np.ones((k_y, k_x), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
    
    return eroded_mask > 0

# ==========================================
# 🌟 核心函数 2：动态获取 SUN RGB-D 相机内参
# ==========================================
def get_sunrgbd_intrinsic(calib_path):
    fallback = [529.50, 529.50, 365.00, 265.00] 
    if not os.path.exists(calib_path): 
        return fallback
    try:
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('K:'):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    return [vals[0], vals[4], vals[2], vals[5]]
    except: 
        pass
    return fallback

# ==========================================
# 主程序开始
# ==========================================
print("正在加载深度图和结构化实例数据...")
depth_map = np.load("output_depth_sunrgbd.npy")  
instances = np.load("Grounded-SAM-2/test_instances.npy", allow_pickle=True)

calib_path = "test.txt"
fx, fy, cx, cy = get_sunrgbd_intrinsic(calib_path)
print(f"当前相机的真实内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

all_clean_pcds = []
print(f"检测到画面中有 {len(instances)} 个目标，开始批量自适应升维...")

for i, obj in enumerate(instances):
    label = obj['label']
    mask = obj['mask']
    
    print(f"\n---> 正在处理第 {i+1} 个目标: [{label}]")
    
    # ==========================================
    # ⚡ 升级点：执行自适应掩码腐蚀 (替换了固定 5x5)
    # 使用室内场景推荐参数：(12, 2, 6, 2)
    # ==========================================
    mask_clean = adaptive_erode_mask(mask, max_k_y=12, min_k_y=2, max_k_x=6, min_k_x=2)
    
    v, u = np.where(mask_clean == True)
    z = depth_map[v, u]
    
    valid_mask = z > 0
    u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
    
    if len(z) < 50:
        print(f"    ⚠️ 腐蚀后剩余点数太少({len(z)}点)，跳过该目标！")
        continue

    # 核心物理公式：计算真实 3D 坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 绝对对齐机器视觉原始坐标系 (x, y, z)，无负号翻转！
    points_3d = np.stack((x, y, z), axis=-1) 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # 统一涂成纯红色
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # 彻底告别 SOR 洗点，直接信任自适应腐蚀的结果！
    print(f"    ✅ 升维成功！生成 {len(pcd.points)} 个自适应纯净点。")
    
    save_name = f"instance_{i}_{label}_adaptive_clean.ply"
    o3d.io.write_point_cloud(save_name, pcd)
    
    all_clean_pcds.append(pcd)

# ==========================================
# 拼接与保存全局点云
# ==========================================
print("\n🎉 所有目标处理完毕！正在拼接全局点云...")

if len(all_clean_pcds) > 0:
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in all_clean_pcds:
        merged_pcd += pcd

    scene_save_path = "scene_all_objects_adaptive.ply"
    o3d.io.write_point_cloud(scene_save_path, merged_pcd)
    print(f"✅ 完美！基于自适应腐蚀的整个房间 3D 伪点云已保存为: {scene_save_path}")
else:
    print("⚠️ 没有生成任何有效的点云。")

print("=====================================================")
print(f"👉 请使用 SCP/Xftp 将 {scene_save_path} 下载到本地，用 MeshLab 查看原生物理坐标系！")
print("=====================================================")