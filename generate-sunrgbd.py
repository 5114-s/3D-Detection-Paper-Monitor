import numpy as np
import open3d as o3d
import cv2
import os

# ==========================================
# 1. 动态获取 SUN RGB-D 相机内参的函数
# ==========================================
def get_sunrgbd_intrinsic(calib_path):
    """精准提取 SUN RGB-D 当前照片的专属物理相机参数"""
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

print("正在加载深度图和结构化实例数据...")
# ==========================================
# 2. 读取对应的数据 (请确保文件名与之前生成的对应)
# ==========================================
depth_map = np.load("output_depth_sunrgbd.npy")  
instances = np.load("Grounded-SAM-2/test_instances.npy", allow_pickle=True)

# 读取这张图对应的真实内参
calib_path = "test.txt"
fx, fy, cx, cy = get_sunrgbd_intrinsic(calib_path)
print(f"当前相机的真实内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

# 准备一个列表，用来放所有清洗好的点云
all_clean_pcds = []

# ==========================================
# 3. 遍历图片里的每一个目标，流水线作业！
# ==========================================
print(f"检测到画面中有 {len(instances)} 个目标，开始批量升维...")

for i, obj in enumerate(instances):
    label = obj['label']
    mask = obj['mask']
    
    print(f"\n---> 正在处理第 {i+1} 个目标: [{label}]")
    
    # --- 外科手术：5x5 腐蚀 ---
    # 乘以 255 是为了让 OpenCV 更稳定地处理二值图像
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    mask_clean = mask_eroded > 0
    
    # --- 提取坐标与深度 ---
    v, u = np.where(mask_clean == True)
    z = depth_map[v, u]
    
    # 过滤无效深度
    valid_mask = z > 0
    u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
    
    if len(z) < 50:
        print(f"⚠️ [{label}] 腐蚀后剩余点数太少({len(z)}点)，跳过该目标！")
        continue

    # --- 核心物理公式：计算真实 3D 坐标 ---
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 🌟 关键修复 1：绝对对齐机器视觉原始坐标系 (x, y, z)，去掉负号！
    points_3d = np.stack((x, y, z), axis=-1) 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # 🌟 关键修复 2：统一涂成纯红色，对齐批量处理的外观！
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # --- 核心提升：SOR (统计离群点去除) ---
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(clean_pcd.points) < 20:
        print(f"⚠️ [{label}] SOR 清洗后几乎不剩点，跳过！")
        continue
        
    print(f"✅ [{label}] 升维成功！生成 {len(clean_pcd.points)} 个高质量实例点。")
    
    # --- 存盘个体 (可选，方便单独查看某个物体) ---
    save_name = f"instance_{i}_{label}_clean.ply"
    o3d.io.write_point_cloud(save_name, clean_pcd)
    
    all_clean_pcds.append(clean_pcd)

# ==========================================
# 4. 将所有物体合并为一个完整的 3D 伪点云场景
# ==========================================
print("\n🎉 所有目标处理完毕！正在拼接全局点云...")

if len(all_clean_pcds) > 0:
    # 创建一个空的点云对象作为“大容器”
    merged_pcd = o3d.geometry.PointCloud()
    
    # 将列表里的所有独立点云全部加进去
    for pcd in all_clean_pcds:
        merged_pcd += pcd

    # 保存为一个总的场景文件
    scene_save_path = "scene_all_objects.ply"
    o3d.io.write_point_cloud(scene_save_path, merged_pcd)
    print(f"✅ 完美！整个房间的 3D 伪点云已合并并保存为: {scene_save_path}")
else:
    print("⚠️ 没有生成任何有效的点云。")

# 优雅地处理服务器无界面的情况，不再强制弹出窗口
print("=====================================================")
print("👉 因为当前是在无图形界面的服务器上运行，无法直接显示。")
print(f"👉 请使用 SCP/Xftp 将服务器上的 {scene_save_path} 下载到本地。")
print("👉 推荐使用开源软件 MeshLab 或 CloudCompare 直接打开查看！")
print("=====================================================")