import numpy as np
import open3d as o3d
import cv2

# ==========================================
# 🌟 核心函数：OVM3D-Det 原版自适应掩码腐蚀
# ==========================================
def adaptive_erode_mask(mask, max_k_y, min_k_y, max_k_x, min_k_x):
    """
    根据物体 2D Bounding Box 的大小，动态计算腐蚀核。
    防止远处的极小物体被完全抹除，同时保证近处的特大物体能被有效切除拖尾。
    """
    mask_uint8 = mask.astype(np.uint8)
    
    # 找到物体的有效像素边界
    y_indices, x_indices = np.where(mask_uint8 > 0)
    if len(y_indices) == 0:
        return mask_uint8 # 空掩码直接返回
        
    obj_h = y_indices.max() - y_indices.min() + 1
    obj_w = x_indices.max() - x_indices.min() + 1
    
    # 根据物体大小的 5% 动态计算核尺寸
    scale_factor = 0.05 
    
    k_y = int(np.clip(obj_h * scale_factor, min_k_y, max_k_y))
    k_x = int(np.clip(obj_w * scale_factor, min_k_x, max_k_x))
    
    print(f"  -> 当前物体尺寸: {obj_w}x{obj_h}, 动态生成的腐蚀核大小: X={k_x}, Y={k_y}")
    
    # 构造非对称矩形腐蚀核并执行腐蚀
    kernel = np.ones((k_y, k_x), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
    
    return eroded_mask

# ==========================================
# 主程序开始
# ==========================================
print("正在加载深度图和掩码数据...")
# 1. 加载深度图和原始掩码
depth_map = np.load("output_depth_flawless.npy")  
mask = np.load("Grounded-SAM-2/test_mask.npy")

# ==========================================
# === 外科手术：自适应 2D 掩码边缘腐蚀 ===
# ==========================================
print("正在执行 OVM3D-Det 自适应 Mask 边缘腐蚀...")
# 这里的参数 (4, 2, 4, 2) 是清华原版代码里用于室外(Outdoor)行人/车辆的参数。
# 如果你的图片是室内(Indoor)拥挤场景，请改成 (12, 2, 6, 2)！
mask_eroded = adaptive_erode_mask(mask, max_k_y=4, min_k_y=2, max_k_x=4, min_k_x=2)

# 把瘦身后的 Mask 替换回去
mask = mask_eroded.astype(bool) 

# 2. 注入 KITTI 左侧相机真实内参 [fx, fy, cx, cy]
fx, fy = 718.856, 718.856
cx, cy = 607.1928, 185.2157

print("正在进行针孔相机物理反投影...")
# 3. 提取 Mask 覆盖区域的像素坐标
v, u = np.where(mask == True)

# 提取这些像素点对应的绝对物理深度 (Z)
z = depth_map[v, u]

# 过滤掉无效深度（比如深度 <= 0 的点）
valid_mask = z > 0
u = u[valid_mask]
v = v[valid_mask]
z = z[valid_mask]

# 4. 核心物理公式：计算真实 3D 坐标 (X, Y, Z)
x = (u - cx) * z / fx
y = (v - cy) * z / fy

# 组合成 (N, 3) 的点云矩阵
points_3d = np.stack((x, y, z), axis=-1)
print(f"反投影完成！成功生成 {points_3d.shape[0]} 个 3D 坐标点。")

# ==========================================
# 5. Open3D 实例化与可视化 (彻底告别 SOR)
# ==========================================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.paint_uniform_color([1, 0.0, 0.0]) # 统一涂成红色

# 6. 直接存盘：因为已经经过了极其精准的自适应腐蚀，我们相信剩下的点云足以让网络学习
o3d.io.write_point_cloud("person_eroded_only.ply", pcd)
print("去除了严重边缘拖尾的极净 3D 点云已保存为 person_eroded_only.ply！")

# 7. 视窗展示代码
try:
    o3d.visualization.draw_geometries([pcd], window_name="Diffu-OVM3D: Adaptive Eroded Point Cloud")
except Exception as e:
    print("服务器无图形界面，请将 person_eroded_only.ply 下载到本地用 MeshLab 查看！")