import numpy as np
import open3d as o3d
import cv2
print("正在加载深度图和掩码数据...")
# 1. 加载我们辛苦得来的“无暇深度图”和“行人 Mask”
# 请确保这两个文件在你的当前目录下
depth_map = np.load("output_depth_flawless.npy")  # (376, 1232)
mask = np.load("Grounded-SAM-2/test_mask.npy")                   # (376, 1232) 布尔型


# === 新增的外科手术代码 ===
# 建立一个 5x5 的腐蚀核，把 Mask 的边缘向内剔除 2 个像素
kernel = np.ones((5, 5), np.uint8)
mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
mask = mask_eroded.astype(bool) # 把腐蚀后的 Mask 替换回去
# =========================

# 2. 注入 KITTI 左侧相机真实内参 [fx, fy, cx, cy]
fx, fy = 718.856, 718.856
cx, cy = 607.1928, 185.2157

print("正在进行针孔相机物理反投影...")
# 3. 提取 Mask 覆盖区域的像素坐标
# v 是图像的行(y)，u 是图像的列(x)
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
print(f"反投影完成！成功生成 {points_3d.shape[0]} 个原始 3D 坐标点。")

# ==========================================
# 5. Open3D 实例化与可视化
# ==========================================
# 将 numpy 数组转为 Open3D 的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# 为了好看，我们给点云涂上统一的红色
pcd.paint_uniform_color([1, 0.0, 0.0])

# ==========================================
# 6. 核心提升：SOR (统计离群点去除)
# ==========================================
print("正在执行 SOR 几何除杂 (Statistical Outlier Removal)...")
# nb_neighbors: 算多少个邻居，std_ratio: 标准差阈值（越小过滤越狠）
clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 提取被过滤掉的“飞线”噪点（画成灰色，用于对比展示）
outlier_pcd = pcd.select_by_index(ind, invert=True)
outlier_pcd.paint_uniform_color([0.5, 0.5, 0.5])

print(f"SOR 清洗完毕！剩余 {len(clean_pcd.points)} 个高质量实例点。")

# 7. 存盘，留给 MonoDiff 训练用！
o3d.io.write_point_cloud("person_clean.ply", clean_pcd)
print("极其干净的 3D 点云已保存为 person_clean.ply！")

# 8. 见证奇迹的时刻：弹出 3D 交互窗口
print("正在打开 3D 视窗，你可以用鼠标拖拽、旋转、缩放查看你的 3D 行人！")
print("红色的是干净的点云，灰色的是被 SOR 剔除的飞线噪点。")
o3d.visualization.draw_geometries([clean_pcd, outlier_pcd], window_name="Diffu-OVM3D: Clean Instance Point Cloud")