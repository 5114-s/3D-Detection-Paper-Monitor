import numpy as np
import open3d as o3d

# 读取npy点云
points = np.load("enhanced_pseudo_lidar.npy")

# 创建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 保存为ply
o3d.io.write_point_cloud("pseudo_lidar.ply", pcd)

print("转换完成: pseudo_lidar.ply")