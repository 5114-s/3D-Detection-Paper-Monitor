import open3d as o3d
import numpy as np
import os
import glob

def batch_clean_and_export(input_dir, output_dir_npy, output_dir_ply):
    # 创建输出文件夹
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_ply, exist_ok=True)

    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        print(f"⚠️ 找不到任何点云文件，请检查目录: {input_dir}")
        return

    print(f"🔍 找到 {len(npy_files)} 个目标点云，开始批量 SOR 去噪...")

    for npy_path in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # 1. 加载 numpy 点云
        points = np.load(npy_path)
        
        # 过滤掉点数过少的废框
        if len(points) < 30:
            print(f"  ⏭️ 跳过 {base_name}: 点云数量极少 ({len(points)} 个)")
            continue

        # 2. 转换为 Open3D 格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 3. 执行 SOR 3D 空间去噪
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        
        # 获取去噪后的 numpy 数组
        clean_points = np.asarray(pcd_clean.points)

        # 4. 分别保存为算法所需的 .npy 和可视化的 .ply
        save_npy_path = os.path.join(output_dir_npy, f"{base_name}_clean.npy")
        save_ply_path = os.path.join(output_dir_ply, f"{base_name}_clean.ply")
        
        # 存 NPY 给 Step 3
        np.save(save_npy_path, clean_points)
        
        # 上色并存 PLY 给 MeshLab
        pcd_clean.paint_uniform_color([0.1, 0.7, 0.7]) 
        o3d.io.write_point_cloud(save_ply_path, pcd_clean)

        print(f"  ✅ {base_name}: 原始 {len(points)}点 -> 去噪后 {len(clean_points)}点")

    print(f"\n🎉 批量去噪完成！")
    print(f"算法文件已存入: {output_dir_npy}")
    print(f"可视化文件已存入: {output_dir_ply}")

if __name__ == "__main__":
    # 替换为你 Step 1 的输出目录
    INPUT_DIR = "/data/ZhaoX/OVM3D-Det-1/detany3d_pts"
    
    # 我们把去噪后的 npy 和 ply 分开存放，保持整洁
    OUTPUT_NPY_DIR = "/data/ZhaoX/OVM3D-Det-1/detany3d_pts_clean_npy"
    OUTPUT_PLY_DIR = "/data/ZhaoX/OVM3D-Det-1/detany3d_pts_clean_ply"
    
    batch_clean_and_export(INPUT_DIR, OUTPUT_NPY_DIR, OUTPUT_PLY_DIR)