import numpy as np
import os
import glob

# ==========================================
# 📐 Step 3: L-Shape 几何拟合与截断核心算法
# ==========================================
class LShape3DBBoxFitter:
    def __init__(self, angle_resolution=1.0, noise_percentile=5.0):
        """
        :param angle_resolution: 搜索角度的步长 (度)，默认 1度，越小越准但越慢。
        :param noise_percentile: 用于截断深度拖尾噪点的百分比。5.0 表示砍掉最外围 5% 的脏点。
        """
        self.angle_res = angle_resolution
        self.p_low = noise_percentile
        self.p_high = 100.0 - noise_percentile

    def fit(self, points):
        """
        核心拟合函数
        :param points: 提取出的 3D 伪点云, Shape: [N, 3]。
                       注意：这里假设使用的是标准相机坐标系 (X向右，Y向下，Z向前)
        :return: bbox_dict 包含 3D 框的中心点、长宽高和朝向角
        """
        if len(points) < 10:
            return None # 点太少，无法拟合

        # ---------------------------------------------------------
        # 1. 高度轴 (Y轴) 独立截断处理
        # 相机坐标系下，Y 轴代表物体的上下高度。我们用百分位数滤除天花板和地面的噪点。
        # ---------------------------------------------------------
        y_points = points[:, 1]
        y_min, y_max = np.percentile(y_points, [self.p_low, self.p_high])
        h = y_max - y_min
        center_y = (y_max + y_min) / 2.0

        # ---------------------------------------------------------
        # 2. 鸟瞰图 (BEV) 投影：提取 X-Z 平面
        # ---------------------------------------------------------
        bev_points = points[:, [0, 2]]  # 取出 X 和 Z 坐标，Shape: [N, 2]
        
        # ---------------------------------------------------------
        # 3. L-Shape 核心暴搜寻角 (0 ~ 90 度)
        # 为什么是 0~90度？因为矩形有 90 度对称性。
        # 我们寻找那个能让点云被“最小面积的矩形”严丝合缝包裹住的旋转角度。
        # ---------------------------------------------------------
        min_area = float('inf')
        best_yaw = 0.0
        best_box_2d = None

        # 遍历所有可能的朝向角
        angles = np.arange(0, 90, self.angle_res)
        for angle in angles:
            theta = np.deg2rad(angle)
            # 构造 2D 旋转矩阵 (顺时针旋转，相当于把坐标系转过去)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))

            # 将 BEV 点云旋转到当前测试的局部坐标系
            rotated_points = np.dot(bev_points, R.T)

            # 在当前角度下，使用百分位数截断长宽两端的噪点（专治单目深度拖尾）
            u_min, u_max = np.percentile(rotated_points[:, 0], [self.p_low, self.p_high])
            v_min, v_max = np.percentile(rotated_points[:, 1], [self.p_low, self.p_high])

            # 计算当前角度下，包裹点云所需的矩形面积
            area = (u_max - u_min) * (v_max - v_min)

            # 如果面积更小，说明贴合得更紧密，记录为最佳参数
            if area < min_area:
                min_area = area
                best_yaw = theta
                best_box_2d = (u_min, u_max, v_min, v_max)

        # ---------------------------------------------------------
        # 4. 逆向重构 3D 物理边界框
        # ---------------------------------------------------------
        u_min, u_max, v_min, v_max = best_box_2d
        
        # 计算物体的长度和宽度
        l = u_max - u_min  # 沿局部 U 轴的长度
        w = v_max - v_min  # 沿局部 V 轴的宽度

        # 计算局部坐标系下的中心点
        center_u = (u_max + u_min) / 2.0
        center_v = (v_max + v_min) / 2.0

        # 将中心点逆旋转回原始的世界/相机 X-Z 坐标系
        c_inv, s_inv = np.cos(-best_yaw), np.sin(-best_yaw)
        R_inv = np.array(((c_inv, -s_inv), (s_inv, c_inv)))
        
        center_bev = np.dot(np.array([center_u, center_v]), R_inv.T)
        center_x, center_z = center_bev[0], center_bev[1]

        # 打包返回最终的 7自由度 3D 边界框
        return {
            "X": center_x, "Y": center_y, "Z": center_z,
            "L": l, "W": w, "H": h,
            "Yaw": best_yaw  # 弧度制
        }

# ==========================================
# 🧪 测试入口：批量处理 Step 1 生成的点云
# ==========================================
if __name__ == "__main__":
    # 指向你刚才 Step 1 生成点云的文件夹
    PTS_DIR = "/data/ZhaoX/OVM3D-Det-1/detany3d_pts"
    
    print("========== 启动 Step 3 L-Shape 几何测量仪 ==========")
    fitter = LShape3DBBoxFitter(angle_resolution=1.0, noise_percentile=5.0)
    
    # 查找所有 .npy 文件
    npy_files = glob.glob(os.path.join(PTS_DIR, "*.npy"))
    
    if len(npy_files) == 0:
        print(f"⚠️ 在 {PTS_DIR} 中没有找到点云文件，请确保 Step 1 已成功运行！")
    else:
        for file_path in npy_files:
            obj_name = os.path.basename(file_path)
            
            # 加载点云
            points = np.load(file_path)
            
            # 核心拟合！
            bbox = fitter.fit(points)
            
            if bbox is not None:
                print(f"\n✅ 目标 [{obj_name}] 拟合成功 (过滤了 {len(points)} 个点中的噪点):")
                print(f"   📍 中心坐标 (X, Y, Z) : ({bbox['X']:.2f}m, {bbox['Y']:.2f}m, {bbox['Z']:.2f}m)")
                print(f"   📏 物理尺寸 (L, W, H) : ({bbox['L']:.2f}m, {bbox['W']:.2f}m, {bbox['H']:.2f}m)")
                print(f"   🧭 完美朝向角 (Yaw)   : {np.rad2deg(bbox['Yaw']):.1f}°")
            else:
                print(f"⚠️ 目标 [{obj_name}] 点云数量不足，被丢弃。")