"""
分析SUNRGBD相机参数
"""
import numpy as np

def analyze_camera():
    print("=== SUNRGBD 相机分析 ===")
    
    # 从depth2img提取参数
    # depth2img 是 3x3 矩阵，不是 3x4
    # 这意味着SUNRGBD可能已经包含了投影
    
    # 样本数据
    depth2img = np.array([
        [536.1855101361871, 349.31819581612945, -63.87781824171543],
        [116.1350419446826, -7.52736322581768, -580.5604480206966],
        [0.047954000532627106, 0.887470006942749, -0.45837000012397766]
    ])
    
    print("depth2img (3x3):")
    print(depth2img)
    
    # 分析SUNRGBD坐标系
    # SUNRGBD使用不同的相机模型
    # 
    # 标准针孔相机:
    # u = fx * (Xc / Zc) + cx
    # v = fy * (Yc / Zc) + cy
    #
    # SUNRGBD坐标系:
    # - X轴朝右
    # - Y轴朝下
    # - Z轴朝前(从相机到物体)
    #
    # 但depth2img是直接从3D(X,Y,Z)投影到(u,v)
    # 所以它可能包含了旋转
    
    print("\n=== 提取焦距 ===")
    
    # 从depth2img的列向量范数估计焦距
    col1_norm = np.linalg.norm(depth2img[:, 0])
    col2_norm = np.linalg.norm(depth2img[:, 1])
    col3_norm = np.linalg.norm(depth2img[:, 2])
    
    print(f"列1范数 (fx估计): {col1_norm:.2f}")
    print(f"列2范数 (fy估计): {col2_norm:.2f}")
    print(f"列3范数 (fz估计): {col3_norm:.2f}")
    
    # 实际上SUNRGBD的depth2img不是标准K矩阵
    # 它是从点云坐标系到图像坐标系的投影
    
    print("\n=== 分析 ===")
    
    # SUNRGBD相机的fx约为577像素（标准）
    # 但这个样本的fx估计约为548
    
    print("标准SUNRGBD fx≈577像素")
    print("估计fx≈548，与标准接近")
    
    # 对于730宽度的图像
    img_w = 730
    fx_730 = 577.87 * img_w / 640
    print(f"\n对于730宽度: 估计fx≈{fx_730:.0f}像素")
    
    # 你使用的K
    print("\n=== 你使用的相机参数 ===")
    print("你使用的K: [[529.5, 0, 365], [0, 529.5, 262], [0, 0, 1]]")
    print("fx = 529.5")
    print("但标准应该是 fx≈548 或更大")
    print("\n这可能导致DepthPro的深度估计不准确!")

if __name__ == '__main__':
    analyze_camera()
