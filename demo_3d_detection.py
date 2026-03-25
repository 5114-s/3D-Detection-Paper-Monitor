"""
完整3D检测演示脚本
测试教师端的: RAM+GPT -> Grounding DINO -> MoGe+DepthPro融合深度 -> SAM2掩码 -> 3D框拟合
"""
import os
import sys
import cv2
import numpy as np
import torch

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, PROJECT_ROOT)

def draw_3d_box(img, K, center_cam, dims, R_cam, color=(0, 255, 0), thickness=2):
    """绘制3D边界框"""
    W, H, L = dims[1], dims[2], dims[0]  # [L, W, H] -> [W, H, L]
    cx, cy, cz = center_cam[0], center_cam[1], center_cam[2]
    
    box3d = np.array([[cx, cy, cz, W, H, L]], dtype=np.float32)
    box3d_t = torch.from_numpy(box3d).float()
    R_t = torch.from_numpy(R_cam).float()
    
    from cubercnn.util import math_util as util_math
    verts_t, _ = util_math.get_cuboid_verts_faces(box3d_t, R_t)
    verts = verts_t.squeeze(0).numpy()
    
    # 投影到2D
    proj = (K @ verts.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    
    # 绘制边
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        pt1 = (int(proj[i,0]), int(proj[i,1]))
        pt2 = (int(proj[j,0]), int(proj[j,1]))
        if 0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0]:
            if 0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]:
                cv2.line(img, pt1, pt2, color, thickness)
    
    return img

def main():
    print("="*60)
    print("3D检测完整流程演示")
    print("="*60)
    
    # 1. 加载图像
    image_path = '/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"\n[1] 加载图像: {img_rgb.shape}")
    
    # 2. 相机内参 (SUN RGB-D)
    K = np.array([
        [529.5, 0, 365.0],
        [0, 529.5, 262.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 3. 加载教师模型 (不使用RAM+GPT，使用手动text_prompt以便快速演示)
    print("\n[2] 加载教师模型...")
    from teacher_student.teacher_detany3d import TeacherDetAny3D
    
    teacher = TeacherDetAny3D(
        device='cuda',
        use_sam2_mask=True,  # 使用SAM2掩码
        use_ram_gpt=False,   # 演示时不用RAM+GPT
    )
    print("教师模型加载完成")
    
    # 4. 设置text prompt
    text_prompt = "bed. table. chair. sofa."
    
    # 5. 运行教师管道
    print(f"\n[3] 运行教师管道...")
    print(f"    Text Prompt: {text_prompt}")
    
    pseudo_list, F_fused = teacher.generate_pseudo_3d_boxes(
        img_rgb,
        text_prompt=text_prompt,
        K=K,
        use_lshape=True,
        depth_scale=1.0,
    )
    
    # 6. 可视化结果
    print(f"\n[4] 可视化结果...")
    vis_img = img.copy()
    
    # 绘制检测到的物体
    num_detected = 0
    for i, pseudo in enumerate(pseudo_list):
        if pseudo is None:
            continue
        
        num_detected += 1
        center = pseudo['center_cam']
        dims = pseudo['dimensions']
        R = pseudo['R_cam']
        phrase = pseudo.get('phrase', f'object_{i}')
        
        print(f"    [{num_detected}] {phrase}:")
        print(f"        中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] m")
        print(f"        尺寸: [{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}] m (L×W×H)")
        
        # 绘制3D框 (绿色)
        draw_3d_box(vis_img, K, center, dims, R, color=(0, 255, 0), thickness=2)
        
        # 绘制2D框 (蓝色)
        box2d = pseudo.get('box_2d_xyxy', None)
        if box2d is not None:
            x1, y1, x2, y2 = box2d.astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 添加标签
        label = f"{phrase}: z={center[2]:.1f}m"
        if box2d is not None:
            cv2.putText(vis_img, label, (int(box2d[0]), int(box2d[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制图例
    cv2.putText(vis_img, "Green: 3D Box | Blue: 2D Box", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Detected: {num_detected} objects", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 保存结果
    output_path = '/data/ZhaoX/OVM3D-Det-1/demo_3d_result.jpg'
    cv2.imwrite(output_path, vis_img)
    print(f"\n结果已保存: {output_path}")
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)

if __name__ == "__main__":
    main()
