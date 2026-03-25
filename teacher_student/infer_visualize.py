"""
学生模型推理脚本 - 可视化3D检测结果（简化版，不依赖 gradio）
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "teacher_student"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cubercnn"))

from teacher_student.student_3d_head import Student3DHead
from teacher_student.teacher_detany3d import TeacherDetAny3D as TeacherDetector3D
from cubercnn.util import math_util as util_math


# ============== 简化版 3D 绘制函数 ==============
def _clip_segment_to_image(p0, p1, w, h, margin=50):
    """将线段 (p0,p1) 裁剪到图像内 [0,w] x [0,h]，margin 为允许的超出量。返回 (q0,q1) 或 None。"""
    x1, y1 = float(p0[0]), float(p0[1])
    x2, y2 = float(p1[0]), float(p1[1])
    xmin, xmax = -margin, w + margin
    ymin, ymax = -margin, h + margin
    # 完全在外则跳过
    if (x1 < xmin and x2 < xmin) or (x1 > xmax and x2 > xmax) or (y1 < ymin and y2 < ymin) or (y1 > ymax and y2 > ymax):
        return None
    # Liang–Barsky 式裁剪到 [xmin,xmax] x [ymin,ymax]
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for (p, q, lo, hi) in [(-dx, x1 - xmin, 0, 1), (dx, xmax - x1, 0, 1), (-dy, y1 - ymin, 0, 1), (dy, ymax - y1, 0, 1)]:
        if abs(p) < 1e-9:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None
    q0 = (x1 + t0 * dx, y1 + t0 * dy)
    q1 = (x1 + t1 * dx, y1 + t1 * dy)
    return (q0, q1)


def _draw_3d_box_from_verts(img, K, verts3d, color=(0, 255, 0), thickness=2, zplane=0.05, z_min_draw=0.2, z_max_draw=50.0, eps=1e-4):
    """将 8x3 的相机坐标系顶点用 K 投影到图像并画线；裁剪到图像内，避免 z 过小导致射线飞出画面。"""
    K = np.asarray(K, dtype=np.float32)
    verts3d = np.asarray(verts3d, dtype=np.float32)
    h_img, w_img = img.shape[0], img.shape[1]
    if verts3d.shape != (8, 3):
        return
    if np.any(np.isnan(verts3d)) or np.any(np.isinf(verts3d)):
        return
    z = verts3d[:, 2]
    if np.any(z <= 0) or np.any(z > z_max_draw):
        return
    if np.any(z < zplane) and np.all(z < zplane):
        return
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 5), (5, 6), (6, 2), (4, 5), (4, 7), (6, 7), (0, 4), (3, 7)]
    for i, j in edges:
        v0, v1 = verts3d[i], verts3d[j]
        z0, z1 = v0[2], v1[2]
        if z0 < zplane and z1 < zplane:
            continue
        s = (zplane - z0) / max(z1 - z0, eps)
        new_v = v0 + s * (v1 - v0)
        if z0 < zplane and z1 >= zplane:
            v0 = new_v
        elif z0 >= zplane and z1 < zplane:
            v1 = new_v
        # 若投影前 z 过小或过大，投影会爆炸，只画在合理深度内的边
        if v0[2] < z_min_draw or v1[2] < z_min_draw or v0[2] > z_max_draw or v1[2] > z_max_draw:
            continue
        p0 = (K @ v0) / max(v0[2], eps)
        p1 = (K @ v1) / max(v1[2], eps)
        clipped = _clip_segment_to_image(p0, p1, w_img, h_img, margin=100)
        if clipped is None:
            continue
        q0, q1 = clipped
        cv2.line(img, (int(round(q0[0])), int(round(q0[1]))), (int(round(q1[0])), int(round(q1[1]))), color, thickness)


def _draw_text(img, text, pos, scale=0.6, bg_color=(0, 255, 255)):
    x, y = int(np.clip(pos[0], 0, img.shape[1] - 1)), int(np.clip(pos[1], 0, img.shape[0] - 1))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 4, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)


def _cuboid_verts_with_pose(center_cam, dimensions, R_cam):
    """
    根据教师输出的 center_cam、dimensions (3,) [W, H, L] 和 R_cam，
    用 cubercnn 约定得到相机系下 8 个顶点。

    【尺寸语义统一】：
    当前 dimensions 严格按照 [dz, dy, dx] 排列，即：
      dimensions[0] = W (Z方向深度/纵深)
      dimensions[1] = H (Y方向高度)
      dimensions[2] = L (X方向宽度)

    get_cuboid_verts_faces 的 box3d=[cx, cy, cz, W, H, L] 约定：
      局部 W→Z轴, H→Y轴, L→X轴
    """
    cx, cy, cz = center_cam
    
    # 核心修改：直接按 [W, H, L] 顺序解包（与 teacher_geometry 完美对齐）
    W_cam = float(dimensions[0])  # dz
    H_cam = float(dimensions[1])  # dy
    L_cam = float(dimensions[2])  # dx
    
    # 组装给 cubercnn 的标准输入 [X, Y, Z, W, H, L]
    box3d = np.array([cx, cy, cz, W_cam, H_cam, L_cam], dtype=np.float32)
    R = np.asarray(R_cam, dtype=np.float32)
    
    # 调用 cubercnn 的底层投影逻辑
    verts, _ = util_math.get_cuboid_verts_faces(box3d, R)
    
    if hasattr(verts, "numpy"):
        verts = verts.numpy()
    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim == 3:
        verts = verts[0]
        
    return verts  # 返回 8x3 的角点坐标


def estimate_K(width, height, hfov_deg=60.0):
    f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float32)


def get_sunrgbd_K(image_path):
    """
    若图像来自 SUN RGB-D (路径含 sunrgbd 且存在 calib 文件)，返回真实内参 K；
    否则返回 None，调用方用 estimate_K。
    """
    image_path = os.path.abspath(image_path)
    if "sunrgbd" not in image_path.lower():
        return None
    base = os.path.splitext(os.path.basename(image_path))[0]
    dir_img = os.path.dirname(image_path)
    # sunrgbd_trainval/image/xxx.jpg -> sunrgbd_trainval/calib/xxx.txt
    root = os.path.dirname(dir_img)
    calib_path = os.path.join(root, "calib", base + ".txt")
    if not os.path.isfile(calib_path):
        return None
    try:
        with open(calib_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    vals = [float(x) for x in parts[:9]]
                    # K 行满足: [fx,0,0, 0,fy,0, cx,cy,1]
                    if abs(vals[8] - 1.0) < 1e-6 and abs(vals[1]) < 1e-6 and abs(vals[3]) < 1e-6:
                        fx, fy, cx, cy = vals[0], vals[4], vals[6], vals[7]
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)
                        return K
    except Exception:
        pass
    return None


# ============== Grounding DINO 2D 检测 ==============
def run_2d_detection(image_rgb, caption, device="cuda", box_threshold=0.25, text_threshold=0.2):
    """使用 Grounding DINO 进行 2D 检测"""
    from PIL import Image
    import grounding_dino.groundingdino.datasets.transforms as T
    from grounding_dino.groundingdino.util.inference import load_model, predict as gdino_predict
    
    _REPO_ROOT = PROJECT_ROOT
    _GSAM = os.path.join(_REPO_ROOT, "Grounded-SAM-2")
    _GDINO = os.path.join(_GSAM, "grounding_dino")
    
    config = os.path.join(_GDINO, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    ckpt = os.path.join(_GSAM, "checkpoints", "groundingdino_swint_ogc.pth")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(_REPO_ROOT, "checkpoints", "groundingdino_swint_ogc.pth")
    
    print("加载 Grounding DINO...")
    model = load_model(config, ckpt, device=device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 转换为 PIL Image
    image_pil = Image.fromarray(image_rgb)
    image_transformed, _ = transform(image_pil, None)
    caption = caption.strip().lower()
    
    boxes, scores, phrases = gdino_predict(model, image_transformed, caption, box_threshold, text_threshold)
    # DINO 返回归一化 (cx, cy, w, h)；转为像素 (x1, y1, x2, y2) 供 SAM2/几何/可视化使用
    h, w = image_rgb.shape[:2]
    if len(boxes) > 0:
        boxes = boxes.cpu().numpy()
        boxes_abs = boxes * np.array([w, h, w, h], dtype=np.float32)
        x1 = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
        y1 = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
        x2 = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
        y2 = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    else:
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
    return boxes_xyxy, scores, phrases


# ============== Teacher 3D 几何推理 ==============
def get_pseudo_boxes(teacher, rgb, K, boxes_xyxy, phrases, depth_scale=1.0):
    """获取伪 3D 框和特征图 F_fused。传入的 boxes_xyxy/phrases 与 2D 检测一致，教师只做深度+Mask+几何，不重复跑 DINO。"""
    text_prompt = " ".join([p + "." for p in phrases])
    pseudo_boxes, F_fused = teacher.generate_pseudo_3d_boxes(
        rgb, text_prompt, K,
        boxes_xyxy=boxes_xyxy,
        phrases=phrases,
        depth_scale=depth_scale,
    )

    if pseudo_boxes is None or len(pseudo_boxes) == 0:
        print("警告: 教师未生成伪标签")
        pseudo_list = [None] * len(phrases)
        F_fused = None
    else:
        pseudo_list = pseudo_boxes
    
    return pseudo_list, F_fused


def run_student_inference(student_head, F_fused, boxes_xyxy, image_size_hw, device="cuda"):
    """用学生头预测 3D 框"""
    import torch.nn.functional as F
    
    if F_fused is None or len(boxes_xyxy) == 0:
        return []
    
    # boxes 归一化到 [0,1]
    boxes_t = torch.from_numpy(np.array(boxes_xyxy)).float().to(device)
    boxes_t[:, [0, 2]] /= image_size_hw[1]
    boxes_t[:, [1, 3]] /= image_size_hw[0]
    
    # 学生前向 (LIFT 架构：预测 2D 偏移 + 深度)
    with torch.no_grad():
        deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
            F_fused.to(device), boxes_t, torch.tensor(image_size_hw, device=device)
        )
    
    # LIFT 反投影得到 3D 中心
    from teacher_student.student_3d_head import compute_2d_box_info, apply_2d_deltas, lift_project_to_3d
    src_ctr_x, src_ctr_y, src_widths, src_heights = compute_2d_box_info(boxes_t, image_size_hw)
    cube_x, cube_y = apply_2d_deltas(deltas, src_ctr_x, src_ctr_y, src_widths, src_heights)
    K_tensor = torch.tensor(K, device=device)
    pred_center = lift_project_to_3d(cube_x, cube_y, pred_z, K_tensor).cpu().numpy()
    pred_dims_np = pred_dims.cpu().numpy()
    
    # 6D -> 3x3 旋转矩阵
    from teacher_student.student_3d_head import rotation_6d_to_matrix
    R_matrices = rotation_6d_to_matrix(pred_pose_6d).cpu().numpy()
    
    results = []
    for i in range(len(pred_center)):
        results.append({
            "center_cam": pred_center[i],
            "dimensions": pred_dims_np[i],
            "R_cam": R_matrices[i],
        })
    return results


# ============== 主推理函数 ==============
def load_student_model(ckpt_path, device="cuda"):
    """加载训练好的学生头"""
    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        use_uncertainty=False,
        use_disentangled=True,
    ).to(device)
    
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        student_head.load_state_dict(state_dict, strict=False)
        print(f"加载学生头权重: {ckpt_path}")
    else:
        print(f"警告: 权重文件不存在 {ckpt_path}")
    
    student_head.eval()
    return student_head


def run_inference(image_path, text_prompt=None, student_ckpt="output/distill/student_3d_head.pth", 
                  device="cuda", use_sam2_mask=True, use_ram_gpt=True, depth_scale=1.0):
    """
    推理并可视化
    
    Args:
        image_path: 图像路径
        text_prompt: 文本提示(可选,如果use_ram_gpt=True会自动生成)
        student_ckpt: 学生模型权重路径
        device: 计算设备
        use_sam2_mask: 是否使用SAM2生成mask
        use_ram_gpt: 是否使用RAM+GPT自动生成text_prompt
        depth_scale: 深度缩放因子
    """
    # 加载图像
    rgb = cv2.imread(image_path)
    if rgb is None:
        print(f"无法读取图像: {image_path}")
        return
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    
    # 内参：SUN RGB-D 优先用真实 calib，否则用估计
    K = get_sunrgbd_K(image_path)
    if K is not None:
        print(f"使用 SUN RGB-D 内参: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    else:
        K = estimate_K(w, h)
        print(f"使用估计内参 (hfov=60°)")
    print(f"图像尺寸: {w}x{h}")
    
    # 改进1和3: RAM+GPT自动生成text_prompt
    if use_ram_gpt and text_prompt is None:
        print(">> 使用RAM+Gemini自动生成text prompt...")
        from teacher_student.ram_gpt_labeler import create_ram_gpt_labeler
        try:
            ram_gpt_labeler = create_ram_gpt_labeler(device=device, use_gemini=True)
            text_prompt = ram_gpt_labeler.generate_text_prompt(rgb)
            if text_prompt is None:
                print("警告: RAM+Gemini生成失败,使用默认prompt")
                text_prompt = "chair. table. bed. sofa."
        except Exception as e:
            print(f"警告: RAM+GPT初始化失败: {e},使用默认prompt")
            text_prompt = "chair. table. bed. sofa."
    
    if text_prompt is None:
        text_prompt = "chair. table. bed. sofa."  # 默认prompt
    
    print(f">> 使用text_prompt: {text_prompt[:80]}...")
    
    # 2D 检测（返回已转为像素 xyxy）
    print("运行 2D 检测...")
    boxes_xyxy, scores, phrases = run_2d_detection(rgb, text_prompt, device)
    print(f"检测到 {len(boxes_xyxy)} 个物体: {phrases}")
    
    if len(boxes_xyxy) == 0:
        print("未检测到物体")
        return
    
    # Teacher 3D 几何推理（默认用 SAM2 生成 2D mask，避免 DetAny3D decoder 的 3D 向输出）
    print("运行几何推理...")
    teacher = TeacherDetector3D(device=device, use_sam2_mask=use_sam2_mask)
    pseudo_list, F_fused = get_pseudo_boxes(teacher, rgb, K, boxes_xyxy, phrases, depth_scale=depth_scale)
    n_pseudo = sum(1 for p in pseudo_list if p is not None)
    print(f"生成 {n_pseudo} 个伪标签 (共 {len(pseudo_list)} 个检测)")
    
    # 学生推理
    print("运行学生头推理...")
    student_head = load_student_model(student_ckpt, device)
    image_size_hw = (h, w)
    student_preds = run_student_inference(student_head, F_fused, boxes_xyxy, image_size_hw, device)
    print(f"学生预测 {len(student_preds)} 个 3D 框")
    
    # 同时画教师（红）和学生（蓝）
    print("可视化：教师 3D 伪标签（红）vs 学生预测（蓝）...")
    
    vis_img = rgb.copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # pseudo_list 与 phrases 同长，失败项为 None；按索引对齐 2D 框、教师伪标签、学生预测
    for i in range(len(boxes_xyxy)):
        box = boxes_xyxy[i]
        phrase = phrases[i]
        x1, y1, x2, y2 = box
        # 2D 框（绿色）
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 3D 框 - 教师 (红色)，仅当该索引有伪标签时绘制
        pseudo = pseudo_list[i] if i < len(pseudo_list) else None
        if pseudo is not None and "center_cam" in pseudo and "R_cam" in pseudo:
            center_cam = pseudo["center_cam"]
            dims = pseudo["dimensions"]
            R_cam = pseudo["R_cam"]
            verts_teacher = _cuboid_verts_with_pose(center_cam, dims, R_cam)
            _draw_3d_box_from_verts(vis_img, K, verts_teacher, color=(0, 0, 255), thickness=2)
        elif pseudo is not None and "center_cam" in pseudo:
            center_cam = pseudo["center_cam"]
            dims = pseudo["dimensions"]
            verts_teacher = _cuboid_verts_with_pose(center_cam, dims, np.eye(3))
            _draw_3d_box_from_verts(vis_img, K, verts_teacher, color=(0, 0, 255), thickness=2)

        # 3D 框 - 学生 (蓝色)：对 center 深度和尺寸做合理裁剪，避免异常预测导致投影爆炸
        if i < len(student_preds):
            student_pred = student_preds[i]
            if "center_cam" in student_pred:
                center_cam_s = np.array(student_pred["center_cam"], dtype=np.float64)
                dims_s = np.array(student_pred["dimensions"], dtype=np.float64)
                R_cam_s = student_pred["R_cam"]
                center_cam_s[2] = np.clip(center_cam_s[2], 0.3, 20.0)
                dims_s = np.clip(dims_s, 0.05, 5.0)
                verts_student = _cuboid_verts_with_pose(center_cam_s, dims_s, R_cam_s)
                _draw_3d_box_from_verts(vis_img, K, verts_student, color=(255, 0, 0), thickness=2)

        _draw_text(vis_img, phrase, (int(x1), int(y1)-10), scale=0.7, bg_color=(0, 255, 0))
    
    # 保存结果
    output_path = image_path.replace(".jpg", "_result.jpg")
    cv2.imwrite(output_path, vis_img)
    print(f"结果保存到: {output_path}")
    
    return vis_img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg")
    parser.add_argument("--text", type=str, default=None, help="文本提示(可选,不提供则使用RAM+GPT自动生成)")
    parser.add_argument("--ckpt", type=str, default="output/distill/student_3d_head.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_sam2_mask", action="store_true", help="禁用 SAM2，改用 DetAny3D 的 mask decoder（易出现全图/噪点）")
    parser.add_argument("--no_ram_gpt", action="store_true", help="禁用RAM+Gemini,使用手动text_prompt")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="深度→米的缩放。若 3D 框整体偏近/飘在空中可试 2.0 或 3.0")
    args = parser.parse_args()

    run_inference(
        args.image, 
        text_prompt=args.text, 
        student_ckpt=args.ckpt, 
        device=args.device, 
        use_sam2_mask=not args.no_sam2_mask,
        use_ram_gpt=not args.no_ram_gpt,  # 默认启用RAM+Gemini
        depth_scale=args.depth_scale
    )
