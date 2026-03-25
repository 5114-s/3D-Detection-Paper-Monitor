"""
Teacher 管线批量测试脚本 - 完全自动模式
与 train_distill_sunrgbd.py 流程一致

流程：
1. 加载 SUNRGBD 图像（从 Omni3D JSON）
2. Teacher 自动完成：RAM+Gemini → DINO → 深度融合 → SAM2 Mask → 几何拟合
3. 可视化 3D 伪标签

用法:
    python tools/test_teacher_pipeline.py --num_images 100 --output_dir test_teacher_results
"""
import os
import sys
import cv2
import numpy as np
import torch
import warnings
from tqdm import tqdm
import argparse
import json

# CPU 模式下禁用 xformers flash attention（CPU 不支持，报错 "No operator found"）
if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED_ATTENTION_OPTS"] = "all"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "teacher_student"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cubercnn"))

from teacher_student.teacher_detany3d import TeacherDetAny3D as TeacherDetector3D
from cubercnn.util import math_util as util_math


# ============== 工具函数 ==============
def estimate_K(width, height, hfov_deg=60.0):
    f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float32)


def _clip_segment_to_image(p0, p1, w, h, margin=50):
    """线段裁剪到图像内"""
    x1, y1 = float(p0[0]), float(p0[1])
    x2, y2 = float(p1[0]), float(p1[1])
    xmin, xmax = -margin, w + margin
    ymin, ymax = -margin, h + margin
    if (x1 < xmin and x2 < xmin) or (x1 > xmax and x2 > xmax) or (y1 < ymin and y2 < ymin) or (y1 > ymax and y2 > ymax):
        return None
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
    return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)


def _draw_3d_box(img, K, verts3d, color=(0, 255, 0), thickness=2):
    """绘制 3D 框"""
    K = np.asarray(K, dtype=np.float32)
    verts3d = np.asarray(verts3d, dtype=np.float32)
    h_img, w_img = img.shape[0], img.shape[1]
    if verts3d.shape != (8, 3):
        return
    if np.any(np.isnan(verts3d)) or np.any(np.isinf(verts3d)):
        return
    z = verts3d[:, 2]
    if np.any(z <= 0) or np.any(z > 50):
        return
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 5), (5, 6), (6, 2), (4, 5), (4, 7), (6, 7), (0, 4), (3, 7)]
    for i, j in edges:
        v0, v1 = verts3d[i], verts3d[j]
        if v0[2] < 0.1 or v1[2] < 0.1 or v0[2] > 50 or v1[2] > 50:
            continue
        p0 = (K @ v0) / max(v0[2], 1e-4)
        p1 = (K @ v1) / max(v1[2], 1e-4)
        clipped = _clip_segment_to_image(p0, p1, w_img, h_img)
        if clipped is None:
            continue
        q0, q1 = clipped
        cv2.line(img, (int(round(q0[0])), int(round(q0[1]))), (int(round(q1[0])), int(round(q1[1]))), color, thickness)


def _cuboid_verts(center_cam, dimensions, R_cam):
    """计算立方体角点"""
    cx, cy, cz = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])
    W, H, L = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    box3d = np.array([cx, cy, cz, W, H, L], dtype=np.float32)
    R = np.asarray(R_cam, dtype=np.float32)
    verts, _ = util_math.get_cuboid_verts_faces(box3d, R)
    if hasattr(verts, "numpy"):
        verts = verts.numpy()
    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim == 3:
        verts = verts[0]
    return verts


def _draw_text(img, text, pos, scale=0.5, color=(0, 255, 0)):
    x, y = int(np.clip(pos[0], 0, img.shape[1] - 1)), int(np.clip(pos[1], 0, img.shape[0] - 1))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)


# ============== 数据加载 ==============
def load_sunrgbd_samples(dataset_root, max_images=100, start_idx=0):
    """从 Omni3D JSON 加载 SUNRGBD 样本（与 train_distill_sunrgbd.py 一致）"""
    json_paths = [
        os.path.join(dataset_root, "Omni3D", "SUNRGBD_train.json"),
        os.path.join(dataset_root, "Omni3D_pl", "SUNRGBD_train.json"),
    ]
    
    json_path = None
    for p in json_paths:
        if os.path.exists(p):
            json_path = p
            break
    
    if json_path is None:
        print(f"找不到 JSON 文件，尝试过: {json_paths}")
        return []
    
    with open(json_path) as f:
        data = json.load(f)
    
    samples = []
    for img_info in data["images"][start_idx:start_idx + max_images]:
        file_path = os.path.join(dataset_root, img_info["file_path"])
        if not os.path.exists(file_path):
            continue
        
        samples.append({
            "path": file_path,
            "K": np.array(img_info["K"], dtype=np.float32) if "K" in img_info else None,
            "image_id": img_info["id"],
            "width": img_info["width"],
            "height": img_info["height"],
        })
    
    print(f"从 {json_path} 加载了 {len(samples)} 张图像")
    return samples


# ============== 主测试函数 ==============
def test_teacher_pipeline(
    samples,
    output_dir="test_teacher_results",
    device="cuda",
    use_sam2_mask=True,
    use_ram_gpt=True,
    depth_scale=1.0,
):
    """
    完全自动模式的 Teacher 管线测试
    
    Teacher 自动完成：
    1. RAM+Gemini 生成 text_prompt（如果 use_ram_gpt=True）
    2. Grounding DINO 2D 检测
    3. MoGe+DepthPro 深度融合
    4. SAM2 Mask 生成
    5. L-Shape 几何拟合生成 3D 伪标签
    """
    # 绝对路径
    output_dir = os.path.abspath(output_dir)
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "vis"), exist_ok=True)
    
    # 默认 text_prompt（当 RAM+Gemini 不可用时）
    default_text_prompt = "bed. chair. table. sofa. cabinet. desk. bookshelf. door. pillow. monitor."
    
    # 初始化 Teacher（完全自动模式）
    print(f"初始化 Teacher...")
    print(f"  - use_sam2_mask={use_sam2_mask}")
    print(f"  - use_ram_gpt={use_ram_gpt}")
    teacher = TeacherDetector3D(
        device=device,
        use_sam2_mask=use_sam2_mask,
        use_ram_gpt=use_ram_gpt,
    )
    
    # 统计
    stats = {
        "total_images": 0,
        "successful": 0,
        "failed": 0,
        "total_boxes": 0,
        "depth_stats": [],
        "no_detection": 0,
        "error_images": [],
    }
    
    # 处理每张图
    for sample in tqdm(samples, desc="Teacher 推理", unit="img"):
        img_path = sample["path"]
        stats["total_images"] += 1
        
        # 读取图像
        rgb = cv2.imread(img_path)
        if rgb is None:
            print(f"\n无法读取: {img_path}")
            stats["failed"] += 1
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # 内参：优先使用 JSON 中的 K，否则估计
        K = sample["K"] if sample["K"] is not None else estimate_K(w, h)
        
        # ============== Teacher 完全自动推理 ==============
        # text_prompt: RAM+Gemini 自动生成 或 使用默认 prompt
        text_prompt = None if use_ram_gpt else default_text_prompt
        
        # 直接调用 generate_pseudo_3d_boxes（内部包含所有步骤）
        try:
            pseudo_boxes, F_fused = teacher.generate_pseudo_3d_boxes(
                rgb,
                text_prompt=text_prompt,
                K=K,
                boxes_xyxy=None,  # 让 Teacher 内部做 DINO 检测
                phrases=None,
                depth_scale=depth_scale,
            )
        except Exception as e:
            print(f"\nTeacher 推理失败 {img_path}: {e}")
            stats["failed"] += 1
            stats["error_images"].append(img_path)
            continue
        
        # generate_pseudo_3d_boxes 返回 None 表示没有检测到
        if pseudo_boxes is None or (isinstance(pseudo_boxes, list) and len(pseudo_boxes) == 0):
            print(f"\n未检测到物体: {img_path}")
            stats["no_detection"] += 1
            vis_img = rgb.copy()
            cv2.putText(vis_img, "No objects detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, "vis", os.path.basename(img_path)),
                       cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            continue
        
        # 统计成功的伪标签
        n_valid = 0
        for pseudo in pseudo_boxes:
            if pseudo is not None and "center_cam" in pseudo:
                n_valid += 1
                stats["depth_stats"].append(float(pseudo["center_cam"][2]))
        
        stats["successful"] += 1
        stats["total_boxes"] += n_valid
        
        # 可视化
        vis_img = rgb.copy()
        valid_count = 0
        
        for pseudo in pseudo_boxes:
            if pseudo is None:
                continue
            
            phrase = pseudo.get("phrase", "unknown")
            
            # 2D 框 (绿色)
            if "box_2d_xyxy" in pseudo:
                box = pseudo["box_2d_xyxy"]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            
            # 3D 框 (红色) - 仅有效伪标签
            if "center_cam" in pseudo:
                try:
                    R = pseudo.get("R_cam", np.eye(3))
                    verts = _cuboid_verts(pseudo["center_cam"], pseudo["dimensions"], R)
                    # 用 pseudo 中的 K（融合后的K）
                    vis_K = pseudo.get("K", K)
                    _draw_3d_box(vis_img, vis_K, verts, color=(0, 0, 255), thickness=2)
                    depth_text = f"z={pseudo['center_cam'][2]:.2f}m"
                    valid_count += 1
                except Exception as e:
                    depth_text = f"ERR:{e}"
            else:
                depth_text = "NO_3D"
            
            # 标签
            label = f"{phrase[:15]} {depth_text}"
            _draw_text(vis_img, label, (x1, max(y1 - 10, 20)), scale=0.45, color=(0, 255, 0))
        
        # 保存
        out_path = os.path.join(output_dir, "vis", os.path.basename(img_path))
        
        # 调试
        print(f"[DEBUG] vis_img shape: {vis_img.shape}, dtype: {vis_img.dtype}")
        
        vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        print(f"[DEBUG] vis_bgr shape: {vis_bgr.shape}, dtype: {vis_bgr.dtype}")
        
        result = cv2.imwrite(out_path, vis_bgr)
        print(f"[DEBUG] imwrite result: {result}, exists: {os.path.exists(out_path)}")
        
        if result:
            print(f"  保存: {os.path.basename(img_path)}, 有效3D: {valid_count}")
        else:
            print(f"  保存失败: {os.path.basename(img_path)}")
    
    # 输出统计
    print("\n" + "=" * 50)
    print("Teacher 测试结果统计")
    print("=" * 50)
    print(f"总图像数: {stats['total_images']}")
    print(f"成功处理: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"未检测到物体: {stats['no_detection']}")
    print(f"总 3D 框: {stats['total_boxes']}")
    if stats['depth_stats']:
        print(f"深度范围: {min(stats['depth_stats']):.2f}m - {max(stats['depth_stats']):.2f}m")
        print(f"深度均值: {np.mean(stats['depth_stats']):.2f}m")
    print(f"结果保存: {output_dir}/vis/")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher 管线批量测试（完全自动模式）")
    parser.add_argument("--num_images", type=int, default=100, help="测试图像数量")
    parser.add_argument("--start_idx", type=int, default=0, help="起始索引")
    parser.add_argument("--output_dir", type=str, default="test_teacher_results", help="输出目录")
    parser.add_argument("--data_root", type=str, default="/data/ZhaoX/OVM3D-Det-1/datasets", help="数据集根目录")
    parser.add_argument("--device", type=str, default="cuda", help="GPU 设备 (cuda:0 或 cuda:1)")
    parser.add_argument("--no_sam2", action="store_true", help="禁用 SAM2，改用 DetAny3D mask")
    parser.add_argument("--no_ram_gpt", action="store_true", help="禁用 RAM+Gemini，使用固定 prompt")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="深度缩放因子")
    args = parser.parse_args()
    
    # 从 Omni3D JSON 加载样本
    samples = load_sunrgbd_samples(args.data_root, args.num_images, args.start_idx)
    
    if len(samples) == 0:
        print("未找到图像，请检查数据集路径")
        print(f"  期望: {args.data_root}/Omni3D/SUNRGBD_train.json")
        sys.exit(1)
    
    # 测试
    stats = test_teacher_pipeline(
        samples,
        output_dir=args.output_dir,
        device=args.device,
        use_sam2_mask=not args.no_sam2,
        use_ram_gpt=not args.no_ram_gpt,
        depth_scale=args.depth_scale,
    )
