"""
离线伪标签批量生成脚本
Teacher 管线自动完成：RAM+Gemini → DINO → 深度融合 → SAM2 Mask → 几何拟合
将伪标签保存为 JSON/Pickle，供后续学生训练使用。

用法:
    # 基本用法（使用默认 prompt）
    python tools/generate_offline_pseudo_labels.py --num_images 100 --output_dir pseudo_labels/sunrgbd

    # 使用 RAM+GPT 自动生成 prompt
    python tools/generate_offline_pseudo_labels.py --num_images 100 --use_ram_gpt --output_dir pseudo_labels/sunrgbd

    # 离线模式（跳过 RAM+GPT，使用默认 prompt）
    python tools/generate_offline_pseudo_labels.py --num_images 1000 --offline --output_dir pseudo_labels/sunrgbd
"""
import os
import sys
import cv2
import json
import pickle
import numpy as np
import torch
import warnings
from tqdm import tqdm
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "teacher_student"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cubercnn"))

from teacher_student.teacher_detany3d import TeacherDetAny3D
from cubercnn.util import math_util as util_math


def estimate_K(width, height, hfov_deg=60.0):
    f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float32)


def load_sunrgbd_samples(dataset_root, max_images=100, start_idx=0, json_name="SUNRGBD_train.json"):
    """从 Omni3D JSON 加载 SUNRGBD 样本"""
    json_paths = [
        os.path.join(dataset_root, "Omni3D", json_name),
        os.path.join(dataset_root, "Omni3D_pl", json_name),
        os.path.join(dataset_root, json_name),
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
            "image_id": img_info.get("id", -1),
            "width": img_info["width"],
            "height": img_info["height"],
        })

    print(f"从 {json_path} 加载了 {len(samples)} 张图像")
    return samples


def serialize_pseudo_box(pseudo):
    """将伪标签 dict 序列化为 JSON 兼容格式"""
    if pseudo is None:
        return None
    result = {}
    for key, value in pseudo.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = float(value)
        elif value is None:
            result[key] = None
        elif isinstance(value, (list, tuple)):
            result[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
        else:
            result[key] = str(value) if not isinstance(value, (int, float, bool, str)) else value
    return result


def draw_3d_visualization(rgb, K, pseudo_boxes, output_path):
    """绘制 3D 伪标签可视化图"""
    img = rgb.copy()
    h_img, w_img = img.shape[:2]

    for pseudo in pseudo_boxes:
        if pseudo is None:
            continue
        center_cam = np.asarray(pseudo["center_cam"], dtype=np.float32)
        dimensions = np.asarray(pseudo["dimensions"], dtype=np.float32)
        R_cam = np.asarray(pseudo["R_cam"], dtype=np.float32)
        phrase = pseudo.get("phrase", "object")

        # 计算 8 个角点
        W, H, L = dimensions
        cx, cy, cz = center_cam
        corners_local = np.array([
            [-L/2, -W/2, -H/2], [ L/2, -W/2, -H/2],
            [-L/2,  W/2, -H/2], [ L/2,  W/2, -H/2],
            [-L/2, -W/2,  H/2], [ L/2, -W/2,  H/2],
            [-L/2,  W/2,  H/2], [ L/2,  W/2,  H/2],
        ], dtype=np.float32)
        corners_world = (corners_local @ R_cam.T) + center_cam

        # 投影到 2D
        valid_pts = []
        for v in corners_world:
            if v[2] > 0.1 and v[2] < 50:
                p = K @ v
                p = p / p[2]
                if 0 <= p[0] < w_img and 0 <= p[1] < h_img:
                    valid_pts.append((int(p[0]), int(p[1])))
        if len(valid_pts) < 4:
            continue

        # 绘制线框
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (1, 5), (5, 6), (6, 2), (4, 5), (4, 7),
                 (6, 7), (0, 4), (3, 7)]
        for i, j in edges:
            if i < len(valid_pts) and j < len(valid_pts):
                cv2.line(img, valid_pts[i], valid_pts[j], (0, 255, 0), 2)

        # 绘制类别标签
        if valid_pts:
            cx_px = int(np.mean([p[0] for p in valid_pts]))
            cy_px = int(np.mean([p[1] for p in valid_pts]))
            cv2.putText(img, phrase, (cx_px, max(cy_px - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def generate_offline_pseudo_labels(
    dataset_root,
    output_dir,
    max_images=100,
    start_idx=0,
    device="cuda",
    use_sam2_mask=True,
    use_ram_gpt=True,
    depth_scale=1.0,
    save_features=True,
    save_visualization=True,
    overwrite=False,
):
    """
    离线批量生成伪标签

    Args:
        dataset_root: SUNRGBD 数据集根目录
        output_dir: 输出目录（包含 labels/ 和 vis/ 子目录）
        max_images: 最大处理图像数
        start_idx: JSON 中的起始索引
        device: cuda/cpu
        use_sam2_mask: 是否使用 SAM2 mask
        use_ram_gpt: 是否使用 RAM+GPT 自动生成 prompt
        depth_scale: 深度缩放因子
        save_features: 是否保存 F_fused 特征图（pickle 格式）
        save_visualization: 是否保存可视化图
        overwrite: 是否覆盖已有结果
    """
    output_dir = os.path.abspath(output_dir)
    labels_dir = os.path.join(output_dir, "labels")
    features_dir = os.path.join(output_dir, "features")
    vis_dir = os.path.join(output_dir, "vis")

    os.makedirs(labels_dir, exist_ok=True)
    if save_features:
        os.makedirs(features_dir, exist_ok=True)
    if save_visualization:
        os.makedirs(vis_dir, exist_ok=True)

    # 默认 prompt（离线模式）
    default_prompt = "bed. chair. table. sofa. cabinet. desk. bookshelf. door. pillow. monitor. person. book."

    # 初始化 Teacher
    print("=" * 60)
    print("离线伪标签生成器")
    print(f"  数据集: {dataset_root}")
    print(f"  输出目录: {output_dir}")
    print(f"  最大图像数: {max_images}")
    print(f"  SAM2 Mask: {use_sam2_mask}")
    print(f"  RAM+GPT: {use_ram_gpt}")
    print(f"  深度缩放: {depth_scale}")
    print(f"  保存特征: {save_features}")
    print("=" * 60)

    teacher = TeacherDetAny3D(
        device=device,
        use_sam2_mask=use_sam2_mask,
        use_ram_gpt=use_ram_gpt,
    )

    # 加载数据
    samples = load_sunrgbd_samples(dataset_root, max_images, start_idx)
    if not samples:
        print("没有加载到样本，检查数据集路径！")
        return

    # 统计
    stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": dataset_root,
        "total_images": 0,
        "successful": 0,
        "no_detection": 0,
        "errors": 0,
        "total_boxes": 0,
        "failed_images": [],
        "category_counts": {},
    }

    for sample in tqdm(samples, desc="生成伪标签", unit="img"):
        img_path = sample["path"]
        image_id = sample["image_id"]
        stats["total_images"] += 1

        # 输出文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.json")
        feature_path = os.path.join(features_dir, f"{base_name}.pkl")

        # 跳过已有结果（除非 overwrite）
        if not overwrite and os.path.exists(label_path):
            print(f"\n跳过已有: {label_path}")
            continue

        # 读取图像
        rgb = cv2.imread(img_path)
        if rgb is None:
            print(f"\n无法读取: {img_path}")
            stats["errors"] += 1
            stats["failed_images"].append(img_path)
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # 内参
        K = sample["K"] if sample["K"] is not None else estimate_K(w, h)

        # text_prompt
        text_prompt = None if use_ram_gpt else default_prompt

        # Teacher 推理
        try:
            pseudo_boxes, F_fused = teacher.generate_pseudo_3d_boxes(
                rgb,
                text_prompt=text_prompt,
                K=K,
                boxes_xyxy=None,
                phrases=None,
                depth_scale=depth_scale,
            )
        except Exception as e:
            print(f"\nTeacher 推理失败 [{img_path}]: {e}")
            stats["errors"] += 1
            stats["failed_images"].append(img_path)
            continue

        if pseudo_boxes is None or (isinstance(pseudo_boxes, list) and len(pseudo_boxes) == 0):
            print(f"\n未检测到物体: {img_path}")
            stats["no_detection"] += 1
            continue

        # 统计有效伪标签
        valid_boxes = []
        for pseudo in pseudo_boxes:
            if pseudo is not None and "center_cam" in pseudo:
                valid_boxes.append(pseudo)
                phrase = pseudo.get("phrase", "unknown")
                stats["category_counts"][phrase] = stats["category_counts"].get(phrase, 0) + 1

        stats["successful"] += 1
        stats["total_boxes"] += len(valid_boxes)

        # 保存伪标签 JSON
        pseudo_data = {
            "image_path": img_path,
            "image_id": image_id,
            "width": w,
            "height": h,
            "K": K.tolist(),
            "depth_scale": depth_scale,
            "num_boxes": len(valid_boxes),
            "pseudo_boxes": [serialize_pseudo_box(p) for p in pseudo_boxes],
        }
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(pseudo_data, f, indent=2, ensure_ascii=False)

        # 保存特征（如果需要）
        if save_features and F_fused is not None:
            feature_data = {
                "image_path": img_path,
                "image_id": image_id,
                "F_fused": F_fused.cpu().numpy(),
            }
            with open(feature_path, "wb") as f:
                pickle.dump(feature_data, f)

        # 保存可视化
        if save_visualization:
            vis_path = os.path.join(vis_dir, f"{base_name}.jpg")
            draw_3d_visualization(rgb, K, pseudo_boxes, vis_path)

    # 保存统计
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("伪标签生成完成")
    print(f"  总图像数: {stats['total_images']}")
    print(f"  成功: {stats['successful']}")
    print(f"  未检测: {stats['no_detection']}")
    print(f"  错误: {stats['errors']}")
    print(f"  总 3D 框: {stats['total_boxes']}")
    print(f"  输出目录: {output_dir}")
    print(f"  统计文件: {stats_path}")
    print("=" * 60)

    if stats["category_counts"]:
        print("\n类别分布:")
        for cat, count in sorted(stats["category_counts"].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="离线伪标签批量生成")
    parser.add_argument("--dataset_root", type=str,
                       default="/data/ZhaoX/OVM3D-Det-1",
                       help="项目根目录（JSON 中的 file_path 已是相对于 datasets/ 的相对路径）")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels/sunrgbd",
                       help="输出目录")
    parser.add_argument("--num_images", type=int, default=100,
                       help="最大处理图像数")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="起始索引")
    parser.add_argument("--device", type=str, default="cuda",
                       help="cuda/cpu")
    parser.add_argument("--use_sam2_mask", action="store_true", default=True,
                       help="使用 SAM2 Mask")
    parser.add_argument("--no_sam2_mask", action="store_true",
                       help="不使用 SAM2 Mask")
    parser.add_argument("--use_ram_gpt", action="store_true", default=False,
                       help="使用 RAM+GPT 自动生成 prompt")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="深度缩放因子")
    parser.add_argument("--save_features", action="store_true", default=True,
                       help="保存特征图")
    parser.add_argument("--no_save_features", action="store_true",
                       help="不保存特征图")
    parser.add_argument("--save_visualization", action="store_true", default=True,
                       help="保存可视化")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已有结果")
    parser.add_argument("--json_name", type=str, default="SUNRGBD_train.json",
                       help="JSON 文件名")

    args = parser.parse_args()

    use_sam2_mask = not args.no_sam2_mask
    save_features = not args.no_save_features

    generate_offline_pseudo_labels(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        max_images=args.num_images,
        start_idx=args.start_idx,
        device=args.device,
        use_sam2_mask=use_sam2_mask,
        use_ram_gpt=args.use_ram_gpt,
        depth_scale=args.depth_scale,
        save_features=save_features,
        save_visualization=args.save_visualization,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
