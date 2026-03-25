"""
对比测试：SAM2 Mask 生成策略

目标：验证以下两个改进对 mask 质量的影响
1. multimask_output=True + 按面积选最大
2. 不做 _restrict_masks_to_boxes (允许 mask 溢出检测框)

用法:
    python tools/test_sam2_mask_comparison.py --img_idx 121
"""
import os, sys, cv2, numpy as np, argparse, json

PROJECT_ROOT = "/data/ZhaoX/OVM3D-Det-1"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Grounded-SAM-2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Grounded-SAM-2", "grounding_dino"))

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T

def estimate_K(width, height, hfov_deg=60.0):
    f = 0.5 * width / np.tan(np.rad2deg(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float32)

def load_sunrgbd_sample(img_idx, dataset_root="/data/ZhaoX/OVM3D-Det-1/datasets"):
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
        raise FileNotFoundError(f"JSON not found in {json_paths}")

    with open(json_path) as f:
        data = json.load(f)

    img_info = data["images"][img_idx]
    file_path = os.path.join(dataset_root, img_info["file_path"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")

    return {
        "path": file_path,
        "K": np.array(img_info["K"], dtype=np.float32) if "K" in img_info else None,
        "image_id": img_info["id"],
        "width": img_info["width"],
        "height": img_info["height"],
    }

def run_gdino(rgb_np, model, transform, text_prompt, box_threshold=0.3, text_threshold=0.25):
    h, w = rgb_np.shape[:2]
    img_pil = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    img_pil = cv2.resize(img_pil, (w, h))
    img_pil = Image.fromarray(img_pil)
    img_tensor, _ = transform(img_pil, None)

    boxes, logits, phrases = gdino_predict(
        model=model,
        image=img_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cuda:1",
    )

    boxes = boxes.cpu() if boxes is not None else None
    return boxes, phrases

def restrict_mask_to_box(mask, box_xyxy):
    """_restrict_masks_to_boxes 的单 mask 版本"""
    x1, y1, x2, y2 = np.round(box_xyxy).astype(np.int32)
    h, ww = mask.shape[:2]
    x1, x2 = int(np.clip(x1, 0, max(ww - 1, 0))), int(np.clip(x2, 0, max(ww - 1, 0)))
    y1, y2 = int(np.clip(y1, 0, max(h - 1, 0))), int(np.clip(y2, 0, max(h - 1, 0)))

    box_mask = np.zeros_like(mask, dtype=np.float32)
    if x2 > x1 and y2 > y1:
        box_mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    return mask * box_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_idx", type=int, default=121, help="SUNRGBD image index")
    parser.add_argument("--gdino_prompt", type=str,
        default="bed. chair. table. sofa. cabinet. desk. bookshelf. door. pillow. monitor.",
        help="Grounding DINO text prompt")
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    args = parser.parse_args()

    # 加载数据
    sample = load_sunrgbd_sample(args.img_idx)
    img_path = sample["path"]
    rgb = cv2.imread(img_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    K = sample["K"] if sample["K"] is not None else estimate_K(w, h)
    print(f"图像: {os.path.basename(img_path)}, 尺寸: {w}x{h}")

    # 加载 SAM2
    print("\n加载 SAM2.1 Large...")
    sam2_ckpt = os.path.join(PROJECT_ROOT, "Grounded-SAM-2", "checkpoints", "sam2.1_hiera_large.pt")
    sam2_model = build_sam2("facebook/sam2.1-hiera-large", sam2_ckpt, device="cuda:1")
    predictor = SAM2ImagePredictor(sam2_model)

    # 加载 Grounding DINO
    print("加载 Grounding DINO...")
    gdino_cfg = os.path.join(PROJECT_ROOT, "Grounded-SAM-2", "grounding_dino", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    gdino_ckpt = os.path.join(PROJECT_ROOT, "Grounded-SAM-2", "checkpoints", "groundingdino_swint_ogc.pth")
    gdino_model = load_gdino(gdino_cfg, gdino_ckpt, device="cuda:1")
    gdino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # DINO 检测
    boxes, phrases = run_gdino(rgb, gdino_model, gdino_transform, args.gdino_prompt,
                                 args.box_thresh, args.text_thresh)
    if boxes is None or len(boxes) == 0:
        print("DINO 未检测到任何物体！")
        return
    print(f"\nDINO 检测到 {len(boxes)} 个物体:")
    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        x1, y1, x2, y2 = box.tolist()
        print(f"  [{i}] {phrase}: box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), "
              f"面积={int((x2-x1)*(y2-y1))} px²")

    # SAM2 推理
    predictor.set_image(rgb)

    # 策略 A: multimask_output=False (当前 teacher 的做法)
    print(f"\n{'='*60}")
    print("策略 A: multimask_output=False (Teacher 当前做法)")
    print(f"{'='*60}")
    masks_A, scores_A, _ = predictor.predict(
        box=boxes.numpy(), multimask_output=False, normalize_coords=False
    )
    if masks_A.ndim == 4:
        masks_A = masks_A[:, 0]
    print(f"返回 mask 数量: {len(masks_A)}")
    for i, (m, s) in enumerate(zip(masks_A, scores_A)):
        pts = int(np.sum(m > 0.5))
        print(f"  Mask {i}: 面积={pts:>7} px ({pts/(h*w)*100:.3f}%), score={s:.6f}")

    # 策略 B: multimask_output=True + 按面积选最大
    print(f"\n{'='*60}")
    print("策略 B: multimask_output=True + 按面积选最大 (改进方案)")
    print(f"{'='*60}")
    masks_B, scores_B, _ = predictor.predict(
        box=boxes.numpy(), multimask_output=True, normalize_coords=False
    )
    if masks_B.ndim == 4:
        masks_B = masks_B[:, 0]
    print(f"返回 mask 总数: {len(masks_B)} (每框3个)")
    best_idx = np.argmax([int(np.sum(m > 0.5)) for m in masks_B])
    best_mask_B = masks_B[best_idx]
    best_pts = int(np.sum(best_mask_B > 0.5))
    best_score = scores_B[best_idx]
    print(f"  最佳 mask idx={best_idx}: 面积={best_pts:>7} px ({best_pts/(h*w)*100:.3f}%), score={best_score:.6f}")

    # 对比展示
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        x1, y1, x2, y2 = box.tolist()
        m_A = masks_A[i] if i < len(masks_A) else None
        pts_A = int(np.sum(m_A > 0.5)) if m_A is not None else 0

        m_A_clipped = restrict_mask_to_box(masks_A[i], box) if m_A is not None else None
        pts_A_clipped = int(np.sum(m_A_clipped > 0.5)) if m_A_clipped is not None else 0

        print(f"\n  [{i}] {phrase}")
        print(f"      DINO 框: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), 框面积={int((x2-x1)*(y2-y1))} px²")
        print(f"      策略A (multimask=False):  面积={pts_A:>7} px, clip后={pts_A_clipped:>7} px")

        # 每框的3个候选中找最大的
        base = i * 3
        sub_masks = masks_B[base:base+3]
        sub_scores = scores_B[base:base+3]
        areas = [int(np.sum(m > 0.5)) for m in sub_masks]
        best_local = int(np.argmax(areas))
        best_local_pts = areas[best_local]
        best_local_score = sub_scores[best_local]
        print(f"      策略B (multimask=True+面积选):  面积={best_local_pts:>7} px, score={best_local_score:.4f}")
        print(f"        3个候选: {areas}")
        improvement = best_local_pts - pts_A_clipped
        print(f"      提升: {improvement:+d} px ({improvement/max(pts_A_clipped,1)*100:+.1f}%)")

    # 生成对比图
    vis_dir = os.path.join(PROJECT_ROOT, "test_sam21_results", "comparison")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]

        # 策略 A 原始
        m_A_raw = masks_A[i] if i < len(masks_A) else None
        # 策略 A clip 后
        m_A_clipped = restrict_mask_to_box(masks_A[i], box) if m_A is not None else None
        # 策略 B 最佳
        base = i * 3
        sub_masks = masks_B[base:base+3]
        best_B = sub_masks[int(np.argmax([int(np.sum(m > 0.5)) for m in sub_masks]))]

        def make_grid(images, titles, ncols=3, thumb_h=200):
            rows = []
            for row_imgs, row_titles in zip(
                [images[j::ncols] for j in range(ncols)],
                [titles[j::ncols] for j in range(ncols)]
            ):
                row = []
                for img, title in zip(row_imgs, row_titles):
                    if img is None:
                        h_i, w_i = thumb_h, int(thumb_h * w / h)
                        img_disp = np.zeros((h_i, w_i, 3), dtype=np.uint8)
                    else:
                        img_disp = cv2.resize(img, (int(thumb_h * w / h), thumb_h))
                        if img_disp.ndim == 2:
                            img_disp = np.stack([img_disp] * 3, axis=-1)
                    cv2.putText(img_disp, title, (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    row.append(img_disp)
                rows.append(np.hstack(row))
            return np.vstack(rows) if len(rows) > 1 else rows[0] if rows else None

        overlay_A = rgb.copy()
        overlay_B = rgb.copy()
        overlay_A_clipped = rgb.copy()
        if m_A_raw is not None:
            color = np.array([0, 255, 0], dtype=np.uint8)
            overlay_A[m_A_raw > 0.5] = (0.6 * overlay_A[m_A_raw > 0.5] + 0.4 * color).astype(np.uint8)
        if m_A_clipped is not None:
            color = np.array([0, 255, 255], dtype=np.uint8)
            overlay_A_clipped[m_A_clipped > 0.5] = (0.6 * overlay_A_clipped[m_A_clipped > 0.5] + 0.4 * color).astype(np.uint8)
        if best_B is not None:
            color = np.array([255, 0, 0], dtype=np.uint8)
            overlay_B[best_B > 0.5] = (0.6 * overlay_B[best_B > 0.5] + 0.4 * color).astype(np.uint8)

        cv2.rectangle(overlay_A, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(overlay_A_clipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(overlay_B, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{phrase} [box{i}]"
        for img, txt in [(overlay_A, "A:raw"), (overlay_A_clipped, "A:clipped"), (overlay_B, "B:bestArea")]:
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        grid = make_grid(
            [overlay_A, overlay_A_clipped, overlay_B,
             (m_A_raw * 255).astype(np.uint8) if m_A_raw is not None else None,
             (m_A_clipped * 255).astype(np.uint8) if m_A_clipped is not None else None,
             (best_B * 255).astype(np.uint8) if best_B is not None else None],
            ["A_raw", "A_clipped", "B_bestArea",
             f"pts={int(np.sum(m_A_raw>0.5))}" if m_A_raw is not None else "N/A",
             f"pts={int(np.sum(m_A_clipped>0.5))}" if m_A_clipped is not None else "N/A",
             f"pts={int(np.sum(best_B>0.5))}" if best_B is not None else "N/A"],
            ncols=3
        )
        out_path = os.path.join(vis_dir, f"comparison_{os.path.basename(img_path).split('.')[0]}_box{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\n  保存对比图: {out_path}")

if __name__ == "__main__":
    main()
