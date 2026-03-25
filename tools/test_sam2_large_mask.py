#!/usr/bin/env python3
"""
测试 SAM2.1 Large 的 mask 质量。

重要：2D 框必须与当前图像对应（同一分辨率下的像素坐标 xyxy）。
若用 A 图的框去测 B 图，框会“飘”、mask 会像噪点——这不是 SAM 坏了，是输入错了。
"""

import argparse
import sys
import os
import numpy as np
import cv2

PROJECT_ROOT = "/data/ZhaoX/OVM3D-Det-1"
GROUNDED_SAM_DIR = os.path.join(PROJECT_ROOT, "Grounded-SAM-2")
sys.path.insert(0, GROUNDED_SAM_DIR)

# Omni3D SUNRGBD_train 第一张图（与之前 Teacher 调试日志一致）
DEFAULT_IMAGE = os.path.join(
    PROJECT_ROOT,
    "datasets/SUNRGBD/kv2/kinect2data/"
    "000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/image/0000121.jpg",
)

# 来自同一张 0000121.jpg 上 Grounding DINO 的检测框（与日志一致，勿挪到别的图）
# 注意：第二个框非常小（~49×70），SAM 容易失败；默认会按中心扩张到 min_short_side（可用 --no_expand 关闭）
DEFAULT_BOXES_0000121 = np.array(
    [
        [159, 137, 623, 415],  # desk
        [383, 74, 432, 144],   # chair（DINO 很紧，仅椅背一条）
        [306, 69, 357, 149],   # chair
    ],
    dtype=np.float64,
)


def expand_box_min_short_side(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    min_short_side: float,
    pad_ratio: float = 0.12,
) -> tuple:
    """以框中心对称扩张：短边至少 min_short_side，并额外 pad_ratio 比例留白（SAM 更稳）。"""
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    # 仅当框过小时才扩张，避免把整张桌子框也整体放大
    if min(bw, bh) >= min_short_side:
        return float(x1), float(y1), float(x2), float(y2)
    new_w = max(bw * (1.0 + 2.0 * pad_ratio), min_short_side)
    new_h = max(bh * (1.0 + 2.0 * pad_ratio), min_short_side)
    nx1 = cx - new_w * 0.5
    ny1 = cy - new_h * 0.5
    nx2 = cx + new_w * 0.5
    ny2 = cy + new_h * 0.5
    nx1 = float(np.clip(nx1, 0, img_w - 1))
    nx2 = float(np.clip(nx2, 0, img_w - 1))
    ny1 = float(np.clip(ny1, 0, img_h - 1))
    ny2 = float(np.clip(ny2, 0, img_h - 1))
    if nx2 <= nx1:
        nx2 = min(nx1 + 1.0, img_w - 1)
    if ny2 <= ny1:
        ny2 = min(ny1 + 1.0, img_h - 1)
    return nx1, ny1, nx2, ny2


def parse_boxes(s: str) -> np.ndarray:
    """格式: 'x1,y1,x2,y2;x1,y1,x2,y2' 多个框用分号分隔"""
    rows = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [float(x) for x in part.split(",")]
        if len(nums) != 4:
            raise ValueError(f"每框需 4 个数 xyxy，得到: {part}")
        rows.append(nums)
    return np.array(rows, dtype=np.float64)


def test_sam2_large_mask(
    image_path: str,
    boxes: np.ndarray,
    out_dir: str,
    device: str = "cuda",
    min_short_side: float = 0.0,
    pad_ratio: float = 0.12,
):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_ckpt = os.path.join(GROUNDED_SAM_DIR, "checkpoints", "sam2.1_hiera_large.pt")
    if not os.path.isfile(sam2_ckpt):
        sam2_ckpt = os.path.join(PROJECT_ROOT, "weights", "sam2.1_hiera_large.pt")

    print(f">> 图像: {image_path}")
    print(f">> SAM2.1 Large: {sam2_ckpt}")

    sam2_model = build_sam2("facebook/sam2.1-hiera-large", sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f">> 尺寸: {w}x{h}, 框数量: {len(boxes)}")

    # 先 clip 到图像内得到「DINO 框」；过小再扩张得到喂 SAM 的框
    boxes_dino = []
    boxes_sam = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        x1 = float(np.clip(x1, 0, w - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        if x2 <= x1:
            x2 = min(x1 + 1, w - 1)
        if y2 <= y1:
            y2 = min(y1 + 1, h - 1)
        dx1, dy1, dx2, dy2 = x1, y1, x2, y2
        if min_short_side > 0:
            x1, y1, x2, y2 = expand_box_min_short_side(
                x1, y1, x2, y2, w, h, min_short_side=min_short_side, pad_ratio=pad_ratio
            )
            if (x1, y1, x2, y2) != (dx1, dy1, dx2, dy2):
                print(
                    f">> 框 {i + 1} 过小已扩张: DINO { [int(round(dx1)), int(round(dy1)), int(round(dx2)), int(round(dy2))] } "
                    f"-> SAM { [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))] }"
                )
        boxes_dino.append([dx1, dy1, dx2, dy2])
        boxes_sam.append([x1, y1, x2, y2])
    boxes = np.array(boxes_sam, dtype=np.float64)
    boxes_dino = np.array(boxes_dino, dtype=np.float64)

    os.makedirs(out_dir, exist_ok=True)
    predictor.set_image(rgb)

    for i, box in enumerate(boxes):
        print(f"\n--- 框 {i + 1}: {box} ---")

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[np.newaxis, :],
            multimask_output=True,
            normalize_coords=False,
        )

        def mask_area(j):
            mj = masks[j]
            mf = mj[0] if mj.ndim == 3 else mj
            return int(np.sum(mf > 0.5))

        # 优先高分；若分数都接近 0（小框常见），改选非空且面积最大的 mask
        best_j = int(np.argmax(scores))
        if float(scores[best_j]) < 1e-4:
            areas = [mask_area(j) for j in range(len(masks))]
            nonempty = [j for j, a in enumerate(areas) if a > 0]
            if nonempty:
                best_j = max(nonempty, key=lambda j: areas[j])
                print(
                    f"  [多Mask] 分数均≈0，按面积选 mask#{best_j + 1}, "
                    f"score={float(scores[best_j]):.4f}, 点数={areas[best_j]}"
                )
            else:
                print(
                    f"  [多Mask] 选分数最高 mask#{best_j + 1}, score={float(scores[best_j]):.4f}, 点数=0"
                )
        else:
            print(
                f"  [多Mask] 选分数最高 mask#{best_j + 1}, score={float(scores[best_j]):.4f}, "
                f"点数={mask_area(best_j)}"
            )

        m_best = masks[best_j]
        m_flat = m_best[0] if m_best.ndim == 3 else m_best
        mask_points = int(np.sum(m_flat > 0.5))

        vis_img = rgb.copy()
        x1, y1, x2, y2 = map(int, box)
        # 绿框：SAM 实际使用的框（可能已扩张）
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 若扩张过：绿色=喂 SAM 的框；黄色细线=clip 后的原始 DINO 框
        d = boxes_dino[i]
        dx1, dy1, dx2, dy2 = int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))
        if (dx1, dy1, dx2, dy2) != (x1, y1, x2, y2):
            cv2.rectangle(vis_img, (dx1, dy1), (dx2, dy2), (255, 255, 0), 1)
            cv2.putText(
                vis_img,
                "yellow=DINO clipped",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
        mask_vis = (m_flat > 0.5).astype(np.uint8) * 200
        overlay = np.zeros_like(vis_img)
        overlay[:, :, 1] = mask_vis
        vis_img = cv2.addWeighted(vis_img, 0.65, overlay, 0.35, 0)
        cv2.putText(
            vis_img,
            f"box{i + 1} score={float(scores[best_j]):.3f} pts={mask_points}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        out_path = os.path.join(out_dir, f"aligned_box_{i + 1}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"  >> 保存: {out_path}")


def main():
    p = argparse.ArgumentParser(description="SAM2.1 Large mask 测试（图像与框必须匹配）")
    p.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="RGB 图像路径")
    p.add_argument(
        "--boxes",
        type=str,
        default=None,
        help="自定义框，格式 x1,y1,x2,y2;x1,y1,x2,y2；不设则用 0000121 默认框",
    )
    p.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "test_sam21_results"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--min_short_side",
        type=float,
        default=100.0,
        help="短边小于该值时从中心扩张后再喂 SAM（0 表示不扩张，与原始 DINO 框一致）",
    )
    p.add_argument("--no_expand", action="store_true", help="等价于 --min_short_side 0")
    p.add_argument(
        "--pad_ratio",
        type=float,
        default=0.12,
        help="扩张时在宽高上额外加的比例留白（相对原框）",
    )
    args = p.parse_args()

    if args.boxes:
        boxes = parse_boxes(args.boxes)
    else:
        # 若换了图但仍用默认框，容易错位——仅在与默认图一致时安全
        if os.path.abspath(args.image) != os.path.abspath(DEFAULT_IMAGE):
            print(
                "⚠️  警告: 你换了 --image 但未提供 --boxes。"
                "默认框只适用于 Omni3D 第一张 0000121.jpg；否则请用 --boxes 传入该图上的真实 xyxy。"
            )
        boxes = DEFAULT_BOXES_0000121

    min_side = 0.0 if args.no_expand else args.min_short_side
    test_sam2_large_mask(
        args.image,
        boxes,
        args.out_dir,
        device=args.device,
        min_short_side=min_side,
        pad_ratio=args.pad_ratio,
    )


if __name__ == "__main__":
    main()
