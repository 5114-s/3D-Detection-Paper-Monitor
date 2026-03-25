#!/usr/bin/env python3
"""
单图片3D检测测试
使用教师模型(MoGe+DepthPro融合深度 + 几何拟合)生成伪3D框
"""
import cv2
import numpy as np
import sys
import os
import json

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

import torch

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import (
    MoGeLoader, DepthProLoader, align_depth_ransac
)
from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance


def get_image_from_omni3d_json(json_path, index=0):
    """从 Omni3D JSON 中获取图片信息"""
    with open(json_path) as f:
        data = json.load(f)
    img_info = data['images'][index]
    return img_info, data


def draw_3d_boxes(img_rgb, pseudo_list, K, save_path=None):
    """在图像上绘制3D框的2D投影"""
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()

    for pseudo in pseudo_list:
        center = np.array(pseudo['center_cam'])
        dims = np.array(pseudo['dimensions'])
        R = np.array(pseudo['R_cam']) if pseudo['R_cam'] is not None else np.eye(3)

        # 画3D框的8个角点
        d = dims / 2
        corners_local = np.array([
            [-d[0], -d[1], -d[2]], [d[0], -d[1], -d[2]],
            [d[0], d[1], -d[2]], [-d[0], d[1], -d[2]],
            [-d[0], -d[1], d[2]], [d[0], -d[1], d[2]],
            [d[0], d[1], d[2]], [-d[0], d[1], d[2]],
        ])
        corners_world = (R @ corners_local.T).T + center

        # 投影到2D
        pts_2d = (K @ corners_world.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

        # 画边
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            pt1 = pts_2d[e[0]].astype(int)
            pt2 = pts_2d[e[1]].astype(int)
            cv2.line(result, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

        # 标注
        cx, cy = int(center[0]/center[2]*K[0,0]+K[0,2]), int(center[1]/center[2]*K[1,1]+K[1,2])
        cv2.putText(result, pseudo['category_name'], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"3D可视化保存到: {save_path}")

    return result


def run_single_image_3d_detection(
    image_path=None,
    depth_path=None,
    K=None,
    text_prompt="chair. table. bed. sofa. cabinet.",
    show_visualization=True,
    device='cuda:1',
):
    """
    单图片3D检测推理

    Args:
        image_path: 图片路径（可选，默认用 Omni3D JSON 第一张）
        depth_path: 深度图路径（可选）
        K: 相机内参矩阵 (3,3)，可选
        text_prompt: 2D检测的文本提示
        show_visualization: 是否可视化结果
        device: GPU设备
    """
    # 如果没有提供路径，从 Omni3D JSON 获取
    if image_path is None:
        json_path = f'{PROJECT_ROOT}/datasets/Omni3D_pl/SUNRGBD_train.json'
        img_info, data = get_image_from_omni3d_json(json_path, index=0)
        image_path = f'{PROJECT_ROOT}/datasets/{img_info["file_path"]}'
        depth_path = image_path.replace('/image/', '/depth/').replace('.jpg', '.png')
        K = np.array(img_info['K'], dtype=np.float32)
        print(f"图片: {img_info['file_path']}")
        print(f"深度: {depth_path}")
        print(f"内参 K:\n{K}")

    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"\n图片尺寸: {w}x{h}")

    # 加载GT深度（用于对比）
    gt_depth = None
    if depth_path and os.path.exists(depth_path):
        gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        print(f"GT深度范围: {gt_depth[gt_depth>0].min():.2f}m - {gt_depth[gt_depth>0].max():.2f}m")

    # ===== 深度估计 =====
    print("\n===== 深度估计 =====")

    # 1. MoGe 推理
    print("运行 MoGe...")
    moge_loader = MoGeLoader()
    moge_loader.device = torch.device(device)
    moge_loader.load_model()
    moge_result = moge_loader.infer(img_rgb)
    moge_depth = moge_result['depth']
    moge_K = moge_result['intrinsics']
    if K is None:
        K = np.asarray(moge_K, dtype=np.float32)
        print("  未提供 K，使用 MoGe 估计内参")
    print(f"  MoGe深度: {moge_depth[moge_depth>0].min():.2f}-{moge_depth[moge_depth>0].max():.2f}m")
    print(f"  MoGe内参: fx={moge_K[0,0]:.2f}")

    # 2. DepthPro 推理（显存不足或缺权重时退化为仅 MoGe）
    print("运行 DepthPro...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    depthpro_loader = DepthProLoader()
    depthpro_loader.device = torch.device(device)
    dp_result = depthpro_loader.infer(img_rgb, focal_length_px=float(moge_K[0, 0]))

    if dp_result is None:
        print("  DepthPro 不可用，跳过 MoGe+DepthPro 融合，直接使用 MoGe 深度")
        aligned_depth = np.asarray(moge_depth, dtype=np.float32)
        diag = {
            "status": "moge_only",
            "scale": 1.0,
            "inlier_ratio": 1.0,
            "p95_error": 0.0,
        }
    else:
        dp_depth = dp_result["depth"]
        dp_f_px = dp_result["focallength_px"]
        print(f"  DepthPro深度: {dp_depth[dp_depth>0].min():.2f}-{dp_depth[dp_depth>0].max():.2f}m")
        print(f"  DepthPro焦距: {dp_f_px:.2f}")
        print("RANSAC对齐...")
        aligned_depth, diag = align_depth_ransac(moge_depth, dp_depth, max_valid_depth=50.0)
    print(f"  [对齐] {diag['status']} | scale={diag['scale']:.4f} "
          f"| 内点率={diag['inlier_ratio']:.1%} | P95误差={diag['p95_error']:.3f}m")
    print(f"  对齐后深度: {aligned_depth[aligned_depth>0].min():.2f}-{aligned_depth[aligned_depth>0].max():.2f}m")

    if gt_depth is not None:
        valid = (gt_depth > 0.1) & (aligned_depth > 0.1)
        abs_rel = np.mean(np.abs(aligned_depth[valid] - gt_depth[valid]) / gt_depth[valid]) * 100
        thres125 = np.mean(np.maximum(aligned_depth[valid]/gt_depth[valid], gt_depth[valid]/aligned_depth[valid]) < 1.25) * 100
        print(f"  深度误差: AbsRel={abs_rel:.2f}%, Th@1.25={thres125:.2f}%")

    final_depth = aligned_depth

    # ===== 2D 检测 =====
    print("\n===== 2D检测 (Grounding DINO) =====")
    boxes_xyxy = np.array([]).reshape(0, 4)
    scores = np.array([])
    phrases = []

    # 清理显存，确保 Grounding DINO 有足够空间
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        from groundingdino.util.inference import load_model as load_gdino
        from groundingdino.util.inference import predict as gdino_predict
        from groundingdino.util.inference import Model as GDinoModel

        gdino_ckpt = f'{PROJECT_ROOT}/Grounded-SAM-2/checkpoints/groundingdino_swint_ogc.pth'
        if not os.path.exists(gdino_ckpt):
            gdino_ckpt = f'{PROJECT_ROOT}/weights/groundingdino_swinb_cogcoor.pth'
        if not os.path.exists(gdino_ckpt):
            gdino_ckpt = f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino/weights/groundingdino_swint_ogc.pth'

        if os.path.exists(gdino_ckpt):
            print(f"加载 Grounding DINO: {gdino_ckpt}")
            # 使用新版 Model API，避免手动预处理 + 避免重复 .to(device)
            gdino_model = GDinoModel(
                model_config_path=f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                model_checkpoint_path=gdino_ckpt,
                device=device,
            )

            # predict_with_caption 内部已做好图像预处理
            detections, phrases = gdino_model.predict_with_caption(
                image=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                caption=text_prompt,
                box_threshold=0.3,
                text_threshold=0.25,
            )
            boxes_xyxy = detections.xyxy
            scores = detections.confidence
            print(f"检测到 {len(boxes_xyxy)} 个物体: {phrases}")
        else:
            print(f"Grounding DINO 权重不存在")

    except Exception as e:
        import traceback
        print(f"Grounding DINO 加载失败: {e}")
        traceback.print_exc()

    # ===== 几何拟合 =====
    print("\n===== 3D框几何拟合 =====")

    DIM_PRIORS = {
        'chair': [0.60, 0.60, 0.90],
        'table': [1.20, 0.60, 0.75],
        'bed': [2.00, 1.50, 0.50],
        'sofa': [1.80, 0.80, 0.80],
        'cabinet': [0.80, 0.50, 0.85],
        'door': [0.90, 0.05, 2.10],
        'sink': [0.60, 0.50, 0.30],
        'toilet': [0.50, 0.65, 0.80],
        'bathtub': [1.70, 0.70, 0.50],
        'box': [0.50, 0.50, 0.50],
        'bookcase': [0.80, 0.30, 1.50],
        'counter': [1.50, 0.60, 0.90],
        'desk': [1.20, 0.60, 0.75],
        'shelf': [0.80, 0.30, 1.80],
        'picture': [0.80, 0.05, 0.60],
        'lamp': [0.40, 0.40, 1.50],
        'person': [0.50, 0.30, 1.70],
    }

    pseudo_list = []

    for i, (box, phrase, score) in enumerate(zip(boxes_xyxy, phrases, scores)):
        x1, y1, x2, y2 = box.astype(int)
        cat_name = phrase.lower().strip()
        prior = DIM_PRIORS.get(cat_name, [0.5, 0.5, 0.5])

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        try:
            center, dims, R, success = run_teacher_pipeline_per_instance(
                depth=final_depth,
                mask=mask,
                K=K,
                category_name=cat_name,
                prior_dict={'mean': prior, 'std': [0.1, 0.1, 0.1]},
            )

            if success and center is not None:
                pseudo = {
                    'category_name': cat_name,
                    'score': float(score),
                    'bbox2D': [int(x1), int(y1), int(x2), int(y2)],
                    'center_cam': center.tolist(),
                    'dimensions': dims.tolist(),
                    'R_cam': R.tolist() if R is not None else None,
                    'success': True,
                }
                pseudo_list.append(pseudo)
                print(f"  [{i}] {cat_name}: center={center.round(2)}, dims={dims.round(2)}, z={center[2]:.2f}m")
        except Exception as e:
            print(f"  [{i}] {cat_name}: 拟合失败 - {e}")

    print(f"\n===== 结果 =====")
    print(f"检测到 {len(pseudo_list)}/{len(boxes_xyxy)} 个3D框")

    # ===== 可视化 =====
    if show_visualization and len(pseudo_list) > 0:
        print("\n生成可视化...")
        # 3D框投影到2D图像
        vis_img = draw_3d_boxes(img_rgb, pseudo_list, K,
                                save_path=f'{PROJECT_ROOT}/test_3d_projected.png')

        # 也画2D检测结果
        vis2d = img_rgb.copy()
        for box, phrase, score in zip(boxes_xyxy, phrases, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis2d, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis2d, f'{phrase} {score:.2f}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(f'{PROJECT_ROOT}/test_2d_detection.png',
                    cv2.cvtColor(vis2d, cv2.COLOR_RGB2BGR))

        # 深度图
        depth_vis = (aligned_depth / aligned_depth.max() * 255).astype(np.uint8)
        cv2.imwrite(f'{PROJECT_ROOT}/test_fused_depth.png', depth_vis)

        print(f"  2D检测图: {PROJECT_ROOT}/test_2d_detection.png")
        print(f"  3D投影图: {PROJECT_ROOT}/test_3d_projected.png")
        print(f"  融合深度图: {PROJECT_ROOT}/test_fused_depth.png")

        # matplotlib 3D图
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(18, 5))
            ax1 = fig.add_subplot(131)
            ax1.imshow(vis2d)
            ax1.set_title(f'2D Detection ({len(boxes_xyxy)} boxes)')
            ax1.axis('off')

            ax2 = fig.add_subplot(132)
            vmax = np.percentile(aligned_depth[aligned_depth > 0.1], 95)
            ax2.imshow(aligned_depth, cmap='turbo', vmin=0, vmax=vmax)
            ax2.set_title('Fused Depth (m)')
            ax2.axis('off')

            ax3 = fig.add_subplot(133)
            ax3.imshow(vis_img)
            ax3.set_title(f'3D Boxes ({len(pseudo_list)})')
            ax3.axis('off')

            plt.tight_layout()
            plt.savefig(f'{PROJECT_ROOT}/test_3d_result.png', dpi=150, bbox_inches='tight')
            print(f"  综合图: {PROJECT_ROOT}/test_3d_result.png")
            plt.close()
        except ImportError:
            pass

    return {
        'boxes_2d': boxes_xyxy.tolist(),
        'phrases': list(phrases),
        'scores': scores.tolist(),
        'pseudo_3d_boxes': pseudo_list,
        'depth': final_depth,
        'K': K.tolist(),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='单图片3D检测')
    parser.add_argument('--image', type=str, default=None, help='图片路径')
    parser.add_argument('--depth', type=str, default=None, help='深度图路径')
    parser.add_argument('--text', type=str, default='chair. table. bed. sofa. cabinet.', help='文本提示')
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU设备')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化')
    args = parser.parse_args()

    run_single_image_3d_detection(
        image_path=args.image,
        depth_path=args.depth,
        text_prompt=args.text,
        show_visualization=not args.no_viz,
        device=args.device,
    )
