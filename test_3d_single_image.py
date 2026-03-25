#!/usr/bin/env python3
"""
单图片3D检测测试脚本
使用教师模型流水线: MoGe+DepthPro融合深度 + Grounding DINO + SAM2 + 几何拟合
"""
import os
import sys
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'
DATA_ROOT = '/data/ZhaoX/OVM3D-Det/datasets'  # 正确的SUNRGBD数据路径
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino')
sys.path.insert(0, f'{PROJECT_ROOT}/Grounded-SAM-2')
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe/moge")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe")
sys.path.insert(0, "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src")

import torch
from PIL import Image

from detany3d_frontend.depth_predictor.moge_depthpro_fusion import (
    MoGeLoader, DepthProLoader, align_depth_ransac
)
from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance
from cubercnn.generate_label.priors import llm_generated_prior
from cubercnn.util import math_util as util_math


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
        colors = [(0, 255, 0), (0, 200, 0), (200, 0, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[len(pseudo_list) % len(colors)]
        for e in edges:
            pt1 = pts_2d[e[0]].astype(int)
            pt2 = pts_2d[e[1]].astype(int)
            # 只画在图像范围内的线段
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(result, tuple(pt1), tuple(pt2), color, 2)

        # 标注
        cx = int(center[0]/center[2]*K[0,0]+K[0,2])
        cy = int(center[1]/center[2]*K[1,1]+K[1,2])
        cx = np.clip(cx, 0, w-1)
        cy = np.clip(cy, 0, h-1)
        label = f"{pseudo['category_name']} z={center[2]:.2f}m"
        cv2.putText(result, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"3D可视化保存到: {save_path}")

    return result


def draw_2d_boxes(img_rgb, boxes_xyxy, phrases, scores, save_path=None):
    """在图像上绘制2D检测框"""
    result = img_rgb.copy()
    h, w = img_rgb.shape[:2]

    for box, phrase, score in zip(boxes_xyxy, phrases, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = np.clip([x1, x2], 0, w-1)
        y1, y2 = np.clip([y1, y2], 0, h-1)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{phrase} {float(score):.2f}'
        cv2.putText(result, label, (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"2D检测保存到: {save_path}")

    return result


def run_single_image_3d_detection(
    image_index=0,
    device='cuda:1',
    text_prompt="chair. table. bed. sofa. cabinet. desk. door.",
    show_visualization=True,
    output_dir=None,
    use_moge_depthpro=True,
    use_detany3d=False,
):
    """
    单图片3D检测推理 - 教师模型流水线

    Args:
        image_index: SUNRGBD训练集的图片索引
        device: GPU设备
        text_prompt: 2D检测的文本提示
        show_visualization: 是否可视化结果
        output_dir: 输出目录
        use_moge_depthpro: True则使用MoGe+DepthPro融合深度
        use_detany3d: True则使用DetAny3D前端
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'test_output')
    os.makedirs(output_dir, exist_ok=True)

    # ===== 加载SUNRGBD数据 =====
    print("\n" + "="*60)
    print("===== 步骤1: 加载SUNRGBD图片 =====")
    print("="*60)

    json_path = f'{DATA_ROOT}/Omni3D_pl/SUNRGBD_train.json'
    with open(json_path) as f:
        data = json.load(f)

    img_info = data['images'][image_index]
    image_path = f'{DATA_ROOT}/{img_info["file_path"]}'
    K = np.array(img_info['K'], dtype=np.float32)
    img_id = img_info['id']
    h_img, w_img = img_info['height'], img_info['width']

    print(f"图片索引: {image_index}")
    print(f"图片ID: {img_id}")
    print(f"文件路径: {img_info['file_path']}")
    print(f"图片尺寸: {w_img}x{h_img}")
    print(f"内参 K:\n{K}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"图片加载成功: {img_rgb.shape}")

    # ===== 深度估计 =====
    final_depth = None
    if use_moge_depthpro:
        print("\n" + "="*60)
        print("===== 步骤2: MoGe + DepthPro 融合深度估计 =====")
        print("="*60)

        # 1. MoGe
        print("  [MoGe] 加载模型...")
        moge_loader = MoGeLoader()
        moge_loader.device = torch.device(device)
        moge_loader.load_model()
        print("  [MoGe] 推理...")
        moge_result = moge_loader.infer(img_rgb)
        moge_depth = moge_result['depth']
        moge_K = moge_result.get('intrinsics')
        if moge_K is not None:
            moge_K = np.array(moge_K)
            if moge_K.max() < 2.0:
                moge_K = moge_K * np.array([[w_img, 1, w_img], [1, h_img, h_img], [1, 1, 1]])
        else:
            moge_K = K
        print(f"  [MoGe] 深度范围: {moge_depth[moge_depth>0].min():.2f} - {moge_depth[moge_depth>0].max():.2f}m")

        # 2. DepthPro
        print("  [DepthPro] 加载模型...")
        depthpro_loader = DepthProLoader()
        depthpro_loader.device = torch.device(device)
        depthpro_loader.load_model()
        print("  [DepthPro] 推理...")
        dp_result = depthpro_loader.infer(img_rgb, focal_length_px=moge_K[0, 0])
        dp_depth = dp_result['depth']
        dp_f_px = dp_result.get('focallength_px', moge_K[0, 0])
        print(f"  [DepthPro] 深度范围: {dp_depth[dp_depth>0].min():.2f} - {dp_depth[dp_depth>0].max():.2f}m")
        print(f"  [DepthPro] 焦距: {dp_f_px:.2f}")

        # 3. RANSAC对齐 (使用MoGe mask过滤有效点)
        print("  [融合] RANSAC对齐...")
        moge_mask = moge_result.get('mask')
        aligned_depth, diag = align_depth_ransac(
            moge_depth, dp_depth,
            mask=moge_mask,
            max_valid_depth=50.0
        )
        print(f"  [对齐] {diag['status']} | scale={diag['scale']:.4f} "
              f"| 内点率={diag['inlier_ratio']:.1%} | P95误差={diag['p95_error']:.3f}m")
        final_depth = aligned_depth
        print(f"  [融合] 对齐后深度: {aligned_depth[aligned_depth>0].min():.2f} - {aligned_depth[aligned_depth>0].max():.2f}m")

        # 保存深度图
        depth_vis = (aligned_depth / aligned_depth.max() * 255).astype(np.uint8)
        cv2.imwrite(f'{output_dir}/fused_depth.png', depth_vis)
        print(f"  [保存] 深度图 -> {output_dir}/fused_depth.png")

    elif use_detany3d:
        print("\n" + "="*60)
        print("===== 步骤2: DetAny3D 深度估计 =====")
        print("="*60)
        try:
            from teacher_student.teacher_detany3d import TeacherDetAny3D
            print("  [DetAny3D] 加载教师模型...")
            teacher = TeacherDetAny3D(
                device=device,
                use_sam2_mask=True,
                use_ram_gpt=False,
            )
            depth_np, masks, boxes_xyxy, phrases, F_fused = teacher.get_depth_mask_and_boxes(
                img_rgb, text_prompt, K
            )
            if depth_np is not None:
                final_depth = depth_np
                print(f"  [DetAny3D] 深度范围: {depth_np[depth_np>0].min():.2f} - {depth_np[depth_np>0].max():.2f}m")
        except Exception as e:
            print(f"  [DetAny3D] 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("错误: 必须选择一种深度估计方法")
        return None

    if final_depth is None:
        print("错误: 深度估计失败")
        return None

    # ===== 2D检测 =====
    print("\n" + "="*60)
    print("===== 步骤3: Grounding DINO 2D检测 =====")
    print("="*60)

    try:
        from groundingdino.util.inference import load_model as load_gdino
        from groundingdino.util.inference import predict as gdino_predict
        import groundingdino.datasets.transforms as T

        gdino_cfg = f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        gdino_ckpt = f'{PROJECT_ROOT}/Grounded-SAM-2/checkpoints/groundingdino_swint_ogc.pth'
        if not os.path.exists(gdino_ckpt):
            gdino_ckpt = f'{PROJECT_ROOT}/Grounded-SAM-2/grounding_dino/weights/groundingdino_swint_ogc.pth'

        print("  [Grounding DINO] 加载模型...")
        gdino_model = load_gdino(gdino_cfg, gdino_ckpt).to(device)
        gdino_model.eval()

        gdino_transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_tensor, _ = gdino_transform(image_pil, None)

        print(f"  [Grounding DINO] 检测: {text_prompt}")
        boxes_xyxy, scores, phrases = gdino_predict(
            model=gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device,
        )

        if hasattr(boxes_xyxy, 'cpu'):
            boxes_xyxy = boxes_xyxy.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        phrases = list(phrases)

        # Grounding DINO 输出是 [cx, cy, w, h] 格式，需要转换为 [x1, y1, x2, y2]
        boxes_scaled = boxes_xyxy * np.array([w_img, h_img, w_img, h_img])
        boxes_xyxy_orig = np.zeros_like(boxes_scaled)
        boxes_xyxy_orig[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy_orig[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy_orig[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy_orig[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2  # y2 = cy + h/2

        print(f"  [Grounding DINO] 检测到 {len(boxes_xyxy_orig)} 个物体: {phrases}")

    except Exception as e:
        print(f"  [Grounding DINO] 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 保存2D检测结果
    vis2d = draw_2d_boxes(img_rgb, boxes_xyxy_orig, phrases, scores,
                          save_path=f'{output_dir}/2d_detection.png')

    # ===== SAM2 Mask生成 =====
    print("\n" + "="*60)
    print("===== 步骤4: SAM2 Box-Prompted Mask =====")
    print("="*60)

    masks = None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_cfg = f'{PROJECT_ROOT}/Grounded-SAM-2/sam2/configs/sam2/sam2_hiera_s.yaml'
        sam2_ckpt = f'{PROJECT_ROOT}/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt'
        if not os.path.exists(sam2_cfg):
            sam2_cfg = f'{PROJECT_ROOT}/Grounded-SAM-2/sam2/configs/sam2/sam2_hiera_s.yaml'
        if not os.path.exists(sam2_ckpt):
            sam2_ckpt = f'{PROJECT_ROOT}/weights/sam2_hiera_small.pt'

        if os.path.exists(sam2_cfg) and os.path.exists(sam2_ckpt):
            print("  [SAM2] 加载模型...")
            cwd = os.getcwd()
            os.chdir(f'{PROJECT_ROOT}/Grounded-SAM-2')
            try:
                sam2_model = build_sam2("sam2_hiera_s.yaml", sam2_ckpt, device=device)
            finally:
                os.chdir(cwd)

            sam2_predictor = SAM2ImagePredictor(sam2_model)

            print("  [SAM2] 生成Mask...")
            sam2_predictor.set_image(img_rgb)
            xyxy_np = np.array(boxes_xyxy_orig, dtype=np.float64)
            if xyxy_np.ndim == 1:
                xyxy_np = xyxy_np[np.newaxis, :]

            # SAM2 接受像素坐标 (xyxy格式)
            masks, _, _ = sam2_predictor.predict(
                point_coords=None, point_labels=None,
                box=xyxy_np, multimask_output=False,
                normalize_coords=False,  # 使用像素坐标
            )
            if masks.ndim == 4:
                masks = masks[:, 0]
            masks = masks.astype(np.float32)

            # 限制mask在2D框内
            clipped_masks = []
            for mask_i, box_i in zip(masks, boxes_xyxy_orig):
                x1, y1, x2, y2 = np.round(box_i).astype(np.int32)
                h_full, w_full = h_img, w_img
                x1_c = np.clip(x1, 0, max(w_full-1, 0))
                y1_c = np.clip(y1, 0, max(h_full-1, 0))
                x2_c = np.clip(x2, 0, max(w_full-1, 0))
                y2_c = np.clip(y2, 0, max(h_full-1, 0))
                box_mask = np.zeros_like(mask_i, dtype=np.float32)
                if x2_c > x1_c and y2_c > y1_c:
                    box_mask[y1_c:y2_c+1, x1_c:x2_c+1] = 1.0
                clipped_masks.append(mask_i * box_mask)
            masks = np.stack(clipped_masks, axis=0)

            print(f"  [SAM2] 生成了 {len(masks)} 个Mask")
        else:
            print(f"  [SAM2] 配置文件或权重不存在,跳过Mask生成")

    except Exception as e:
        print(f"  [SAM2] 失败: {e}")
        import traceback
        traceback.print_exc()
        print("  -> 将使用简单的框内Mask")

    # ===== 几何反投影与3D框拟合 =====
    print("\n" + "="*60)
    print("===== 步骤5: 几何反投影与3D框拟合 =====")
    print("="*60)

    prior_dict = llm_generated_prior.get("SUNRGBD", {})
    pseudo_list = []

    for i, (box, phrase, score) in enumerate(zip(boxes_xyxy_orig, phrases, scores)):
        x1, y1, x2, y2 = box.astype(int)
        cat_name = phrase.lower().strip()

        if masks is not None and i < len(masks):
            mask = masks[i]
        else:
            mask = np.zeros((h_img, w_img), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0

        print(f"\n  [{i}] 类别: {cat_name}, 2D框: [{x1},{y1},{x2},{y2}]")
        print(f"      Mask像素: {mask.sum():.0f}")

        try:
            center, dims, R, success = run_teacher_pipeline_per_instance(
                depth_np=final_depth,
                mask_np=mask,
                K=K,
                category_name=cat_name,
                prior_dict=prior_dict,
                ground_equ=None,
                use_lshape=False,  # 使用PCA分支（与原版一致）
                depth_scale=1.0,
                use_adaptive_erode=True,
                debug=True,
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
                print(f"      ✅ 成功: center={center.round(2)}, dims={dims.round(2)}, z={center[2]:.2f}m")
            else:
                print(f"      ❌ 拟合失败")

        except Exception as e:
            print(f"      ❌ 异常: {e}")

    print(f"\n" + "="*60)
    print(f"===== 结果汇总 =====")
    print(f"="*60)
    print(f"2D检测: {len(boxes_xyxy_orig)} 个物体")
    print(f"3D框拟合: {len(pseudo_list)} 个成功")

    # ===== 可视化 =====
    if show_visualization and len(pseudo_list) > 0:
        print("\n" + "="*60)
        print("===== 步骤6: 可视化 =====")
        print("="*60)

        # 3D框投影
        vis_3d = draw_3d_boxes(img_rgb, pseudo_list, K,
                                save_path=f'{output_dir}/3d_projected.png')

        # 组合图
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(20, 5))

            ax1 = fig.add_subplot(141)
            ax1.imshow(vis2d)
            ax1.set_title(f'2D Detection ({len(boxes_xyxy_orig)} boxes)', fontsize=11)
            ax1.axis('off')

            ax2 = fig.add_subplot(142)
            vmax = np.percentile(final_depth[final_depth > 0.1], 95)
            ax2.imshow(final_depth, cmap='turbo', vmin=0, vmax=vmax)
            ax2.set_title('Fused Depth (m)', fontsize=11)
            ax2.axis('off')
            cbar = plt.colorbar(ax2.images[0], ax=ax2, fraction=0.03, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            ax3 = fig.add_subplot(143)
            ax3.imshow(vis_3d)
            ax3.set_title(f'3D Boxes ({len(pseudo_list)})', fontsize=11)
            ax3.axis('off')

            # 逐个框信息
            ax4 = fig.add_subplot(144)
            ax4.axis('off')
            info_text = f"Image Index: {image_index}\n"
            info_text += f"2D Detections: {len(boxes_xyxy_orig)}\n"
            info_text += f"3D Boxes: {len(pseudo_list)}\n\n"
            info_text += "3D Results:\n"
            for p in pseudo_list:
                c = p['center_cam']
                d = p['dimensions']
                info_text += f"  {p['category_name']}: "
                info_text += f"z={c[2]:.2f}m "
                info_text += f"dims=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f}]\n"
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('Summary', fontsize=11)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/result_3d_detection.png', dpi=150, bbox_inches='tight')
            print(f"  综合图保存到: {output_dir}/result_3d_detection.png")
            plt.close()
        except ImportError:
            print("  matplotlib未安装,跳过综合图")

    # ===== 保存结果 =====
    result = {
        'image_index': image_index,
        'image_id': img_id,
        'image_path': image_path,
        'K': K.tolist(),
        'boxes_2d': boxes_xyxy_orig.tolist(),
        'phrases': phrases,
        'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
        'pseudo_3d_boxes': pseudo_list,
        'depth_range': [float(final_depth[final_depth>0].min()), float(final_depth[final_depth>0].max())],
    }

    result_path = f'{output_dir}/result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {result_path}")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='单图片3D检测测试')
    parser.add_argument('--index', type=int, default=0, help='SUNRGBD图片索引')
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU设备')
    parser.add_argument('--text', type=str,
                        default='chair. table. bed. sofa. cabinet. desk. door. box.',
                        help='文本提示')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化')
    parser.add_argument('--no-moge', action='store_true', help='不使用MoGe+DepthPro(改用DetAny3D)')
    parser.add_argument('--detany3d', action='store_true', help='使用DetAny3D深度估计')
    args = parser.parse_args()

    use_moge = not args.detany3d and not args.no_moge

    result = run_single_image_3d_detection(
        image_index=args.index,
        device=args.device,
        text_prompt=args.text,
        show_visualization=not args.no_viz,
        output_dir=args.output,
        use_moge_depthpro=use_moge,
        use_detany3d=args.detany3d,
    )

    if result:
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        print(f"检测到 {len(result['pseudo_3d_boxes'])} 个3D框")
        for p in result['pseudo_3d_boxes']:
            c = p['center_cam']
            d = p['dimensions']
            print(f"  - {p['category_name']}: z={c[2]:.2f}m, dims=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f}]")
    else:
        print("\n测试失败!")
        sys.exit(1)
